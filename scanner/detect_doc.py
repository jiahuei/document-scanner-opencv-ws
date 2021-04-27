# -*- coding: utf-8 -*-
"""
Created on 07 Mar 2021 14:49:38
@author: jiahuei
"""
import os
import json
import numpy as np
import cv2
import logging
from scanner import utils
from scanner.utils import image as image_utils
from scanner.utils.misc import get

logger = logging.getLogger(__name__)


# noinspection PyPep8Naming
# noinspection PyAttributeOutsideInit
class _Detector:
    DOC_ASPECT_RATIO: float
    DOC_TYPE: str
    REF: str
    REF_RESIZE_WIDTH: int

    def __init__(self, **kwargs):
        self.doc_extract_height = kwargs.get("doc_extract_height", 224)
        self.doc_extract_width = kwargs.get("doc_extract_width", 355)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Adaptive Histogram Equalization

        # ### Simple thresholding pipeline ###
        self.grayscale_method = kwargs.get("grayscale_method", "hue")
        self.threshold_method = kwargs.get("threshold_method", "otsu")

        # ### Image features pipeline ###
        self.grayscale_mode = kwargs.get("grayscale_mode", True)
        self.keypoint_detector, self.keypoint_matcher = image_utils.get_keypoint_detector_and_matcher(
            detector_name=kwargs.get("detector_name", "sift"),
            matcher_name=kwargs.get("matcher_name", "bf"),
            affine_invariant=kwargs.get("affine_invariant", False),
            affine_params=kwargs.get("affine_params", None),
        )
        self.compute_homography = image_utils.ComputeHomography(
            matcher=self.keypoint_matcher,
            match_ratio=kwargs.get("match_ratio", 0.8),
            min_match_count=kwargs.get("min_match_count", 40),
            top_k_keypoints=kwargs.get("top_k_keypoints", 50),
            homography_method=kwargs.get("homography_method", cv2.LMEDS),
            reproj_threshold=kwargs.get("reproj_threshold", 5.0),
        )
        ref_image = image_utils.read_image(
            os.path.join(self.get_ref_dir(), self.REF), grayscale=self.grayscale_mode
        )
        ref_image, _ = image_utils.resize_image(
            ref_image, max_side=self.REF_RESIZE_WIDTH
        )
        self.ref_image = self._preprocess_image(ref_image)
        self.g_keypoints, self.g_descriptors = self._detect_and_compute(
            self.ref_image, mask=None
        )
        self.ref_height, self.ref_width = self.ref_image.shape[0], self.ref_image.shape[1]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_ref_dir(cls):
        return os.path.join(utils.misc.REPO_DIR, "scanner", "resources")

    def detect_doc_simple(self, query_image):
        h, w = query_image.shape[:2]
        if self.grayscale_method == "hue":
            query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2HSV)[..., 0]
            end_val = 50  # Split off at green hue
            query_image = gray = np.uint8(np.where(
                query_image >= end_val, query_image - end_val, query_image + (180 - end_val)
            ))
        else:
            query_image = gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        query_image = image_utils.otsu_thresholding(
            query_image,
            gauss_blur_ksize=self._get_kernel_size(h, w, 5 / 480),
            med_blur_pre_ksize=self._get_kernel_size(h, w, 5 / 480),
            med_blur_post_ksize=self._get_kernel_size(h, w, 5 / 480),
        )[0]
        # Take a sample from center of image after thresholding
        # If average value is less than 128, means background is darker than subject
        # In this case, flip the image
        nrow, ncol = query_image.shape[:2]
        center_sample = query_image[int(nrow * 0.3):int(nrow * 0.6), int(ncol * 0.3):int(ncol * 0.6)]
        if np.mean(center_sample) < 128:
            query_image = np.uint8(np.where(query_image > 0, 0, 255))
        bbox = image_utils.generate_bbox(query_image)
        if bbox is not None:
            # Expand to (4, 1, 2)
            bbox = np.expand_dims(bbox, axis=1)
        return {"dst_points": bbox, "debug": {"img_threshold": query_image}}

    @staticmethod
    def _get_kernel_size(qh, qw, factor):
        max_input_side = max(qh, qw)
        ksize = max(3, utils.misc.round_to_nearest_odd(factor * max_input_side))
        return ksize

    def detect_doc_features(self, query_image):
        query_image = self._preprocess_image(query_image)
        query_image = self.clahe.apply(query_image)
        q_keypoints, q_descriptors = self._detect_and_compute(query_image, mask=None)
        results = self._compute_homography(q_keypoints, q_descriptors)
        M = results["homography"]
        if M is None:
            dst_pts = None
        else:
            h, w = self.ref_height, self.ref_width
            src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst_pts = cv2.perspectiveTransform(src_pts, M)
        return {"dst_points": dst_pts, "debug": results}

    @utils.misc.time_function
    def _detect_and_compute(self, image, mask=None):
        # Running `detect` followed by `compute` is slower than `detectAndCompute`
        return self.keypoint_detector.detectAndCompute(image, mask=mask)

    @utils.misc.time_function
    def _compute_homography(self, q_keypoints, q_descriptors):
        return self.compute_homography(
            q_keypoints, q_descriptors, self.g_keypoints, self.g_descriptors
        )

    @utils.misc.time_function
    def extract_doc(self, corner_points, query_image):
        self._image_type_check(query_image)
        try:
            doc_corners = image_utils.order_points(corner_points)
            # Filter small detections
            doc_area = image_utils.polygon_area(doc_corners[:, 0], doc_corners[:, 1])
            if doc_area < (query_image.size * 0.02):
                raise ValueError
        except (cv2.error, ValueError, AttributeError):
            doc = doc_corners = None
        else:
            doc = image_utils.four_point_transform(
                query_image, doc_corners,
                output_height=self.doc_extract_height, output_width=self.doc_extract_width
            )
        return doc, doc_corners

    def _preprocess_image(self, image):
        self._image_type_check(image)
        if image.ndim == 3 and image.shape[2] == 3 and self.grayscale_mode:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def _image_type_check(image):
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected `image` of type `np.ndarray`, saw {type(image)}"
            )
        return True


# noinspection PyPep8Naming
# noinspection PyAttributeOutsideInit
class A4Detector(_Detector):
    DOC_ASPECT_RATIO = 210 / 297
    DOC_TYPE = "A4"
    REF = "W-8BEN.png"
    REF_RESIZE_WIDTH = 1024

    def __init__(self, use_image_features=False, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(use_image_features, bool):
            raise TypeError(
                f"Expected `use_image_features` of type `bool`, saw {type(use_image_features)}"
            )
        self.use_image_features = use_image_features

    @utils.misc.time_function
    def detect_doc(self, query_image):
        if self.use_image_features:
            return self.detect_doc_features(query_image)
        else:
            return self.detect_doc_simple(query_image)

    def __call__(self, query_image, debug=False):
        self._image_type_check(query_image)
        if query_image.ndim != 3 or (query_image.ndim == 3 and query_image.shape[2] != 3):
            raise ValueError(
                f"Expected `image` with 3 channels (BGR), saw {query_image.shape}"
            )
        doc_det = self.detect_doc(query_image)
        doc, doc_points = self.extract_doc(doc_det["dst_points"], query_image)
        if doc is not None:
            doc = self.clahe.apply(self._preprocess_image(doc))
            doc_points = utils.misc.numpy_tolist(doc_points, 1)
        return {
            "doc": doc,
            "doc_points": doc_points,  # will be None if doc detection failed
            "debug": {"doc_vis": get(doc_det["debug"], "img_threshold")},
        }


def visualise_detection(detector: _Detector, image: np.ndarray):
    """
    Args:
        detector: `_Detector` instance
        image: `np.ndarray`

    Returns:
        A dict.
    """
    logger.info(f"Received image of shape: {image.shape}")
    try:
        res = detector(image)
        det_success = res["doc"] is not None
    except (ValueError, TypeError, KeyError):
        res = {}
        det_success = False
    output = {
        "det_success": det_success,
        "doc": image_utils.ndarray_to_base64str(get(res, "doc", np.full_like(image, 128))),
        "doc_vis": image_utils.ndarray_to_base64str(get(res["debug"], "doc_vis", np.full_like(image, 128))),
        "doc_points": json.dumps(get(res, "doc_points", [None] * 4)),
    }
    return output
