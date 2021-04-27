# -*- coding: utf-8 -*-
"""
Created on 30 Oct 2020 22:10:31
@author: jiahuei
"""
import logging
import cv2
import base64
import numpy as np
from typing import Dict
from functools import lru_cache
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)


class FlannEnum:
    # https://github.com/opencv/opencv/blob/4.5.0/modules/flann/include/opencv2/flann/defines.h#L70
    FLANN_INDEX_LINEAR = 0
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    FLANN_INDEX_COMPOSITE = 3
    FLANN_INDEX_KDTREE_SINGLE = 4
    FLANN_INDEX_HIERARCHICAL = 5
    FLANN_INDEX_LSH = 6
    FLANN_INDEX_SAVED = 254
    FLANN_INDEX_AUTOTUNED = 255


# noinspection PyPep8Naming
class ComputeHomography:
    """
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    https://github.com/abidrahmank/OpenCV2-Python-Tutorials/blob/master/source/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.rst
    """

    def __init__(
            self, matcher,
            match_ratio=0.8, min_match_count=40, top_k_keypoints=100,
            homography_method=cv2.LMEDS, reproj_threshold=5.0
    ):
        """
        The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
        distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
        correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
        noise is rather small, use the default method (method=0).
        """
        self.matcher = matcher
        self.match_ratio = match_ratio
        self.top_k_keypoints = top_k_keypoints
        self.min_match_count = min_match_count
        self.homography_method = homography_method
        self.reproj_threshold = reproj_threshold

    def __call__(self, q_keypoints, q_descriptors, g_keypoints, g_descriptors) -> Dict:
        """
        Computes homography

        :param q_keypoints: Keypoints of the query image containing the reference object.
        :param q_descriptors: Descriptors of the query image containing the reference object.
        :param g_keypoints: Keypoints of the reference image / object.
        :param g_descriptors: Descriptors of the reference image / object.
        :return:
        """
        match_ratio = self.match_ratio
        top_k_keypoints = self.top_k_keypoints
        min_match_count = self.min_match_count
        if isinstance(match_ratio, float) and match_ratio > 0:
            matches = self.matcher.knnMatch(q_descriptors, g_descriptors, k=2)
            logger.info(
                f"{self.__class__.__name__}: Number of matches before filtering = {len(matches)}"
            )
            try:
                # Get all the good matches as per Lowe's ratio test.
                # ratio = 0.7 in tutorial
                matches = [m for m, n in matches if m.distance < match_ratio * n.distance]
            except ValueError:
                matches = []
        else:
            matches = self.matcher.match(q_descriptors, g_descriptors)
        logger.info(
            f"{self.__class__.__name__}: Number of matches = {len(matches)}"
        )

        if isinstance(top_k_keypoints, int) and top_k_keypoints > min_match_count:
            matches = sorted(matches, key=lambda x: x.distance)[:top_k_keypoints]
            logger.info(
                f"{self.__class__.__name__}: Using top {top_k_keypoints} matches"
            )
        if len(matches) >= min_match_count:
            dst_pts = np.float32([q_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            src_pts = np.float32([g_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            """
            Coordinates of the points is a matrix of the type CV_32FC2 or vector<Point2f>
            CV_32FC2 = float32 with 2 channels
            https://stackoverflow.com/a/47403282
            Shape of (n_points, 1, 2) or (n_points, 1, 3) should work for most OpenCV functions
            Some OpenCV functions will also accept (n_points, n_dimensions)
            Use (n_points, 1, n_dimensions) to keep everything consistent
            """
            M, mask = cv2.findHomography(
                src_pts, dst_pts, self.homography_method, ransacReprojThreshold=self.reproj_threshold
            )
            match_mask = mask.ravel().tolist()
        else:
            logger.info(
                f"{self.__class__.__name__}: Not enough matches are found. "
                f"Matches = {len(matches)}    Min required = {min_match_count}"
            )
            match_mask = M = None
        return {
            "homography": M,
            "matches": matches,
            "match_mask": match_mask,
            "q_keypoints": q_keypoints,
            "g_keypoints": g_keypoints,
        }


# noinspection PyPep8Naming
class Detector_ORB_BEBLID:
    """
    https://github.com/iago-suarez/BEBLID
    BSD-3-Clause License
    """

    def __init__(self, **kwargs):
        self.detector = cv2.ORB_create(
            nfeatures=kwargs.get("nfeatures", 800),
            firstLevel=kwargs.get("firstLevel", 2),
        )
        """
        https://docs.opencv.org/4.5.1/d7/d99/classcv_1_1xfeatures2d_1_1BEBLID.html
        scale_factor	Adjust the sampling window around detected keypoints:
            1.00f should be the scale for ORB keypoints
            6.75f should be the scale for SIFT detected keypoints
            6.25f is default and fits for KAZE, SURF detected keypoints
            5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints
        """
        self.descriptor = cv2.xfeatures2d.BEBLID_create(
            scale_factor=kwargs.get("scale_factor", 1.),
        )

    def detectAndCompute(self, image, mask=None):
        kpts = self.detector.detect(image, mask)
        kpts, descs = self.descriptor.compute(image, kpts)
        return kpts, descs


def read_image(path, grayscale=False):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def resize_image(
        image: np.ndarray, max_side: int = None, inter: int = cv2.INTER_AREA
):
    """
    Aspect ratio preserving image resize.
    """
    assert isinstance(image, np.ndarray), (
        f"Expected `image` of type `np.ndarray`, saw {type(image)}"
    )
    assert isinstance(max_side, int), "`max_side` must be an integer."

    h, w = image.shape[:2]
    if max(h, w) == max_side:
        return image, 1.0
    if h > w:
        resize_ratio = max_side / h
        new_h, new_w = max_side, int(w * resize_ratio)
    else:
        resize_ratio = max_side / w
        new_h, new_w = int(h * resize_ratio), max_side
    image = cv2.resize(image, (new_w, new_h), interpolation=inter)
    return image, resize_ratio


def get_keypoint_detector_and_matcher(
        detector_name="sift", matcher_name="flann", affine_invariant=False, affine_params=None
):
    assert isinstance(detector_name, str), \
        f"Expected `detector_name` of type `str`, saw {type(detector_name)}"
    # https://github.com/opencv/opencv/blob/4.5.0/samples/python/find_obj.py
    detector_name = detector_name.lower()
    if detector_name == "sift":
        # noinspection PyUnresolvedReferences
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif detector_name == "surf":
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif detector_name == "orb":
        detector = cv2.ORB_create(800, firstLevel=2)
        norm = cv2.NORM_HAMMING
    elif detector_name == "orb_beblid":
        detector = Detector_ORB_BEBLID()
        norm = cv2.NORM_HAMMING
    elif detector_name == "akaze":
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif detector_name == "brisk":
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Invalid `detector_name`: {detector_name} (not case-sensitive)")
    if affine_invariant:
        # Affine invariant, described as ASIFT
        if not isinstance(affine_params, dict):
            affine_params = {"maxTilt": 5, "minTilt": 0, "tiltStep": 2, "rotateStepBase": 90}
        # noinspection PyUnresolvedReferences
        detector = cv2.AffineFeature_create(detector, **affine_params)

    matcher_name = matcher_name.lower()
    if matcher_name == "flann":
        if norm == cv2.NORM_L2:
            flann_params = {"algorithm": FlannEnum.FLANN_INDEX_KDTREE, "trees": 5}
        else:
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
            # py_feature2d/py_matcher/py_matcher.html#flann-based-matcher
            flann_params = {
                "algorithm": FlannEnum.FLANN_INDEX_LSH,
                "table_number": 6,  # 12
                "key_size": 12,  # 20
                "multi_probe_level": 1
            }
        # Number of times the trees in the index should be recursively traversed.
        # Higher values gives better precision, but also takes more time.
        search_params = {"checks": 50}
        matcher = cv2.FlannBasedMatcher(flann_params, search_params)
    elif matcher_name == "bf":
        # Cross-check has to be False to enable kNN-match
        matcher = cv2.BFMatcher(norm, crossCheck=False)
    else:
        raise ValueError(f"Invalid `matcher_name`: {matcher_name} (not case-sensitive)")
    logger.info(
        f"Detector = {detector.__class__.__name__}    Matcher = {matcher.__class__.__name__}"
    )
    return detector, matcher


def otsu_thresholding(
        image,
        gauss_blur_ksize=9,
        med_blur_pre_ksize=17,
        med_blur_post_ksize=17,
):
    """
    Image -> med_blur_pre -> gauss_blur -> Otsu -> med_blur_post -> Image
    References:
        https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
        https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html

    :param image: Image
    :param gauss_blur_ksize: Gaussian blur kernel size
    :param med_blur_pre_ksize: Median blur kernel size (before thresholding)
    :param med_blur_post_ksize: Median blur kernel size (after thresholding)
    :return:
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected `image` of type `np.ndarray`, saw {type(image)}"
        )
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        pass
    else:
        raise ValueError(
            f"Input `image` must be be 2-D or 3-D with 1 channel only, saw {image.shape}"
        )
    # Median blur to remove foreground noise
    image = cv2.medianBlur(image, med_blur_pre_ksize)
    # Gaussian filtering to remove noise
    image = cv2.GaussianBlur(image, (gauss_blur_ksize, gauss_blur_ksize), 0)
    # Otsu's thresholding
    threshold, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.medianBlur(image, med_blur_post_ksize)
    # # Remove foreground noise
    # image = cv2.morphologyEx(
    #     image, cv2.MORPH_OPEN,
    #     kernel=np.ones((morph_open_ksize, morph_open_ksize), np.uint8)
    # )
    # # Remove holes
    # image = cv2.morphologyEx(
    #     image, cv2.MORPH_CLOSE,
    #     kernel=np.ones((morph_close_ksize, morph_close_ksize), np.uint8)
    # )
    return image, threshold


def polygon_area(x, y):
    # https://stackoverflow.com/a/30408825
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def detect_and_sort(detector, img_gray, keep_largest=True):
    bboxes = detector(img_gray, 0)
    if len(bboxes) >= 1:
        if len(bboxes) == 1:
            bbox = bboxes[0]
        else:
            bbox = sorted(bboxes, key=lambda x: x.area(), reverse=keep_largest)[0]
    else:
        bbox = None
    return bbox


def line_intersection(line1, line2):
    """
    Computes the intersection point between line1 and line2.
    https://stackoverflow.com/a/20677983

    Args:
        line1: A length-2 tuple (A, B), where both A and B are each a length-2 tuple (x, y).
        line2: Same as `line1`.
    Returns:
        A length-2 tuple (x, y).
    """
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        raise RuntimeError("The lines do not intersect.")

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return x, y


def generate_bbox(image):
    # contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the biggest area
    if len(contours) < 1:
        return None
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Approx polygon
    perimeter = cv2.arcLength(cnt, True)
    epsilon = 0.01 * perimeter
    bbox = np.squeeze(cv2.approxPolyDP(cnt, epsilon, closed=True))
    # TODO: do contour bounding if more than 4 sides
    if len(bbox.shape) < 2:
        bbox = None
    if bbox.shape[0] < 4:
        bbox = None
    elif bbox.shape[0] == 4:
        pass
    else:
        bbox = None
        # Disable for now
        # # Compute the length of the sides. Index start from negative to wrap around the polygon.
        # side_lens = [dist.euclidean(bbox[i], bbox[i + 1]) for i in range(-1, bbox.shape[0] - 1)]
        # # Get the indices of the top 4 largest lines.
        # # (i -1) is used as we started from negative 1 in above.
        # # TODO: perhaps take into consideration the line angle to ensure only 1 line is kept per side
        # largest_side_idx = [i - 1 for i, _ in enumerate(side_lens) if _ >= sorted(side_lens, reverse=True)[3]]
        # # Form the 4 lines, each represented as ((x0, y0), (x1, y1))
        # lines = [(bbox[i], bbox[i + 1]) for i in largest_side_idx][:4]
        # try:
        #     # Try to compute the intersection points
        #     bbox = [line_intersection(lines[i], lines[i + 1]) for i in range(-1, 3)]
        #     bbox = np.array(bbox, dtype=np.int32)
        # except RuntimeError:
        #     bbox = None
    return bbox


def visualise_bbox(
        image,
        label,
        xmin, ymin, xmax, ymax,
        color=(102, 199, 56),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.4
):
    ret, baseline = cv2.getTextSize(label, font, font_scale, 1)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
    cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
    cv2.putText(image, label, (xmin, ymax - baseline), font, font_scale, (0, 0, 0), 1)
    return image


def order_points(pts):
    """
    https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """
    _error = f"`pts` must have shape (M, 1, N) or (M, N), saw {pts.shape}"
    if pts.ndim == 2:
        pass
    elif pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts.squeeze(1)
    else:
        raise ValueError(_error)
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, output_height=None, output_width=None):
    """
    https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """
    # obtain a consistent order of the points and unpack them individually
    assert isinstance(image, np.ndarray), f"Expected `image` of type `np.ndarray`, saw {type(image)}"
    assert isinstance(pts, np.ndarray), f"Expected `pts` of type `np.ndarray`, saw {type(pts)}"

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    if (
            isinstance(output_height, int) and isinstance(output_width, int) and
            output_height > 0 and output_width > 0
    ):
        max_width = output_width
        max_height = output_height
    else:
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        # width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        # width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width_a = dist.euclidean(br, bl)
        width_b = dist.euclidean(tr, tl)
        max_width = max(int(width_a), int(width_b))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        # height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        # height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        height_a = dist.euclidean(tr, br)
        height_b = dist.euclidean(tl, bl)
        max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # return the warped image
    return warped


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    table = get_gamma_table(1.0 / gamma)
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


@lru_cache(maxsize=4, typed=True)
def get_gamma_table(inv_gamma):
    return np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")


def bytes_to_ndarray(data: bytes) -> np.ndarray:
    # https://stackoverflow.com/a/52495126
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def ndarray_to_bytes(data: np.ndarray, image_type=".jpg"):
    return cv2.imencode(image_type, data)[1].tobytes()


def base64str_to_ndarray(data: str) -> np.ndarray:
    return bytes_to_ndarray(base64.b64decode(data))


def ndarray_to_base64str(data: np.ndarray) -> str:
    # https://stackoverflow.com/a/57830004
    return base64.b64encode(ndarray_to_bytes(data)).decode()
