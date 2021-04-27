# -*- coding: utf-8 -*-
"""
Created on 07 Mar 2021 18:17:27
@author: jiahuei

python -m pytest
"""
import unittest
import os
import cv2
from scanner.detect_doc import A4Detector
from scanner.utils.image import read_image, resize_image
from scanner import utils


class TestA4Detector(unittest.TestCase):

    def setUp(self):
        test_image = read_image(os.path.join(utils.misc.REPO_DIR, "tests", "data", "test_w8_ben.jpg"))
        self.test_image_480, _ = resize_image(test_image, max_side=480)
        self.test_image_224, _ = resize_image(test_image, max_side=224)

    def _test_detector(self, detector, name):
        with self.subTest(f"Sub-test: {name}: detect_doc"):
            dst_points = detector.detect_doc(self.test_image_480)["dst_points"]
            self.assertIsNotNone(dst_points)

        with self.subTest(f"Sub-test: {name}: extract_doc"):
            extracted, _ = detector.extract_doc(dst_points, self.test_image_480)
            self.assertIsNotNone(extracted)

    def test_detector_simple(self):
        det_kwargs = dict(
            use_image_features=False,
            grayscale_method="hue",
            threshold_method="otsu",
        )
        self._test_detector(A4Detector(**det_kwargs), "simple")

    def test_detector_simple_224(self):
        det_kwargs = dict(
            use_image_features=False,
            grayscale_method="hue",
            threshold_method="otsu",
        )
        detector = A4Detector(**det_kwargs)
        results = detector(self.test_image_224)
        self.assertIn("doc_points", results)

    def test_detector_features(self):
        det_kwargs = dict(
            homography_method=cv2.RANSAC,
            detector_name="sift",
            top_k_keypoints=50,
            grayscale_mode=True,
        )
        self._test_detector(A4Detector(**det_kwargs), "features")


if __name__ == '__main__':
    unittest.main()
