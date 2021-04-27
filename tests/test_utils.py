# -*- coding: utf-8 -*-
"""
Created on 08 Mar 2021 17:07:06
@author: jiahuei
"""
import unittest
import numpy as np
from scanner.utils import misc


class TestUtils(unittest.TestCase):

    def test_round_to_nearest_odd(self):
        """ Round to nearest odd number. Round up for ties. """
        inputs = [(_ - 25) / 4 for _ in range(0, 50)]
        for x in inputs:
            with self.subTest(f"Positive float: {x}"):
                y = misc.round_to_nearest_odd(x)
                self.assertEqual(y % 2, 1)
                self.assertLessEqual(abs(x - y), 1)
        # Complex floats
        with self.subTest(f"Complex float"):
            with self.assertRaises(TypeError):
                misc.round_to_nearest_odd(5.2 + 4j)

    def test_numpy_tolist(self):
        """ Convert NumPy array to Python list, with specified precision. """
        with self.subTest(f"Floats"):
            y = misc.numpy_tolist(np.float32([[3.21, 4.5], [5.9124, 21]]), 1)
            self.assertEqual(y, [[3.2, 4.5], [5.9, 21.0]])
        with self.subTest(f"Integers"):
            y = misc.numpy_tolist(np.int8([[3.21, 4.5], [5.9124, 21]]), 1)
            self.assertEqual(y, [[3, 4], [5, 21]])


if __name__ == '__main__':
    unittest.main()
