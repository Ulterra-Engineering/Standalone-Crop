import unittest

import numpy as np
from PIL import Image

from luminance_matching import (
    MAX_CORRECTION_LSTAR,
    REFERENCE_MEDIAN_LSTAR,
    match_crop_luminance,
)


class LuminanceMatchingTests(unittest.TestCase):
    def test_top_crop_is_adjusted_toward_top_reference(self):
        image = Image.new("RGB", (64, 64), (70, 70, 70))
        matched, stats = match_crop_luminance(image, "top")

        self.assertEqual(matched.size, image.size)
        self.assertGreater(stats["after_median_lstar"], stats["before_median_lstar"])
        self.assertEqual(stats["target_median_lstar"], REFERENCE_MEDIAN_LSTAR["top"])
        self.assertLessEqual(abs(stats["applied_shift_lstar"]), MAX_CORRECTION_LSTAR)

    def test_face_crop_is_adjusted_toward_face_reference(self):
        image = Image.new("RGB", (64, 64), (180, 180, 180))
        matched, stats = match_crop_luminance(image, "face")

        self.assertLess(stats["after_median_lstar"], stats["before_median_lstar"])
        self.assertEqual(stats["target_median_lstar"], REFERENCE_MEDIAN_LSTAR["face"])

    def test_original_image_is_not_modified(self):
        pixels = np.full((32, 32, 3), (80, 100, 120), dtype=np.uint8)
        image = Image.fromarray(pixels)
        before = np.asarray(image).copy()

        match_crop_luminance(image, "top")

        np.testing.assert_array_equal(np.asarray(image), before)

    def test_isometric_has_no_reference_yet(self):
        image = Image.new("RGB", (32, 32), (100, 100, 100))
        with self.assertRaisesRegex(ValueError, "No luminance reference"):
            match_crop_luminance(image, "isometric")


if __name__ == "__main__":
    unittest.main()
