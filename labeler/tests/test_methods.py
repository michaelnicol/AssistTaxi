import unittest

import numpy as np

from labeler.methods.alina import AlinaParams, run_alina
from labeler.methods.cdlem import CdlemParams, run_cdlem


class MethodTest(unittest.TestCase):
    def test_alina_finds_yellow_stripe(self):
        image = np.full((240, 240, 3), 40, dtype=np.uint8)
        image[30:210, 100:112] = (0, 255, 255)
        quad = [(70, 220), (70, 20), (150, 20), (150, 220)]
        params = AlinaParams(
            peak_pixel_threshold=10,
            min_white_pixels=40,
            circular_threshold=5,
            mask_ignore_left_columns=0,
            dst_bottom_left=(70, 220),
            dst_top_left=(70, 20),
            dst_top_right=(150, 20),
            dst_bottom_right=(150, 220),
        )
        mask, coords, _stages, note = run_alina(image, quad, params)
        self.assertGreater(len(coords), 40, note)
        on_stripe = np.logical_and(coords[:, 0] >= 90, coords[:, 0] <= 125)
        self.assertGreater(int(on_stripe.sum()), 40)
        self.assertGreater(int(mask.sum()), 0)

    def test_cdlem_finds_bright_line(self):
        image = np.full((200, 200, 3), 30, dtype=np.uint8)
        image[98:103, 20:180] = 255
        polygon = [(10, 10), (190, 10), (190, 190), (10, 190)]
        mask, coords, _stages, _note = run_cdlem(image, polygon, CdlemParams())
        self.assertGreater(len(coords), 0)
        self.assertGreater(int((mask > 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
