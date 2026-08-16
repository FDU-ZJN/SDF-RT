import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdf_rt.camera import make_tiled_pixel_order, select_trace_roi


class TraceRoiTest(unittest.TestCase):
    def test_david_orbit_roi_is_dense_and_unique(self):
        order = make_tiled_pixel_order(1920, 1080)
        positions, local = select_trace_roi(order, 1920, 1080, (732, 252, 456, 604))

        self.assertEqual(positions.size, 275424)
        np.testing.assert_array_equal(np.sort(local), np.arange(275424, dtype=np.uint32))
        selected = order[positions]
        self.assertTrue(np.all((selected[:, 0] >= 732) & (selected[:, 0] < 1188)))
        self.assertTrue(np.all((selected[:, 1] >= 252) & (selected[:, 1] < 856)))

    def test_full_frame_preserves_tiled_order(self):
        order = make_tiled_pixel_order(320, 180)
        positions, local = select_trace_roi(order, 320, 180, (0, 0, 320, 180))

        np.testing.assert_array_equal(positions, np.arange(320 * 180, dtype=np.int64))
        self.assertEqual(np.unique(local).size, 320 * 180)

    def test_rejects_unaligned_or_out_of_bounds_roi(self):
        order = make_tiled_pixel_order(320, 180)
        with self.assertRaises(ValueError):
            select_trace_roi(order, 320, 180, (1, 0, 100, 100))
        with self.assertRaises(ValueError):
            select_trace_roi(order, 320, 180, (300, 0, 24, 100))


if __name__ == "__main__":
    unittest.main()
