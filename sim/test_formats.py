import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formats import load_hardware_image  # noqa: E402


class HardwareImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.image = load_hardware_image(root / "csrc/vivado_mem/mem_killeroo_gold")

    def test_image_dimensions_and_ranges(self):
        image = self.image
        self.assertEqual(image.global_sdf.shape, (32**3,))
        self.assertEqual(image.local_sdf.shape[0] % 64, 0)
        self.assertEqual(image.local_mapping.shape, (32**3,))
        self.assertEqual(image.subgrid_start.shape, (64**3,))
        self.assertEqual(image.subgrid_count.shape, (64**3,))
        self.assertTrue(np.all(image.subgrid_start + image.subgrid_count <= len(image.compact_triangles)))

    def test_ddr_triangles_match_render_sidecar(self):
        image = self.image
        sample = np.linspace(0, len(image.compact_ids) - 1, 257, dtype=np.int32)
        source = image.original_triangles[image.compact_ids[sample]]
        expected = np.concatenate((source[:, 0], source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]), axis=1)
        np.testing.assert_array_equal(image.compact_triangles[sample], expected)

    def test_every_source_triangle_is_reachable(self):
        present = np.zeros(len(self.image.original_triangles), dtype=bool)
        present[self.image.compact_ids] = True
        self.assertTrue(present.all())


if __name__ == "__main__":
    unittest.main()
