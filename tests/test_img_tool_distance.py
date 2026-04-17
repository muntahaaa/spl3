import unittest

from tool.img_tool import _calculate_position_distance


class TestImgToolDistance(unittest.TestCase):
    def test_distance_is_zero_for_identical_boxes(self):
        bbox = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(_calculate_position_distance(bbox, bbox), 0.0)

    def test_distance_is_positive_for_different_boxes(self):
        bbox1 = [0.1, 0.2, 0.3, 0.4]
        bbox2 = [0.6, 0.6, 0.8, 0.9]
        self.assertGreater(_calculate_position_distance(bbox1, bbox2), 0.0)


if __name__ == "__main__":
    unittest.main()
