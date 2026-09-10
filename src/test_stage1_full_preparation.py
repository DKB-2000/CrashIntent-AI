import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from prepare_stage1_full_run import inspect_video


class FullPreparationTests(unittest.TestCase):
    def test_full_decode_and_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10., (32, 24))
            self.assertTrue(writer.isOpened())
            for index in range(4):
                writer.write(np.full((24, 32, 3), index * 40, dtype=np.uint8))
            writer.release()
            expected = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), fps=10., frames=4, height=24, width=32)
            self.assertEqual(inspect_video(path, expected), 4)
            for field, value in [('frames', 5), ('fps', 20.), ('width', 64), ('sha256', 'invalid')]:
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        inspect_video(path, dict(expected, **{field: value}))


if __name__ == '__main__':
    unittest.main()
