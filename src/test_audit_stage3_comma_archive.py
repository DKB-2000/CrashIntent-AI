"""Synthetic fixtures test archive auditing, not model quality."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from audit_stage3_comma_archive import audit


class ArchiveAuditTest(unittest.TestCase):
    def fixture(self, path, shifted=False):
        prefix = 'Chunk_1/car|route/0/'
        times = np.arange(1200) / 20
        arrays = {'global_pose/frame_times': times,
                  'processed_log/CAN/car_speed/t': times + (100 if shifted else 0),
                  'processed_log/CAN/car_speed/value': np.zeros(1200),
                  'processed_log/CAN/steering_angle/t': times,
                  'processed_log/CAN/steering_angle/value': np.zeros(1200)}
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(prefix + 'video.hevc', b'fixture-not-a-video')
            for name, value in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, value)
                archive.writestr(prefix + name, buffer.getvalue())

    def test_stopped_and_windows_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root / 'sample.zip')
            result = audit(root / 'sample.zip', root / 'audit')
            self.assertEqual(result['accepted'], 1)
            self.assertEqual(result['distributions']['accel_label'], {'STOPPED': 600})
            self.assertEqual(result['distributions']['moving_steer'], {})
            self.assertEqual(result['episode_starts_per_segment']['steer_label'], {'STRAIGHT': 1})
            self.assertEqual(json.loads((root / 'audit/audit.json').read_text())['routes'], 1)

    def test_nonoverlapping_sensor_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root / 'sample.zip', shifted=True)
            result = audit(root / 'sample.zip', root / 'audit')
            self.assertEqual(result['accepted'], 0)
            self.assertEqual(len(result['failures']), 1)


if __name__ == '__main__':
    unittest.main()
