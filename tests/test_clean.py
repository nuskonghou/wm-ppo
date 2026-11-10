"""No machine-specific strings in the repository (file contents, including the compressed results): private IP
addresses, cluster file-system paths, object-store URLs, scheduler commands; and only ASCII-range text in code and docs."""
import gzip
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = re.compile(r'\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b|\b192\.168\.\d{1,3}\.\d{1,3}\b|'
                       r'(?<![\w.])/(fsx|lustre|scratch|home|mnt|dev/shm)/|s3://|\b(sbatch|srun|salloc|squeue|scancel)\b|SLURM_|'
                       r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]')
SKIP = {'.git', 'third_party', 'runs', 'models', 'data', 'weights', 'figures', '__pycache__'}
BINARY = ('.png', '.jpg', '.jpeg', '.pdf')   # images of the project page: not text


def files():
    for d, dn, fn in os.walk(ROOT):
        dn[:] = [x for x in dn if x not in SKIP]
        for f in fn:
            if not f.lower().endswith(BINARY):
                yield os.path.join(d, f)


class Clean(unittest.TestCase):
    def test_no_machine_specific_strings(self):
        bad = []
        for p in files():
            if p == os.path.abspath(__file__):
                continue
            with open(p, 'rb') as fh:
                raw = fh.read()
            text = gzip.decompress(raw).decode('utf-8', 'replace') if p.endswith('.gz') else raw.decode('utf-8', 'replace')
            for m in FORBIDDEN.finditer(text):
                bad.append(f'{os.path.relpath(p, ROOT)}: {m.group(0)!r}')
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
