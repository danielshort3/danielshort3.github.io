#!/usr/bin/env python3
"""Create a deterministic ZIP archive from the provided files."""
import sys
from pathlib import Path
import zipfile

CONSTANT_DATE = (2020, 1, 1, 0, 0, 0)


def main():
  if len(sys.argv) < 3:
    sys.stderr.write("Usage: create_zip.py <output> <file> [file...]\n")
    return 1

  output = Path(sys.argv[1])
  files = [Path(p) for p in sys.argv[2:]]
  files = [p for p in files if p.exists()]
  if not files:
    sys.stderr.write("No input files for %s\n" % output)
    return 0

  output.parent.mkdir(parents=True, exist_ok=True)
  with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for file_path in files:
      data = file_path.read_bytes()
      info = zipfile.ZipInfo(file_path.name)
      info.date_time = CONSTANT_DATE
      info.compress_type = zipfile.ZIP_DEFLATED
      zf.writestr(info, data)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
