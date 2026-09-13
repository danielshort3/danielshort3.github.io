"""Download and verify one of the two published Tableau dashboard workbooks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[2]
WORKBOOKS = {
  "pizzaDashboard": {
    "url": "https://public.tableau.com/workbooks/Pizza_Delivery.twb?showVizHome=no",
    "original": "Pizza_Delivery.twbx",
    "output": "pizza-delivery-published",
  },
  "ufoDashboard": {
    "url": "https://public.tableau.com/workbooks/UFO_Sightings_16769494135040.twb?showVizHome=no",
    "original": "UFO_Sightings.twbx",
    "output": "ufo-sightings-published",
  },
}


def sha256(contents):
  return hashlib.sha256(contents).hexdigest()


def inspect_archive(contents):
  with zipfile.ZipFile(io.BytesIO(contents)) as archive:
    files = [
      {"name": item.filename, "bytes": item.file_size, "sha256": sha256(archive.read(item))}
      for item in archive.infolist() if not item.is_dir()
    ]
  if sum(item["name"].endswith(".twb") for item in files) != 1:
    raise ValueError("Expected exactly one Tableau workbook in the downloaded package.")
  if not any(item["name"].endswith(".hyper") for item in files):
    raise ValueError("The downloaded package does not contain a Hyper extract.")
  return files


def hyper_hashes(files):
  return {item["name"]: item["sha256"] for item in files if item["name"].endswith(".hyper")}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("project_id", choices=WORKBOOKS)
  parser.add_argument("--check-only", action="store_true", help="Download and verify without replacing any local files.")
  args = parser.parse_args()
  workbook = WORKBOOKS[args.project_id]
  source = ROOT / "design/tableau/sources" / workbook["original"]
  original_hyper = hyper_hashes(inspect_archive(source.read_bytes()))

  request = Request(workbook["url"], headers={"Cache-Control": "no-cache"})
  with urlopen(request, timeout=60) as response:
    contents = response.read()
  files = inspect_archive(contents)
  unchanged = sorted(hyper_hashes(files).values()) == sorted(original_hyper.values())
  if not unchanged:
    raise ValueError("The published Hyper extract differs from the original snapshot; existing archives were kept.")

  output = ROOT / "design/tableau/published" / f'{workbook["output"]}.twbx'
  manifest = {
    "file": output.relative_to(ROOT).as_posix(),
    "sourceUrl": workbook["url"],
    "downloadedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "sha256": sha256(contents),
    "files": files,
    "originalHyperSha256": original_hyper,
    "publishedHyperUnchanged": unchanged,
  }
  if not args.check_only:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(contents)
    manifest_path = output.with_name(f'{workbook["output"]}-manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({
    "projectId": args.project_id,
    "checkOnly": args.check_only,
    "bytes": len(contents),
    **manifest,
  }, indent=2))


if __name__ == "__main__":
  try:
    main()
  except Exception as error:
    print(f"[tableau-snapshot] {error}", file=sys.stderr)
    sys.exit(1)
