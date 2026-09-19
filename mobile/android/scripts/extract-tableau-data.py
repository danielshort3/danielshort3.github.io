"""Build compact native dashboard assets from the published Tableau sources.

Requires tableauhyperapi. Run from the repository root. No report narratives,
precise coordinates, or customer addresses are included in the app assets.
"""
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from tableauhyperapi import Connection, HyperProcess, Telemetry

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "mobile/android/app/src/main/assets/native-demos"
DEST.mkdir(parents=True, exist_ok=True)

source = ROOT / "design/tableau/published/ufo-sightings-published.twbx"
states = "AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".lower().split()
with tempfile.TemporaryDirectory() as directory:
  with zipfile.ZipFile(source) as archive:
    name = next(name for name in archive.namelist() if name.endswith(".hyper"))
    target = Path(directory) / "ufo.hyper"
    target.write_bytes(archive.read(name))
  with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as process:
    with Connection(process.endpoint, str(target)) as connection:
      rows = connection.execute_list_query('''SELECT CAST(EXTRACT(YEAR FROM "datetime") AS INTEGER),
        CAST(EXTRACT(MONTH FROM "datetime") AS INTEGER), CAST(EXTRACT(HOUR FROM "datetime") AS INTEGER),
        LOWER("state"), COALESCE(LOWER("shape"), 'unknown'), COUNT(*)
        FROM "Extract"."Extract" WHERE LOWER("country") = 'us'
        GROUP BY 1,2,3,4,5 ORDER BY 1,2,3,4,5''')
rows = [row for row in rows if row[3] in states and row[0] is not None]
assert sum(row[5] for row in rows if row[0] == 2013) == 6334
payload = {"schemaVersion": 1, "source": "Published UFO Sightings Tableau extract", "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
  "scope": "Contiguous United States; historical reports through May 2014. Reports are unverified and not population-normalized.",
  "columns": ["year", "month", "hour", "state", "shape", "count"], "rows": rows}
(DEST / "tableau-ufo.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

source = ROOT / "design/tableau/analysis/pizza-weekday-estimate/source-tips.csv"
with source.open(newline="", encoding="utf-8-sig") as stream:
  rows = [[r["Date"], r["City"], r["Housing"], float(r["Cost"]), float(r["Tip"]),
    int(r["Total Delivery Time"].split(":")[0]) * 60 + int(r["Total Delivery Time"].split(":")[1])] for r in csv.DictReader(stream)]
assert len(rows) == 1251
payload = {"schemaVersion": 1, "source": "Published Pizza Delivery Tableau source records", "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
  "scope": "Historical delivery records; descriptive analysis, not a tip prediction.",
  "columns": ["date", "city", "housing", "cost", "tip", "deliveryMinutes"], "rows": rows}
(DEST / "tableau-pizza.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
for name in ["tableau-ufo.json", "tableau-pizza.json"]:
  path = DEST / name
  print(f"{name}: {path.stat().st_size:,} bytes")
