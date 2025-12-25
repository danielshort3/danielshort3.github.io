import json
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

INPUT_PATH = Path("documents/Project_7.xlsx")
OUTPUT_PATH = Path("aws/target-empty-package/output/data.json")

NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def col_to_index(col):
  idx = 0
  for ch in col:
    idx = idx * 26 + (ord(ch) - 64)
  return idx - 1


def parse_sheet(zipf, sheet_path, shared):
  root = ET.fromstring(zipf.read(sheet_path))
  rows = []
  for row in root.findall("main:sheetData/main:row", NS):
    row_idx = int(row.attrib.get("r", 0))
    cells = {}
    for cell in row.findall("main:c", NS):
      cell_ref = cell.attrib.get("r")
      if not cell_ref:
        continue
      col = "".join(ch for ch in cell_ref if ch.isalpha())
      col_idx = col_to_index(col)
      val_node = cell.find("main:v", NS)
      if val_node is None:
        val = ""
      else:
        val = val_node.text or ""
      if cell.attrib.get("t") == "s":
        try:
          val = shared[int(val)]
        except Exception:
          pass
      cells[col_idx] = val
    rows.append((row_idx, cells))
  return rows


def load_shared_strings(zipf):
  shared = []
  if "xl/sharedStrings.xml" not in zipf.namelist():
    return shared
  sst = ET.fromstring(zipf.read("xl/sharedStrings.xml"))
  for si in sst.findall(".//main:si", NS):
    text = "".join(node.text or "" for node in si.findall(".//main:t", NS))
    shared.append(text)
  return shared


def find_sheet_path(zipf, sheet_name):
  wb = ET.fromstring(zipf.read("xl/workbook.xml"))
  rels = ET.fromstring(zipf.read("xl/_rels/workbook.xml.rels"))
  relmap = {
    rel.attrib["Id"]: rel.attrib["Target"]
    for rel in rels.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
  }
  for sheet in wb.findall("main:sheets/main:sheet", NS):
    if sheet.attrib.get("name") == sheet_name:
      rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
      target = relmap.get(rid)
      if target:
        return f"xl/{target}"
  return None


def parse_cleaned_data(path):
  origin = datetime(1899, 12, 30)
  with zipfile.ZipFile(path) as zipf:
    shared = load_shared_strings(zipf)
    sheet_path = find_sheet_path(zipf, "Cleaned Data")
    if not sheet_path:
      raise RuntimeError("Cleaned Data sheet not found")
    rows = parse_sheet(zipf, sheet_path, shared)

  headers = None
  header_row = None
  for row_idx, cells in rows:
    values = [cells.get(i, "") for i in range(0, 8)]
    if values[:3] == ["REPORTED BY", "DATE_TIME", "RETAIL VALUE"]:
      headers = values
      header_row = row_idx
      break
  if not headers:
    raise RuntimeError("Header row not found")

  data = []
  for row_idx, cells in rows:
    if row_idx <= header_row:
      continue
    row_vals = [cells.get(i, "") for i in range(0, 8)]
    if not any(row_vals):
      continue
    record = dict(zip(headers, row_vals))
    dt_raw = record.get("DATE_TIME", "")
    dt_iso = ""
    if dt_raw != "":
      try:
        dt = origin + timedelta(days=float(dt_raw))
        dt_iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
      except Exception:
        dt_iso = ""
    value_raw = record.get("RETAIL VALUE", "")
    try:
      value = float(value_raw)
    except Exception:
      value = 0
    data.append({
      "employee": record.get("REPORTED BY", ""),
      "datetime": dt_iso,
      "value": value,
      "location": record.get("AREA FOUND", ""),
      "condition": record.get("PACKAGE CONDITION", ""),
      "department": record.get("DPCI_Department", ""),
      "class": record.get("DPCI_Class", ""),
      "item": record.get("DPCI_Item", "")
    })
  return data


def build_output(rows):
  dates = sorted([row["datetime"] for row in rows if row.get("datetime")])
  meta = {
    "recordCount": len(rows),
    "startDate": dates[0] if dates else "",
    "endDate": dates[-1] if dates else ""
  }
  return {
    "meta": meta,
    "rows": rows
  }


def main():
  if not INPUT_PATH.exists():
    raise SystemExit(f"Missing input: {INPUT_PATH}")
  rows = parse_cleaned_data(INPUT_PATH)
  output = build_output(rows)
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
  print(f"Wrote {OUTPUT_PATH} ({len(rows)} records)")


if __name__ == "__main__":
  main()
