"""Read local published Tableau packages and emit bounded structural QA JSON.

Usage: python build/tableau/inspect-published-workbook.py pizzaDashboard
This never downloads, extracts files, modifies archives, or validates rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {
  "pizzaDashboard": "pizza-delivery-published",
  "ufoDashboard": "ufo-sightings-published",
}
MAX_ITEMS = 64
MAX_TEXT = 1600


def bounded(value):
  """Limit unusually large workbooks without silently hiding truncation."""
  if isinstance(value, str):
    return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "...[truncated]"
  if isinstance(value, list):
    result = [bounded(item) for item in value[:MAX_ITEMS]]
    if len(value) > MAX_ITEMS:
      result.append({"omittedItems": len(value) - MAX_ITEMS})
    return result
  if isinstance(value, dict):
    return {key: bounded(item) for key, item in value.items()}
  return value


def attrs(node):
  return {key.rsplit("}", 1)[-1]: value for key, value in node.attrib.items()} if node is not None else {}


def text(node):
  return "".join(node.itertext()).strip() if node is not None else ""


def tree(node, depth=0):
  """Preserve nested filter/action semantics with an explicit depth limit."""
  result = {"tag": node.tag.rsplit("}", 1)[-1], **attrs(node)}
  if node.text and node.text.strip():
    result["text"] = node.text.strip()
  children = list(node)
  if children:
    result["children"] = [tree(child, depth + 1) for child in children[:MAX_ITEMS]] if depth < 6 else []
    if depth >= 6 or len(children) > MAX_ITEMS:
      result["omittedChildren"] = len(children) if depth >= 6 else len(children) - MAX_ITEMS
  return result


def zone_details(zone, parent):
  """Keep raw coordinates and native style keys; do not infer rendered pixels."""
  geometry_keys = ("x", "y", "w", "h")
  return {
    "id": zone.get("id"),
    "name": zone.get("name"),
    "type": zone.get("type-v2", "worksheet"),
    "parentZoneId": parent.get("id") if parent is not None and parent.tag == "zone" else None,
    "geometry": {key: zone.get(key) for key in geometry_keys if key in zone.attrib},
    "layoutAttributes": {key: value for key, value in attrs(zone).items()
      if key not in (*geometry_keys, "id", "name")},
    "style": [tree(child) for child in zone if child.tag in ("zone-style", "style")],
    "otherProperties": [tree(child) for child in zone
      if child.tag not in ("zone", "zone-style", "style", "formatted-text")],
  }


def layout(node, sheet_names, fallback=None):
  zones = node.find("zones")
  inherited = zones is None and fallback is not None
  zone_list = list(zones.iter("zone")) if zones is not None else []
  views = list(dict.fromkeys(zone.get("name") for zone in zone_list
    if zone.get("name") in sheet_names and zone.get("type-v2", "") in ("", "worksheet")))
  controls = [{**attrs(zone), "text": text(zone.find("formatted-text"))}
    for zone in zone_list if zone.get("type-v2") not in (None, "layout-basic", "layout-flow", "empty")]
  parents = {child: parent for parent in zones.iter() for child in parent} if zones is not None else {}
  worksheet_zones = [zone_details(zone, parents.get(zone)) for zone in zone_list
    if zone.get("name") in sheet_names and zone.get("type-v2", "") in ("", "worksheet")]
  blank_zones = [zone_details(zone, parents.get(zone)) for zone in zone_list
    if zone.get("type-v2") in ("empty", "blank")]
  container_zones = [zone_details(zone, parents.get(zone)) for zone in zone_list
    if zone.get("type-v2", "").startswith("layout-")]
  return {
    "name": node.get("name", "Default"),
    "size": attrs(node.find("size")) or (fallback["size"] if inherited else {}),
    "inheritsDefault": inherited,
    "viewNames": fallback["viewNames"] if inherited else views,
    "controls": fallback["controls"] if inherited else controls,
    "worksheetZones": fallback["worksheetZones"] if inherited else worksheet_zones,
    "blankZones": fallback["blankZones"] if inherited else blank_zones,
    "containerZones": fallback["containerZones"] if inherited else container_zones,
  }


def worksheet(node, shared_filters):
  view = node.find("./table/view")
  slices = [text(item) for item in node.findall("./table/view/slices/column")]
  panes = []
  for pane in node.findall("./table/panes/pane"):
    label_formats = [attrs(item) for item in pane.findall(".//format")
      if item.get("attr", "").startswith("mark-labels")]
    enabled = next((item["value"] == "true" for item in reversed(label_formats)
      if item["attr"] == "mark-labels-show"), None)
    panes.append({
      "id": pane.get("id"),
      "markType": attrs(pane.find("mark")),
      "encodings": [tree(item) for item in pane.findall("./encodings/*")],
      "labelsEnabledExplicit": enabled,
      "labelFormats": label_formats,
      "labelTemplate": text(pane.find("./customized-label/formatted-text")),
    })
  calculations = []
  if view is not None:
    for dependencies in view.findall("datasource-dependencies"):
      for column in dependencies.findall("column"):
        calculation = column.find("calculation")
        if calculation is not None:
          calculations.append({
            "datasource": dependencies.get("datasource"), **attrs(column),
            "calculation": tree(calculation),
          })
  axis_styles = [{"scope": "table", "rule": tree(rule)}
    for rule in node.findall('./table/style/style-rule[@element="axis"]')]
  for pane in node.findall("./table/panes/pane"):
    axis_styles.extend({"scope": "pane", "paneId": pane.get("id"), "rule": tree(rule)}
      for rule in pane.findall('./style/style-rule[@element="axis"]'))
  return {
    "name": node.get("name"),
    "title": text(node.find("./layout-options/title/formatted-text")),
    "rows": text(node.find("./table/rows")),
    "columns": text(node.find("./table/cols")),
    "panes": panes,
    "localFilters": [tree(item) for item in node.findall("./table/view/filter")],
    "sharedFilterColumns": [column for column in shared_filters if column in slices],
    "filterShelfColumns": slices,
    "sorts": [tree(item) for item in node.findall("./table/view/shelf-sorts/*")],
    "referenceLines": [tree(item) for item in node.findall(".//reference-line")],
    "axisStyles": axis_styles,
    "usedCalculations": calculations,
  }


def inspect(path, project_id):
  contents = path.read_bytes()
  package_hash = hashlib.sha256(contents).hexdigest()
  with zipfile.ZipFile(io.BytesIO(contents)) as archive:
    names = archive.namelist()
    twbs = [name for name in names if name.lower().endswith(".twb")]
    if len(twbs) != 1:
      raise ValueError("Expected exactly one TWB inside the local published package.")
    root = ET.fromstring(archive.read(twbs[0]))
    hyper_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest()
      for name in names if name.lower().endswith(".hyper")}
  sheet_nodes = {item.get("name"): item for item in root.findall("./worksheets/worksheet")}
  dashboards = []
  shown = set()
  for dashboard in root.findall("./dashboards/dashboard"):
    default = layout(dashboard, sheet_nodes)
    default["name"] = "Default"
    devices = [layout(item, sheet_nodes, default)
      for item in dashboard.findall("./devicelayouts/devicelayout")]
    for item in [default, *devices]:
      shown.update(item["viewNames"])
    dashboards.append({"name": dashboard.get("name"), "default": default, "devices": devices})
  shared = [tree(item) for item in root.findall("./shared-views/shared-view/filter")]
  # Current Tableau packages put shared-view directly under the workbook.
  shared += [tree(item) for item in root.findall("./shared-view/filter")]
  shared_columns = [item.get("column") for item in shared]
  manifest_path = path.with_name(path.stem + "-manifest.json")
  manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
  original_hashes = manifest.get("originalHyperSha256")
  return bounded({
    "projectId": project_id,
    "file": path.relative_to(ROOT).as_posix(),
    "packageSha256": package_hash,
    "matchesSnapshotManifest": package_hash == manifest.get("sha256") if manifest.get("sha256") else None,
    "workbookEntry": twbs[0],
    "scope": "Local archive structure only; no native rendering or live publication verification.",
    "interpretation": "Automatic mark types and absent label flags retain Tableau defaults. Shared filters are matched by worksheet filter-shelf references. Calculations are shown worksheet dependencies, including ad hoc fields.",
    "layoutInterpretation": "Zone geometry and style values are raw workbook attributes, not resolved pixel bounds or computed inherited styles. Blank zones identify native empty/blank objects, not confirmed visible backing panels. Style records retain native corner-radius keys when present; absent keys do not prove square corners. Axis styles include stored range/format encodings, not evaluated automatic limits or rendered ticks. Device layouts without zones explicitly inherit Default. Rendering, overlap and visual alignment still require native view/export review.",
    "hyperSha256": hyper_hashes,
    "matchesRecordedOriginalHyper": sorted(hyper_hashes.values()) == sorted(original_hashes.values()) if original_hashes else None,
    "dashboardCount": len(dashboards),
    "dashboards": dashboards,
    "sharedFilters": shared,
    "actions": [tree(item) for item in root.findall("./actions/action")],
    "shownWorksheetCount": len(shown),
    "shownWorksheets": [worksheet(sheet_nodes[name], shared_columns) for name in sorted(shown)],
    "unshownWorksheetNames": sorted(set(sheet_nodes) - shown),
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("project_id", choices=PACKAGES)
  args = parser.parse_args()
  path = ROOT / "design/tableau/published" / f"{PACKAGES[args.project_id]}.twbx"
  print(json.dumps(inspect(path, args.project_id), indent=2, ensure_ascii=True))


if __name__ == "__main__":
  try:
    main()
  except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile) as error:
    print(f"[tableau-inspect] {error}", file=sys.stderr)
    sys.exit(1)
