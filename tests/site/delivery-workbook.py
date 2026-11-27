"""Validate the downloadable Delivery Tip workbook and its pivot refresh source."""
import re
import sys
from xml.etree import ElementTree
from zipfile import ZipFile


SHEET = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PIVOT_FIELDS = {"City", "Gated Community", "Housing", "Tip", "Tip Percentage"}


def check_workbook(filename):
  with ZipFile(filename) as book:
    parts = book.namelist()
    workbook = ElementTree.fromstring(book.read("xl/workbook.xml"))
    if len(workbook.findall(f"{SHEET}sheets/{SHEET}sheet")) != 6:
      raise ValueError("The six historical worksheets must be available")
    for folder, prefix in (("pivotTables", "pivotTable"), ("charts", "chart")):
      count = sum(bool(re.fullmatch(rf"xl/{folder}/{prefix}\d+\.xml", p)) for p in parts)
      if count != 9:
        raise ValueError(f"The workbook must preserve its nine {folder}")
    if any(p.startswith(("xl/model/", "xl/externalLinks/")) or p == "xl/connections.xml" for p in parts):
      raise ValueError("The public workbook must refresh from its own worksheet data")
    for p in parts:
      if re.fullmatch(r"customXml/item\d+\.xml", p):
        if ElementTree.fromstring(book.read(p)).tag.startswith("{http://microsoft.data.visualization."):
          raise ValueError("The workbook should use its worksheet map preview")

    tables = {}
    for p in parts:
      if re.fullmatch(r"xl/tables/table\d+\.xml", p):
        table = ElementTree.fromstring(book.read(p))
        tables[table.get("name")] = [col.get("name") for col in table.findall(f"{SHEET}tableColumns/{SHEET}tableColumn")]
    caches = [p for p in parts if re.fullmatch(r"xl/pivotCache/pivotCacheDefinition\d+\.xml", p)]
    if not caches:
      raise ValueError("The workbook must contain its interactive PivotTables")
    for p in caches:
      cache = ElementTree.fromstring(book.read(p))
      fields = [f.get("name") for f in cache.findall(f"{SHEET}cacheFields/{SHEET}cacheField")]
      if len(fields) != len(PIVOT_FIELDS) or set(fields) != PIVOT_FIELDS:
        raise ValueError("Unexpected columns in the downloadable pivot data")
      source = cache.find(f"{SHEET}cacheSource/{SHEET}worksheetSource")
      if source is None or tables.get(source.get("name")) != fields:
        raise ValueError("Pivot refresh must use a worksheet table with the same analytical columns")
    records = [p for p in parts if re.fullmatch(r"xl/pivotCache/pivotCacheRecords\d+\.xml", p)]
    if not records:
      raise ValueError("The workbook must open with its historical pivot data available")
    for p in records:
      data = ElementTree.fromstring(book.read(p))
      rows = data.findall(f"{SHEET}r")
      if len(rows) != int(data.get("count", "0")) or not rows:
        raise ValueError("Incomplete saved pivot data")
      if any(len(row) != len(PIVOT_FIELDS) for row in rows):
        raise ValueError("Saved pivot rows must match the analytical fields")


if __name__ == "__main__":
  check_workbook(sys.argv[1])
  print("Delivery Tip workbook and pivot refresh source verified.")
