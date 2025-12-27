import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

BASE_DIR = Path("retail-data/CSV_Anonymized")
OUTPUT_PATH = Path("aws/retail-loss-sales/output/data.json")

SALES_FILE = BASE_DIR / "Sales_2343.csv"
INCIDENTS_FILE = BASE_DIR / "TruCase_Incidents.csv"
INVENTORY_FILE = BASE_DIR / "Inventory_Results.csv"
EMPTY_FILE = BASE_DIR / "Empty_Packages_2343.csv"
STORE_FILE = BASE_DIR / "Store_Information.csv"
DPCI_FILE = BASE_DIR / "DPCI_Info.csv"


def to_float(value):
  if value is None:
    return 0.0
  try:
    return float(str(value).replace(",", "").strip())
  except Exception:
    return 0.0


def to_int(value, fallback=0):
  if value is None:
    return fallback
  try:
    return int(float(str(value).strip()))
  except Exception:
    return fallback


def parse_date(value, fmt):
  if not value:
    return None
  try:
    return datetime.strptime(value.strip(), fmt).date()
  except Exception:
    return None


def month_key(date_obj):
  return f"{date_obj.year:04d}-{date_obj.month:02d}"


def load_store_info():
  info = {}
  if not STORE_FILE.exists():
    return info
  with STORE_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      store = (row.get("Store_Number") or "").strip()
      if not store:
        continue
      info[store] = {
        "type": (row.get("Store_Type") or "").strip(),
        "format": (row.get("Store_Format") or "").strip(),
        "state": (row.get("State") or "").strip(),
        "region": (row.get("Region") or "").strip(),
        "group": (row.get("Group") or "").strip(),
        "district": (row.get("District") or "").strip(),
        "market": (row.get("Market") or "").strip()
      }
  return info


def load_dpci_prices():
  prices = {}
  if not DPCI_FILE.exists():
    return prices
  with DPCI_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      dpci = (row.get("DPCI") or "").strip()
      if not dpci:
        continue
      price = to_float(row.get("Official_Retail_Price_$"))
      if price:
        prices[dpci] = price
  return prices


def parse_sales():
  weekly = {}
  departments = defaultdict(lambda: {"sales": 0.0, "units": 0.0})
  monthly_sales = defaultdict(float)
  store_id = ""
  rows = 0

  with SALES_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows += 1
      store_id = store_id or (row.get("Store_Number") or "").strip()
      date = parse_date(row.get("Sale_Week"), "%m/%d/%Y")
      if not date:
        continue
      key = date.isoformat()
      entry = weekly.get(key)
      if entry is None:
        entry = {
          "week": key,
          "sales": 0.0,
          "online": 0.0,
          "driveUp": 0.0,
          "opu": 0.0,
          "shipToStore": 0.0,
          "returns": 0.0,
          "units": 0.0
        }
        weekly[key] = entry

      sales = to_float(row.get("Sales_$"))
      online = to_float(row.get("Online_Sales_$"))
      drive_up = to_float(row.get("Drive-up_Sales_$"))
      opu = to_float(row.get("OPU_$"))
      ship = to_float(row.get("Ship_to_Store_$"))
      returns = to_float(row.get("Return_$"))
      units = to_float(row.get("Sales_Units"))

      entry["sales"] += sales
      entry["online"] += online
      entry["driveUp"] += drive_up
      entry["opu"] += opu
      entry["shipToStore"] += ship
      entry["returns"] += returns
      entry["units"] += units

      dept_name = (row.get("Department_Name") or "").strip() or (row.get("Department_Number") or "").strip()
      if dept_name:
        dept = departments[dept_name]
        dept["sales"] += sales
        dept["units"] += units

      monthly_sales[month_key(date)] += sales

  weeks = sorted(weekly.values(), key=lambda item: item["week"])
  dept_list = [
    {
      "department": name,
      "sales": values["sales"],
      "units": values["units"]
    }
    for name, values in departments.items()
  ]
  dept_list.sort(key=lambda item: item["sales"], reverse=True)

  monthly_list = [
    {"month": key, "sales": value}
    for key, value in monthly_sales.items()
  ]
  monthly_list.sort(key=lambda item: item["month"])

  monthly_lookup = {}
  for item in monthly_list:
    year, month = item["month"].split("-")
    monthly_lookup[(int(year), int(month))] = item["sales"]

  boycott = []
  for month in (5, 6, 7):
    current = monthly_lookup.get((2023, month), 0.0)
    previous = monthly_lookup.get((2022, month), 0.0)
    pct = ((current - previous) / previous * 100.0) if previous else 0.0
    label = datetime(2023, month, 1).strftime("%b %Y")
    boycott.append({
      "month": f"2023-{month:02d}",
      "label": label,
      "sales": current,
      "prevSales": previous,
      "pct": pct
    })

  return {
    "store": store_id,
    "rows": rows,
    "weekly": weeks,
    "departments": dept_list[:12],
    "monthly": monthly_list,
    "boycott": boycott
  }


def parse_incidents(store_info):
  monthly = defaultdict(lambda: {"incidents": 0.0, "proven": 0.0})
  store_year = defaultdict(lambda: {"incidents": 0.0, "proven": 0.0})
  rows = 0
  max_year = 0

  with INCIDENTS_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows += 1
      store = (row.get("Store_Number") or "").strip()
      count = to_float(row.get("Total_Theft_Incidents_Count"))
      proven = to_float(row.get("Total_Theft_Proven_$"))
      raw_month = (row.get("Incident_Month_Year") or "").strip()
      if not raw_month:
        continue
      dt = None
      try:
        dt = datetime.strptime(raw_month, "%y-%b")
      except Exception:
        continue
      key = f"{dt.year:04d}-{dt.month:02d}"
      monthly[key]["incidents"] += count
      monthly[key]["proven"] += proven
      store_key = (store, dt.year)
      store_year[store_key]["incidents"] += count
      store_year[store_key]["proven"] += proven
      if dt.year > max_year:
        max_year = dt.year

  monthly_list = [
    {"month": key, "incidents": value["incidents"], "proven": value["proven"]}
    for key, value in monthly.items()
  ]
  monthly_list.sort(key=lambda item: item["month"])

  store_list = []
  counts = []
  for (store, year), value in store_year.items():
    if year != max_year:
      continue
    incidents = value["incidents"]
    proven = value["proven"]
    avg_value = proven / incidents if incidents else 0.0
    info = store_info.get(store, {})
    store_list.append({
      "store": store,
      "incidents": incidents,
      "proven": proven,
      "avgValue": avg_value,
      "region": info.get("region", ""),
      "state": info.get("state", ""),
      "format": info.get("format", "")
    })
    counts.append(incidents)

  mean = sum(counts) / len(counts) if counts else 0.0
  variance = sum((val - mean) ** 2 for val in counts) / len(counts) if counts else 0.0
  stdev = sqrt(variance) if variance else 0.0

  for item in store_list:
    if stdev:
      item["zScore"] = (item["incidents"] - mean) / stdev
    else:
      item["zScore"] = 0.0

  store_list.sort(key=lambda item: item["incidents"], reverse=True)

  regions = defaultdict(lambda: {"incidents": 0.0, "proven": 0.0, "stores": 0})
  for item in store_list:
    region = item["region"] or "Unknown"
    regions[region]["incidents"] += item["incidents"]
    regions[region]["proven"] += item["proven"]
    regions[region]["stores"] += 1

  region_list = [
    {"region": region, **values}
    for region, values in regions.items()
  ]
  region_list.sort(key=lambda item: item["incidents"], reverse=True)

  return {
    "rows": rows,
    "year": max_year,
    "monthly": monthly_list,
    "stores": store_list,
    "regions": region_list
  }


def parse_inventory(store_info):
  year_totals = defaultdict(lambda: {"sum": 0.0, "count": 0})
  store_year = defaultdict(lambda: {"sum": 0.0, "count": 0, "sales": 0.0, "shortage": 0.0})
  rows = 0
  max_year = 0

  with INVENTORY_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows += 1
      year = to_int(row.get("Year"))
      store = (row.get("Store_Number") or "").strip()
      shortage_percent = to_float(row.get("Shortage_Percent"))
      sales = to_float(row.get("Sales"))
      shortage = to_float(row.get("Shortage"))

      if year:
        year_totals[year]["sum"] += shortage_percent
        year_totals[year]["count"] += 1
        store_key = (store, year)
        store_year[store_key]["sum"] += shortage_percent
        store_year[store_key]["count"] += 1
        store_year[store_key]["sales"] += sales
        store_year[store_key]["shortage"] += shortage
        if year > max_year:
          max_year = year

  year_list = []
  for year in sorted(year_totals):
    total = year_totals[year]
    avg = total["sum"] / total["count"] if total["count"] else 0.0
    year_list.append({"year": year, "avgShortagePercent": avg})

  store_list = []
  for (store, year), values in store_year.items():
    if year != max_year:
      continue
    avg_pct = values["sum"] / values["count"] if values["count"] else 0.0
    info = store_info.get(store, {})
    store_list.append({
      "store": store,
      "shortagePercent": avg_pct,
      "sales": values["sales"],
      "shortage": values["shortage"],
      "region": info.get("region", ""),
      "state": info.get("state", ""),
      "format": info.get("format", "")
    })

  store_list.sort(key=lambda item: item["shortagePercent"], reverse=True)

  return {
    "rows": rows,
    "year": max_year,
    "years": year_list,
    "stores": store_list[:15]
  }


def parse_empty_packages(prices):
  employees = defaultdict(lambda: {"count": 0, "value": 0.0})
  areas = defaultdict(lambda: {"count": 0, "value": 0.0})
  conditions = defaultdict(lambda: {"count": 0, "value": 0.0})
  monthly = defaultdict(lambda: {"count": 0, "value": 0.0})
  rows = 0

  with EMPTY_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows += 1
      employee = (row.get("Reported_By_ID") or "").strip()
      area = (row.get("Area_Found") or "").strip()
      condition = (row.get("Package_Condition") or "").strip()
      dpci = (row.get("DPCI") or "").strip()
      quantity = to_int(row.get("Quantity"), fallback=1)
      if quantity <= 0:
        quantity = 1
      price = prices.get(dpci, 0.0)
      value = price * quantity

      if employee:
        employees[employee]["count"] += quantity
        employees[employee]["value"] += value
      if area:
        areas[area]["count"] += quantity
        areas[area]["value"] += value
      if condition:
        conditions[condition]["count"] += quantity
        conditions[condition]["value"] += value

      date = parse_date(row.get("Reported_Date"), "%m/%d/%Y")
      if date:
        key = month_key(date)
        monthly[key]["count"] += quantity
        monthly[key]["value"] += value

  def build_list(source, key_name):
    items = []
    for name, values in source.items():
      count = values["count"]
      total_value = values["value"]
      avg_value = total_value / count if count else 0.0
      items.append({
        key_name: name,
        "count": count,
        "estimatedValue": total_value,
        "avgValue": avg_value
      })
    items.sort(key=lambda item: item["estimatedValue"], reverse=True)
    return items

  employees_list = build_list(employees, "employee")[:12]
  areas_list = build_list(areas, "area")[:8]
  conditions_list = build_list(conditions, "condition")[:8]

  monthly_list = [
    {"month": key, "count": value["count"], "estimatedValue": value["value"]}
    for key, value in monthly.items()
  ]
  monthly_list.sort(key=lambda item: item["month"])

  return {
    "rows": rows,
    "employees": employees_list,
    "areas": areas_list,
    "conditions": conditions_list,
    "monthly": monthly_list
  }


def main():
  if not SALES_FILE.exists():
    raise SystemExit(f"Missing sales data: {SALES_FILE}")
  if not INCIDENTS_FILE.exists():
    raise SystemExit(f"Missing incidents data: {INCIDENTS_FILE}")
  if not INVENTORY_FILE.exists():
    raise SystemExit(f"Missing inventory data: {INVENTORY_FILE}")
  if not EMPTY_FILE.exists():
    raise SystemExit(f"Missing empty package data: {EMPTY_FILE}")

  store_info = load_store_info()
  prices = load_dpci_prices()
  sales = parse_sales()
  incidents = parse_incidents(store_info)
  inventory = parse_inventory(store_info)
  empty_packages = parse_empty_packages(prices)

  output = {
    "meta": {
      "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "salesStore": sales.get("store", ""),
      "incidentYear": incidents.get("year", 0),
      "shortageYear": inventory.get("year", 0),
      "currency": "USD"
    },
    "sales": sales,
    "incidents": incidents,
    "inventory": inventory,
    "emptyPackages": empty_packages
  }

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
  print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
  main()
