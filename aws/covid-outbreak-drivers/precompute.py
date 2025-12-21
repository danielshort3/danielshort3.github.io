import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROB_PATH = os.path.join(ROOT, "Covid-Analysis", "icu_breach_probabilities.csv")
RAW_PATH = os.path.join(ROOT, "Covid-Analysis", "data", "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

STATE_INFO = {
  "AL": {"name": "Alabama", "inMap": True},
  "AK": {"name": "Alaska", "inMap": True},
  "AZ": {"name": "Arizona", "inMap": True},
  "AR": {"name": "Arkansas", "inMap": True},
  "CA": {"name": "California", "inMap": True},
  "CO": {"name": "Colorado", "inMap": True},
  "CT": {"name": "Connecticut", "inMap": True},
  "DE": {"name": "Delaware", "inMap": True},
  "DC": {"name": "District of Columbia", "inMap": True},
  "FL": {"name": "Florida", "inMap": True},
  "GA": {"name": "Georgia", "inMap": True},
  "HI": {"name": "Hawaii", "inMap": True},
  "ID": {"name": "Idaho", "inMap": True},
  "IL": {"name": "Illinois", "inMap": True},
  "IN": {"name": "Indiana", "inMap": True},
  "IA": {"name": "Iowa", "inMap": True},
  "KS": {"name": "Kansas", "inMap": True},
  "KY": {"name": "Kentucky", "inMap": True},
  "LA": {"name": "Louisiana", "inMap": True},
  "ME": {"name": "Maine", "inMap": True},
  "MD": {"name": "Maryland", "inMap": True},
  "MA": {"name": "Massachusetts", "inMap": True},
  "MI": {"name": "Michigan", "inMap": True},
  "MN": {"name": "Minnesota", "inMap": True},
  "MS": {"name": "Mississippi", "inMap": True},
  "MO": {"name": "Missouri", "inMap": True},
  "MT": {"name": "Montana", "inMap": True},
  "NE": {"name": "Nebraska", "inMap": True},
  "NV": {"name": "Nevada", "inMap": True},
  "NH": {"name": "New Hampshire", "inMap": True},
  "NJ": {"name": "New Jersey", "inMap": True},
  "NM": {"name": "New Mexico", "inMap": True},
  "NY": {"name": "New York", "inMap": True},
  "NC": {"name": "North Carolina", "inMap": True},
  "ND": {"name": "North Dakota", "inMap": True},
  "OH": {"name": "Ohio", "inMap": True},
  "OK": {"name": "Oklahoma", "inMap": True},
  "OR": {"name": "Oregon", "inMap": True},
  "PA": {"name": "Pennsylvania", "inMap": True},
  "RI": {"name": "Rhode Island", "inMap": True},
  "SC": {"name": "South Carolina", "inMap": True},
  "SD": {"name": "South Dakota", "inMap": True},
  "TN": {"name": "Tennessee", "inMap": True},
  "TX": {"name": "Texas", "inMap": True},
  "UT": {"name": "Utah", "inMap": True},
  "VT": {"name": "Vermont", "inMap": True},
  "VA": {"name": "Virginia", "inMap": True},
  "WA": {"name": "Washington", "inMap": True},
  "WV": {"name": "West Virginia", "inMap": True},
  "WI": {"name": "Wisconsin", "inMap": True},
  "WY": {"name": "Wyoming", "inMap": True},
  "PR": {"name": "Puerto Rico", "inMap": False},
  "VI": {"name": "U.S. Virgin Islands", "inMap": False},
  "AS": {"name": "American Samoa", "inMap": False}
}

FEATURES = [
  {"key": "adult_icu_bed_utilization", "label": "Adult ICU bed utilization", "format": "pct"},
  {"key": "adult_icu_bed_covid_utilization", "label": "ICU beds with COVID", "format": "pct"},
  {"key": "inpatient_bed_covid_utilization", "label": "Inpatient beds with COVID", "format": "pct"},
  {"key": "staffed_icu_adult_patients_confirmed_covid", "label": "ICU COVID patients", "format": "count"},
  {"key": "total_adult_patients_hospitalized_confirmed_covid", "label": "Adult COVID hospitalizations", "format": "count"},
  {"key": "previous_day_admission_adult_covid_confirmed", "label": "Daily adult COVID admissions", "format": "count"},
  {"key": "critical_staffing_shortage_today_yes", "label": "Staffing shortage reports", "format": "count"}
]


def parse_float(raw):
  if raw is None:
    return None
  val = str(raw).strip()
  if val == "" or val.lower() in ("nan", "na", "n/a"):
    return None
  try:
    return float(val)
  except ValueError:
    return None


def load_probabilities(path):
  prob_map = {}
  dates = set()
  states = set()
  with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      state = (row.get("state") or "").strip().upper()
      date = (row.get("date") or "").strip()
      prob = parse_float(row.get("prob_breach_7d"))
      utilization = parse_float(row.get("adult_icu_bed_utilization"))
      if not state or not date:
        continue
      prob_map[(state, date)] = {
        "prob": prob,
        "icuUtilization": utilization
      }
      dates.add(date)
      states.add(state)
  date_counts = Counter(date for (_, date) in prob_map.keys())
  max_count = max(date_counts.values()) if date_counts else 0
  valid_dates = sorted([d for d in dates if date_counts.get(d, 0) == max_count])
  return prob_map, valid_dates, sorted(states)


def load_raw_features(path, valid_keys):
  data_by_date = defaultdict(dict)
  minmax_by_date = defaultdict(lambda: {f["key"]: [math.inf, -math.inf] for f in FEATURES})
  feature_keys = [f["key"] for f in FEATURES]
  with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      state = (row.get("state") or "").strip().upper()
      date = (row.get("date") or "").strip()
      if (state, date) not in valid_keys:
        continue
      record = {}
      for key in feature_keys:
        val = parse_float(row.get(key))
        record[key] = val
        if val is None:
          continue
        mm = minmax_by_date[date][key]
        if val < mm[0]:
          mm[0] = val
        if val > mm[1]:
          mm[1] = val
      data_by_date[date][state] = record
  return data_by_date, minmax_by_date


def risk_band(prob):
  if prob is None:
    return "unknown"
  if prob >= 0.15:
    return "high"
  if prob >= 0.05:
    return "elevated"
  return "low"


def write_json(path, payload):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, separators=(",", ":"))


def main():
  prob_map, dates, states = load_probabilities(PROB_PATH)
  raw_values, minmax = load_raw_features(RAW_PATH, prob_map)

  os.makedirs(OUT_DIR, exist_ok=True)

  meta_states = []
  for code in states:
    info = STATE_INFO.get(code, {"name": code, "inMap": False})
    meta_states.append({
      "id": code,
      "name": info["name"],
      "inMap": info["inMap"]
    })
  meta = {
    "latest": dates[-1] if dates else None,
    "dates": dates,
    "states": sorted(meta_states, key=lambda item: item["id"]),
    "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  }
  write_json(os.path.join(OUT_DIR, "meta.json"), meta)

  feature_lookup = {f["key"]: f for f in FEATURES}

  for date in dates:
    state_rows = []
    for state in states:
      key = (state, date)
      if key not in prob_map:
        continue
      base = prob_map[key]
      info = STATE_INFO.get(state, {"name": state, "inMap": False})
      features = raw_values.get(date, {}).get(state, {})
      drivers = []
      for feat in FEATURES:
        val = features.get(feat["key"])
        if val is None and feat["key"] == "adult_icu_bed_utilization":
          val = base.get("icuUtilization")
        mm = minmax[date][feat["key"]]
        if val is None or mm[0] == math.inf or mm[1] == -math.inf:
          continue
        denom = mm[1] - mm[0]
        norm = 0 if denom == 0 else (val - mm[0]) / denom
        drivers.append({
          "score": norm,
          "key": feat["key"],
          "label": feat["label"],
          "format": feat["format"],
          "value": val
        })
      drivers.sort(key=lambda item: item["score"], reverse=True)
      top_drivers = [
        {k: d[k] for k in ("key", "label", "format", "value")}
        for d in drivers[:3]
      ]
      state_rows.append({
        "id": state,
        "name": info["name"],
        "prob": base.get("prob"),
        "icuUtilization": base.get("icuUtilization"),
        "risk": risk_band(base.get("prob")),
        "drivers": top_drivers,
        "inMap": info["inMap"]
      })
    state_rows.sort(key=lambda item: (item.get("prob") is None, -(item.get("prob") or 0)))
    for idx, row in enumerate(state_rows, start=1):
      row["rank"] = idx
    hotspots = [
      {k: row[k] for k in ("id", "name", "prob", "risk")}
      for row in state_rows[:5]
    ]
    payload = {
      "date": date,
      "states": state_rows,
      "hotspots": hotspots
    }
    write_json(os.path.join(OUT_DIR, "by-date", f"{date}.json"), payload)

  for state in states:
    info = STATE_INFO.get(state, {"name": state, "inMap": False})
    history = []
    for date in dates:
      key = (state, date)
      if key not in prob_map:
        continue
      base = prob_map[key]
      history.append({
        "date": date,
        "prob": base.get("prob"),
        "icuUtilization": base.get("icuUtilization")
      })
    payload = {
      "state": {
        "id": state,
        "name": info["name"],
        "inMap": info["inMap"]
      },
      "history": history
    }
    write_json(os.path.join(OUT_DIR, "state", f"{state}.json"), payload)


if __name__ == "__main__":
  main()
