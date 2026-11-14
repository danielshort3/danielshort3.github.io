import copy
import http.client
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from unittest import TestCase, main
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

TEMPLATE = Path(__file__).with_name("template.yaml").read_text(encoding="utf-8")
NOW = datetime(2026, 11, 13, 3, 10, tzinfo=timezone.utc)
ACCOUNT = "886623862678"


class AwsError(Exception):
  def __init__(self, code):
    self.response = {"Error": {"Code": code}}


class StubClock(datetime):
  @classmethod
  def now(cls, _timezone):
    return NOW


class MemoryTable:
  def __init__(self):
    self.rows = {}
    self.query_requests = []
    self.write_count = 0
    self.conflicts = 0
    self.page_size = None
    self.query_errors = {}

  def get_item(self, Key, ConsistentRead):
    assert ConsistentRead
    return {"Item": copy.deepcopy(self.rows.get((Key["pk"], Key["sk"])))}

  def put_item(self, Item, ConditionExpression, ExpressionAttributeValues=None):
    if self.conflicts:
      self.conflicts -= 1
      raise AwsError("ConditionalCheckFailedException")
    key = (Item["pk"], Item["sk"])
    if ConditionExpression == "attribute_not_exists(pk)":
      assert key not in self.rows
    else:
      assert ConditionExpression == "rev = :expected"
      assert self.rows[key]["rev"] == ExpressionAttributeValues[":expected"]
    self.rows[key] = copy.deepcopy(Item)
    self.write_count += 1

  def query(self, **request):
    self.query_requests.append(request)
    values = request["ExpressionAttributeValues"]
    pk, start, end = (values[key]["S"] for key in (":partition", ":first", ":last"))
    errors = self.query_errors.get(start)
    if errors:
      raise errors.pop(0)
    after = request.get("ExclusiveStartKey", {}).get("sk", {}).get("S", "")
    items = [copy.deepcopy(item) for (p, k), item in sorted(self.rows.items()) if p == pk and start <= k <= end and k > after]
    assert len(items) < request["Limit"]
    if self.page_size and start.startswith("report#") and len(items) > self.page_size:
      items = items[:self.page_size]
      return {"Items": items, "LastEvaluatedKey": {"pk": {"S": pk}, "sk": {"S": items[-1]["sk"]}}}
    return {"Items": items}


def load_runtime(name, table, cloudwatch, sagemaker):
  marker = re.search(rf"^  {name}:\n.*?^        ZipFile: \|\n", TEMPLATE, re.M | re.S)
  if not marker:
    raise AssertionError("Missing inline runtime")
  modules = {
    key: ModuleType(key) for key in (
      "boto3", "boto3.dynamodb", "boto3.dynamodb.types",
      "botocore", "botocore.config", "botocore.exceptions",
    )
  }
  def client(service, region_name, config):
    assert region_name == "us-east-2"
    return {"cloudwatch": cloudwatch, "sagemaker": sagemaker, "dynamodb": table}[service]
  boto = modules["boto3"]
  boto.client = Mock(side_effect=client)
  boto.resource = Mock(return_value=Mock(Table=Mock(return_value=table)))
  modules["boto3.dynamodb.types"].TypeDeserializer = lambda: Mock(deserialize=lambda item: item["M"])
  modules["botocore.config"].Config = Mock()
  modules["botocore.exceptions"].ClientError = AwsError
  modules["botocore.exceptions"].BotoCoreError = OSError
  environment = {"HISTORY_TABLE": "fixture", "STACK_ACCOUNT": ACCOUNT, "SHORT_LINK_SLUG": ""}
  runtime = ModuleType(name)
  lines = []
  for line in TEMPLATE[marker.end():].splitlines(keepends=True):
    if line.strip() and not line.startswith("          "):
      break
    lines.append(line)
  with patch.dict("sys.modules", modules), patch.dict(os.environ, environment):
    exec(compile(dedent("".join(lines)), f"template.yaml:{name}", "exec"), vars(runtime))
  runtime.datetime = StubClock
  return runtime


def sample_report(identifier, state, source, checked=None, expires=None):
  return {"id": identifier, "state": state, "source": source,
          "lastChecked": (checked or NOW).isoformat(),
          "expiresAt": (expires or NOW + timedelta(hours=1)).isoformat()}


class CollectorTests(TestCase):
  def setUp(self):
    self.table, self.cloudwatch, self.sagemaker = MemoryTable(), Mock(), Mock()
    self.reader = load_runtime("Collector", self.table, self.cloudwatch, self.sagemaker)
    self.writer = load_runtime("Writer", self.table, self.cloudwatch, self.sagemaker)
    self.report = load_runtime("Report", self.table, self.cloudwatch, self.sagemaker)
    self.context = Mock(
      get_remaining_time_in_millis=Mock(return_value=20000),
      invoked_function_arn=f"arn:aws:lambda:us-east-2:{ACCOUNT}:function:website-aws-status-report",
    )

  def read(self):
    return self.reader.handler({}, self.context)

  def test_empty_reader_is_read_only_and_produces_ninety_ordered_mountain_days(self):
    result = self.read()
    self.assertTrue(result["ok"])
    self.assertIsNone(result["checkedAt"])
    self.assertEqual(result["days"], 90)
    self.assertEqual(result["checks"], [])
    self.assertEqual([item["id"] for item in result["connections"]],
                     ["cognito", "tracker", "short-links", "contact", "transcribe", "chatbot", "markdown"])
    self.assertEqual(len(self.table.query_requests), 10)
    self.assertEqual(result["catalogHistory"], {
      "hourStart": "2026-11-10T04:00:00.000Z", "dayStart": "2026-08-15",
      "observedAt": [None] * 72, "results": {},
    })
    for item in result["connections"]:
      days = item["days"]
      self.assertEqual(len(days), 90)
      self.assertEqual(days[-1]["date"], "2026-11-12")
      self.assertEqual(days[-1]["from"], "2026-11-12T07:00:00.000Z")
      self.assertEqual(days[-1]["to"], "2026-11-13T07:00:00.000Z")
      self.assertTrue(all(day["state"] == "unknown" for day in days))
      hours = item["hours"]
      self.assertEqual(len(hours), 72)
      self.assertEqual(hours[0]["start"], "2026-11-10T04:00:00.000Z")
      self.assertEqual(hours[-1]["start"], "2026-11-13T03:00:00.000Z")
      self.assertEqual(hours[-1]["to"], "2026-11-13T04:00:00.000Z")
      self.assertTrue(all(hour["state"] == "unknown" and "observedAt" not in hour for hour in hours))
    self.assertFalse(self.cloudwatch.mock_calls)
    self.assertFalse(self.sagemaker.mock_calls)
    self.assertEqual(self.table.write_count, 0)

  def test_mountain_day_matches_iana_dst_boundaries_including_fall_repeat(self):
    mountain = ZoneInfo("America/Denver")
    for day in (date(2026, 3, 8), date(2026, 11, 1), date(2027, 3, 14), date(2027, 11, 7)):
      start, end = self.reader.day_bounds(day)
      self.assertEqual(start, datetime.combine(day, datetime.min.time(), mountain).astimezone(timezone.utc))
      self.assertEqual(end, datetime.combine(day + timedelta(days=1), datetime.min.time(), mountain).astimezone(timezone.utc))
      self.assertIn((end - start).total_seconds() / 3600, (23, 25))
      self.assertEqual(self.writer.day_range(start)[:3], (day.isoformat(), start, end))
      self.assertEqual(self.reader.mountain_day(end - timedelta(seconds=1)), day)
      self.assertEqual(self.reader.mountain_day(end), day + timedelta(days=1))

  def test_hourly_uses_distinct_utc_samples_across_the_repeated_mountain_hour(self):
    first = datetime(2026, 11, 1, 7, tzinfo=timezone.utc)
    second = first + timedelta(hours=1)
    now = second + timedelta(minutes=20)
    for at, code in ((first, "a"), (second, "m")):
      self.writer.sample("tracker", 3600, at, at + timedelta(minutes=10), code, self.context)
    with patch.object(StubClock, "now", return_value=now):
      first_tile, second_tile = self.read()["connections"][1]["hours"][-2:]
    mountain = ZoneInfo("America/Denver")
    local = [self.reader.parse_time(tile["start"]).astimezone(mountain) for tile in (first_tile, second_tile)]
    self.assertEqual([at.hour for at in local], [1, 1])
    self.assertEqual([at.tzname() for at in local], ["MDT", "MST"])
    self.assertEqual([tile["state"] for tile in (first_tile, second_tile)], ["operational", "degraded"])
    self.assertNotIn("observedAt", first_tile)
    self.assertEqual(second_tile["observedAt"], "2026-11-01T08:10:00.000Z")

  def test_hourly_does_not_mark_partial_or_missing_probes_operational(self):
    current = NOW.replace(minute=0)
    complete, gap = current - timedelta(hours=2), current - timedelta(hours=1)
    for index in range(12):
      at = complete + timedelta(minutes=5 * index)
      self.writer.sample("cognito", 300, at, at, "o", self.context)
    for hour in (gap, current):
      for index in (0, 2):
        at = hour + timedelta(minutes=5 * index)
        self.writer.sample("cognito", 300, at, at, "o", self.context)
    hours = self.read()["connections"][0]["hours"]
    self.assertEqual([hour["state"] for hour in hours[-3:]], ["operational", "unknown", "unknown"])
    self.assertIn("incomplete", hours[-1]["issue"])
    self.assertNotIn("observedAt", hours[-2])
    self.assertEqual(hours[-1]["observedAt"], self.reader.iso(NOW))

  def test_hourly_samples_do_not_include_a_future_scheduled_slot(self):
    current = NOW.replace(minute=0)
    future = NOW + timedelta(minutes=5)
    self.writer.sample("cognito", 300, future, NOW, "f", self.context)
    latest = self.read()["connections"][0]["hours"][-1]
    self.assertEqual(latest["start"], self.reader.iso(current))
    self.assertEqual(latest["state"], "unknown")

  def test_hourly_sample_recovery_at_the_next_boundary_does_not_leak_to_next_bucket(self):
    recovered = NOW.replace(minute=0)
    failed = recovered - timedelta(minutes=5)
    self.writer.sample("cognito", 300, failed, failed, "f", self.context)
    self.writer.sample("cognito", 300, recovered, recovered, "o", self.context)
    self.writer.sample("cognito", 300, NOW, NOW, "d", self.context)
    previous, current = self.read()["connections"][0]["hours"][-2:]
    self.assertEqual((previous["state"], previous["from"], previous["to"]), (
      "outage", self.reader.iso(failed), self.reader.iso(recovered)
    ))
    self.assertEqual(previous["issue"], "The scheduled HTTPS check returned a server error.")
    self.assertEqual((current["state"], current["from"]), ("degraded", self.reader.iso(NOW)))

  def test_https_probes_only_fixed_safe_routes_and_disabled_optional_slug(self):
    connections = {}
    def connect(host, port, timeout):
      self.assertEqual((port, timeout), (443, 3))
      response = Mock(status=200)
      if host == self.writer.COGNITO_HOST:
        response.read.return_value = json.dumps({
          "issuer": self.writer.COGNITO_ISSUER,
          "jwks_uri": self.writer.COGNITO_ISSUER + "/.well-known/jwks.json",
          "private": "ignore this",
        }).encode()
      result = Mock(getresponse=Mock(return_value=response))
      connections[host] = result
      return result
    with patch.object(http.client, "HTTPSConnection", side_effect=connect):
      result = self.writer.handler({"operation": "probes"}, self.context)
    self.assertEqual(result, {"ok": True, "collected": 2})
    self.assertEqual(set(connections), {self.writer.HOME_HOST, self.writer.COGNITO_HOST})
    self.assertEqual(connections[self.writer.HOME_HOST].request.call_args.args, ("HEAD", "/"))
    self.assertEqual(connections[self.writer.COGNITO_HOST].request.call_args.args,
                     ("GET", self.writer.COGNITO_PATH))
    self.assertTrue(all(connection.close.call_count == 1 for connection in connections.values()))
    snapshot = self.read()
    self.assertEqual(snapshot["checks"][0]["id"], "project:website")
    self.assertEqual(snapshot["checks"][0]["state"], "operational")
    self.assertNotIn("ignore this", json.dumps(snapshot))
    self.assertNotIn(("short-links", "2026-11-12"), self.table.rows)

  def test_optional_short_links_head_accepts_only_exact_location(self):
    response = Mock(status=302)
    response.getheader.return_value = self.writer.SHORT_TARGET
    self.assertTrue(self.writer.redirect_response(response))
    for bad in ("https://www.danielshort.me/tools/aws-status?tracking=1", "https://another.example/"):
      response.getheader.return_value = bad
      self.assertFalse(self.writer.redirect_response(response))

  def test_hourly_fixed_requests_missing_metrics_and_idle_endpoints(self):
    self.cloudwatch.describe_alarms.return_value = {
      "MetricAlarms": [{"AlarmName": name, "StateValue": "OK"} for name in self.writer.ALARMS]
    }
    self.sagemaker.describe_endpoint.side_effect = lambda EndpointName: {
      "EndpointName": EndpointName, "EndpointStatus": "InService",
      "ProductionVariants": [{"CurrentInstanceCount": 0, "DesiredInstanceCount": 0}],
    }
    self.cloudwatch.get_metric_statistics.return_value = {"Datapoints": []}
    self.assertEqual(self.writer.handler({"operation": "hourly"}, self.context), {"ok": True, "collected": 5})
    request = self.cloudwatch.describe_alarms.call_args.kwargs
    self.assertEqual(set(request["AlarmNames"]), set(self.writer.ALARMS))
    self.assertEqual(request["AlarmTypes"], ["MetricAlarm"])
    self.assertEqual(self.cloudwatch.get_metric_statistics.call_count, 4)
    for call in self.cloudwatch.get_metric_statistics.call_args_list:
      self.assertEqual(call.kwargs["Period"], 3600)
      self.assertEqual(call.kwargs["EndTime"], NOW.replace(minute=0))
      dimensions = {row["Name"]: row["Value"] for row in call.kwargs["Dimensions"]}
      if call.kwargs["Namespace"] == "AWS/Transcribe":
        self.assertEqual(dimensions, {"ServiceType": "Batch", "Operation": "StartTranscriptionJob", "Domain": "Transcribe"})
      else:
        self.assertEqual(set(dimensions), {"EndpointName", "VariantName"})
        self.assertIn(dimensions["EndpointName"], self.writer.ENDPOINTS.values())
        self.assertEqual(dimensions["VariantName"], "AllTraffic")
    self.assertEqual({call.kwargs["EndpointName"] for call in self.sagemaker.describe_endpoint.call_args_list},
                     set(self.writer.ENDPOINTS.values()))
    self.assertEqual(set(self.table.rows[("chatbot", "2026-11-12")]["samples"]) - {"-"}, {"i"})
    self.assertEqual(set(self.table.rows[("transcribe", "2026-11-12")]["samples"]) - {"-"}, {"n"})

  def test_missing_coverage_and_duplicate_observations_do_not_rewrite(self):
    day = self.writer.mountain_day(NOW)
    start, _end = self.reader.day_bounds(day)
    now = start + timedelta(minutes=17)
    self.table.conflicts = 1
    self.writer.sample("cognito", 300, start, start, "o", self.context)
    self.writer.sample("cognito", 300, start, start + timedelta(seconds=1), "f", self.context)
    self.writer.sample("cognito", 300, start + timedelta(minutes=10), start + timedelta(minutes=10), "o", self.context)
    self.assertEqual(self.table.write_count, 2)
    row = self.table.rows[("cognito", day.isoformat())]
    self.assertEqual(row["samples"], "o-o")
    tile = self.reader.tile(day, row, 300, now)
    self.assertEqual(tile["state"], "unknown")
    self.assertEqual(tile["issue"], "Monitoring coverage for this day is incomplete.")

  def test_alarm_incident_uses_exact_transition_and_recovery_without_raw_fields(self):
    name = "website-job-tracker-api-5xx"
    down, up = NOW - timedelta(minutes=25), NOW - timedelta(minutes=15)
    def alarm_event(previous, previous_at, state, at):
      return {
        "source": "aws.cloudwatch", "detail-type": "CloudWatch Alarm State Change",
        "region": "us-east-2", "account": ACCOUNT,
        "detail": {"alarmName": name, "previousState": {"value": previous, "timestamp": previous_at.isoformat()},
                   "state": {"value": state, "timestamp": at.isoformat(), "reason": "private reasons"}},
      }
    raised = alarm_event("OK", down - timedelta(hours=1), "ALARM", down)
    restored = alarm_event("ALARM", down, "OK", up)
    for event in (restored, raised, restored):
      self.assertTrue(self.writer.handler(event, self.context)["ok"])
    self.assertEqual(self.table.write_count, 1)
    tracker = self.read()["connections"][1]["days"][-1]
    self.assertEqual(tracker["state"], "degraded")
    self.assertEqual((tracker["from"], tracker["to"]), (self.reader.iso(down), self.reader.iso(up)))
    self.assertEqual(tracker["issue"], "Tracker API 5xx alarm entered ALARM.")
    self.assertNotIn("private reasons", repr(self.table.rows))

  def test_hourly_alarm_bounds_stop_at_hour_edge_and_exclude_an_exact_boundary_recovery(self):
    day = self.reader.mountain_day(NOW)
    current = NOW.replace(minute=0)
    down, up = current - timedelta(minutes=20), NOW - timedelta(minutes=4)
    row = {
      "checked": self.reader.iso(NOW),
      "events": [[self.reader.iso(down), "ta", "ALARM"], [self.reader.iso(up), "ta", "OK"]],
    }
    earlier = self.reader.hour_tile(current - timedelta(hours=1), day, row, 3600, NOW)
    latest = self.reader.hour_tile(current, day, row, 3600, NOW)
    self.assertEqual(earlier["state"], "degraded")
    self.assertEqual(earlier["from"], self.reader.iso(down))
    self.assertEqual(earlier["to"], self.reader.iso(current))
    self.assertIn("Recovery was observed after this hour.", earlier["issue"])
    self.assertEqual(latest["from"], self.reader.iso(current))
    self.assertEqual(latest["to"], self.reader.iso(up))
    self.assertEqual(latest["state"], "degraded")
    row["events"][1][0] = self.reader.iso(current)
    self.assertEqual(self.reader.hour_tile(current, day, row, 3600, NOW)["state"], "unknown")

  def test_hourly_snapshot_with_full_history_and_catalog_stays_under_api_limit(self):
    today = self.reader.mountain_day(NOW)
    catalog_ids = [f"project:{index:02}" + "c" * 62 for index in range(70)]
    for identifier, _label, cadence in self.reader.SERVICES:
      for offset in range(90):
        day = today - timedelta(days=offset)
        start, end = self.reader.day_bounds(day)
        checked = min(NOW, end - timedelta(seconds=1))
        count = int((checked - start).total_seconds()) // cadence + 1
        start_text = self.reader.iso(start + timedelta(seconds=10))
        self.table.rows[(identifier, day.isoformat())] = {
          "pk": identifier, "sk": day.isoformat(), "samples": "f" * count, "checked": self.reader.iso(checked),
          "events": [[start_text, "ta", "ALARM"]], "truncated": True,
          "expires": int(self.reader.mountain_midnight(day + timedelta(days=90)).timestamp()),
        }
    for source in ("browser", "local", "production"):
      self.table.rows[("catalog", source)] = {
        "sk": source, "results": {
          identifier: [
            "operational", self.reader.iso(NOW), self.reader.iso(NOW + timedelta(hours=2))
          ] for identifier in catalog_ids
        },
      }
    for offset in range(90):
      day = today - timedelta(days=offset)
      start, end = self.reader.day_bounds(day)
      slots = (end - start) // timedelta(hours=1)
      hour_starts = [start + timedelta(hours=index) for index in range(slots)]
      stamps = [self.reader.iso(at) if at <= NOW else "" for at in hour_starts]
      states = "".join("o" if stamp else "-" for stamp in stamps)
      key = "report#" + day.isoformat()
      archive = {
        "pk": "catalog", "sk": key, "observed": stamps, "results": dict.fromkeys(catalog_ids, states),
        "expires": int(self.reader.mountain_midnight(day + timedelta(days=90)).timestamp()), "rev": 24,
      }
      self.assertLess(len(json.dumps(archive).encode("utf-8")), 8 * 1024)
      self.table.rows[("catalog", key)] = archive
    snapshot = self.read()
    self.assertTrue(snapshot["ok"])
    self.assertEqual(len(snapshot["checks"]), 210)
    self.assertEqual(len(snapshot["catalogHistory"]["results"]), 70)
    self.assertEqual(snapshot["catalogHistory"]["results"][catalog_ids[0]], {"hours": "o" * 72, "days": "o" * 90})
    self.assertLess(len(json.dumps(snapshot).encode("utf-8")), 384 * 1024)

  def test_report_validation_sources_order_and_expiry(self):
    identifier = "project:smartSentence"
    requests = [
      sample_report(identifier, "outage", "production"),
      sample_report(identifier, "operational", "browser", checked=NOW - timedelta(minutes=1)),
    ]
    for check in requests:
      self.assertEqual(self.report.handler({"checks": [check]}, self.context), {"ok": True, "accepted": 1})
    self.assertEqual(self.report.handler({"checks": [requests[0]]}, self.context), {"ok": True, "accepted": 1})
    older = sample_report(identifier, "operational", "production", checked=NOW - timedelta(minutes=1))
    self.assertEqual(self.report.handler({"checks": [older]}, self.context), {"ok": True, "accepted": 0})
    self.assertEqual(self.table.write_count, 2)
    self.assertEqual(self.read()["catalogHistory"]["results"], {})
    for event in ({"checks": [{**requests[0], "issue": "private"}]},
                  {"checks": [requests[0], requests[0]]},
                  {"checks": [sample_report("invalid:id", "outage", "production")]},
                  {"checks": [sample_report(identifier, "outage", "production", expires=NOW)]},
                  {"checks": [sample_report(identifier, "outage", "production", checked=NOW + timedelta(minutes=5))]}):
      with self.assertLogs(self.report.LOGGER, "WARNING") as captured:
        self.assertFalse(self.report.handler(event, self.context)["ok"])
      self.assertNotIn("private", " ".join(captured.output))
    self.assertEqual(self.table.write_count, 2)
    results = {check["source"]: check["state"] for check in self.read()["checks"]}
    self.assertEqual(results, {"browser": "operational", "production": "outage"})

  def test_expired_results_hidden_even_while_dynamodb_retains_items(self):
    self.table.rows[("catalog", "local")] = {
      "pk": "catalog", "sk": "local", "expires": int((NOW - timedelta(seconds=1)).timestamp()),
      "results": {"tool:transcribe": ["operational", self.reader.iso(NOW - timedelta(days=1)),
                                     self.reader.iso(NOW - timedelta(seconds=1))]},
    }
    self.assertEqual(self.read()["checks"], [])

  def test_browser_alias_rejects_non_browser_sources_and_long_expiry_before_writing(self):
    self.context.invoked_function_arn += ":browser-ci"
    valid = sample_report("project:website", "operational", "browser", expires=NOW + timedelta(hours=3))
    invalid = [
      {"checks": [valid, sample_report("tool:transcribe", "operational", "production")]},
      {"checks": [valid, sample_report("tool:transcribe", "operational", "browser", checked=NOW - timedelta(minutes=1))]},
      {"checks": [{**valid, "source": "local"}]},
      {"checks": [{**valid, "expiresAt": (NOW + timedelta(hours=3, milliseconds=1)).isoformat(timespec="milliseconds")}]},
      {"checks": [{**valid, "lastChecked": (NOW - timedelta(minutes=1)).isoformat()}]},
      {"checks": [{**valid, "lastChecked": (NOW + timedelta(seconds=30)).isoformat(),
                   "expiresAt": (NOW + timedelta(hours=3, seconds=30)).isoformat()}]},
    ]
    for event in invalid:
      with self.subTest(event=event), self.assertLogs(self.report.LOGGER, "WARNING"):
        self.assertFalse(self.report.handler(event, self.context)["ok"])
    self.assertEqual(self.table.write_count, 0)
    self.assertEqual(self.report.handler({"checks": [valid]}, self.context), {"ok": True, "accepted": 1})
    self.assertEqual(self.table.rows[("catalog", "browser")]["expires"], int((NOW + timedelta(hours=3)).timestamp()))

  def publish_browser(self, checked, outcomes):
    self.context.invoked_function_arn = f"arn:aws:lambda:us-east-2:{ACCOUNT}:function:website-aws-status-report:browser-ci"
    return self.report.handler({"checks": [sample_report(identifier, state, "browser", checked, checked + timedelta(hours=3))
                                           for identifier, state in outcomes.items()]}, self.context)

  def test_browser_archive_records_actual_hours_with_independent_freshness_and_older_retries(self):
    early = NOW - timedelta(hours=2, minutes=5)
    later = NOW - timedelta(hours=2)
    newest = NOW - timedelta(minutes=1)
    self.table.conflicts = 2
    self.assertEqual(self.publish_browser(early, {"tool:one": "degraded", "tool:former": "operational"})["accepted"], 2)
    self.assertEqual(self.publish_browser(later, {"tool:one": "outage"})["accepted"], 1)
    self.assertEqual(self.publish_browser(newest, {"tool:one": "unknown"})["accepted"], 1)
    writes = self.table.write_count
    self.publish_browser(early, {"tool:one": "degraded"})
    self.publish_browser(newest, {"tool:one": "unknown"})
    self.assertEqual(self.table.write_count, writes)
    day = self.reader.mountain_day(newest)
    archive = self.table.rows[("catalog", "report#" + day.isoformat())]
    self.assertNotIn("tool:former", archive["results"])
    self.assertEqual(archive["expires"], int(self.reader.mountain_midnight(day + timedelta(days=90)).timestamp()))
    history = self.read()["catalogHistory"]
    self.assertEqual(history["results"]["tool:one"]["hours"][-3:], "x-u")
    self.assertEqual(history["observedAt"][-3:], [self.reader.iso(later), None, self.reader.iso(newest)])
    self.assertEqual(history["results"]["tool:one"]["days"][-1], "x")
    with patch.object(StubClock, "now", return_value=NOW + timedelta(hours=3)):
      snapshot = self.read()
    self.assertEqual(snapshot["checks"], [])
    self.assertIn("x", snapshot["catalogHistory"]["results"]["tool:one"]["hours"])

  def test_browser_archive_must_be_durable_before_successful_report_receipt(self):
    self.table.conflicts = 3
    with self.assertLogs(self.report.LOGGER, "WARNING"):
      self.assertFalse(self.publish_browser(NOW, {"tool:one": "operational"})["ok"])
    self.assertEqual(self.table.write_count, 0)
    self.assertEqual(self.publish_browser(NOW, {"tool:one": "operational"}), {"ok": True, "accepted": 1})
    next_minute = NOW + timedelta(minutes=1)
    with patch.object(self.report, "record_source", side_effect=AwsError("ValidationException")), \
         patch.object(StubClock, "now", return_value=next_minute), self.assertLogs(self.report.LOGGER, "WARNING"):
      self.assertFalse(self.publish_browser(next_minute, {"tool:one": "degraded"})["ok"])
    with patch.object(StubClock, "now", return_value=next_minute):
      before = self.table.write_count
      self.assertEqual(self.publish_browser(next_minute, {"tool:one": "degraded"}), {"ok": True, "accepted": 1})
      self.assertEqual(self.table.write_count, before + 1)
      history = self.read()["catalogHistory"]
    self.assertEqual(history["results"]["tool:one"]["hours"][-1], "d")

  def test_browser_archive_dst_actual_hours_and_daily_coverage(self):
    mountain = ZoneInfo("America/Denver")
    for day, slots in ((date(2026, 3, 8), 23), (date(2026, 11, 1), 25)):
      with self.subTest(day=day):
        start, end = self.reader.day_bounds(day)
        for offset in range(slots):
          checked = start + timedelta(hours=offset, minutes=2)
          with patch.object(StubClock, "now", return_value=checked):
            self.assertTrue(self.publish_browser(checked, {"tool:one": "operational", "tool:two": "unknown"})["ok"])
        with patch.object(StubClock, "now", return_value=end + timedelta(minutes=2)):
          history = self.read()["catalogHistory"]
        self.assertEqual(history["results"]["tool:one"]["hours"][-slots-1:-1], "o" * slots)
        self.assertEqual(history["results"]["tool:one"]["days"][-2:], "o-")
        self.assertEqual(history["results"]["tool:two"]["days"][-2:], "u-")
        if slots == 25:
          repeat = [self.reader.parse_time(stamp).astimezone(mountain) for stamp in history["observedAt"][-slots-1:-1]
                    if self.reader.parse_time(stamp).astimezone(mountain).hour == 1]
          self.assertEqual([stamp.tzname() for stamp in repeat], ["MDT", "MST"])

  def test_archive_read_retries_and_paginates_without_breaking_current_results(self):
    before_midnight = self.reader.mountain_midnight(self.reader.mountain_day(NOW)) - timedelta(minutes=5)
    with patch.object(StubClock, "now", return_value=before_midnight):
      self.assertTrue(self.publish_browser(before_midnight, {"tool:one": "operational"})["ok"])
    self.assertTrue(self.publish_browser(NOW, {"tool:one": "operational"})["ok"])
    first = self.reader.mountain_day(NOW) - timedelta(days=89)
    query_key = "report#" + first.isoformat()
    self.table.page_size = 1
    self.table.query_errors[query_key] = [AwsError("ThrottlingException")]
    snapshot = self.read()
    self.assertTrue(snapshot["ok"])
    self.assertEqual(snapshot["catalogHistory"]["results"]["tool:one"]["days"][-2:], "uu")
    self.assertTrue(any("ExclusiveStartKey" in request for request in self.table.query_requests))
    self.assertEqual(snapshot["checks"][0]["state"], "operational")
    self.table.query_errors[query_key] = [AwsError("ThrottlingException"), AwsError("ThrottlingException")]
    with self.assertLogs(self.reader.LOGGER, "WARNING"):
      snapshot = self.read()
    self.assertTrue(snapshot["ok"])
    self.assertNotIn("catalogHistory", snapshot)
    self.assertEqual(snapshot["checks"][0]["state"], "operational")

  def test_browser_archive_prefers_current_catalog_ids_when_the_catalog_changes(self):
    earlier = NOW - timedelta(hours=1)
    current = {f"tool:current-{index}": "operational" for index in range(69)}
    self.assertEqual(self.publish_browser(earlier, {**current, "tool:retired": "degraded"})["accepted"], 70)
    # An expired current result can still appear in the archive's previous hour.
    self.table.rows[("catalog", "browser")]["results"]["tool:retired"][2] = self.reader.iso(NOW)
    self.assertEqual(self.publish_browser(NOW, {**current, "tool:new": "unknown"})["accepted"], 70)
    history = self.read()["catalogHistory"]["results"]
    self.assertEqual(len(history), 70)
    self.assertNotIn("tool:retired", history)
    self.assertEqual(history["tool:current-0"]["hours"][-2:], "oo")
    self.assertEqual(history["tool:new"]["hours"][-2:], "-u")

  def test_manual_report_keeps_its_longer_ttl_and_source_selection(self):
    checks = [
      sample_report("project:website", "operational", "production", expires=NOW + timedelta(days=7)),
      sample_report("tool:text-compare", "operational", "local", expires=NOW + timedelta(days=7)),
    ]
    self.assertEqual(self.report.handler({"checks": checks}, self.context), {"ok": True, "accepted": 2})

  def test_github_stack_pins_role_trust_and_single_alias_scope(self):
    template = Path(__file__).with_name("github-reporter.yaml").read_text(encoding="utf-8")
    self.assertIn("Default: 'false'", template)
    self.assertIn("CodeSha256: !Ref ReportCodeSha256", template)
    self.assertIn("CodeSha256: !Ref HistoryReportCodeSha256", template)
    self.assertIn("FunctionVersion: !If [HistoryEnabled, !GetAtt BrowserHistoryReportVersion.Version, !GetAtt BrowserReportVersion.Version]", template)
    self.assertIn("Name: browser-ci", template)
    role = template.split("  GithubReporterRole:\n")[1].split("\nOutputs:")[0]
    self.assertIn("'token.actions.githubusercontent.com:aud': sts.amazonaws.com", role)
    self.assertIn("'token.actions.githubusercontent.com:sub': 'repo:danielshort3/danielshort3.github.io:ref:refs/heads/main'", role)
    self.assertIn("'token.actions.githubusercontent.com:repository_id': '698339955'", role)
    self.assertIn("'token.actions.githubusercontent.com:repository_owner_id': '143550027'", role)
    self.assertIn("status-report-publish.yml@refs/heads/main", role)
    self.assertEqual(re.findall(r"^\s+Action: (.+)$", role, re.M), [
      "sts:AssumeRoleWithWebIdentity", "lambda:InvokeFunction",
    ])
    self.assertEqual(re.findall(r"^\s+Resource: (.+)$", role, re.M), ["!Ref BrowserReportAlias"])

  def test_scope_limits_in_template(self):
    # The private S3 TemplateURL allows up to 1 MB.
    self.assertLessEqual(len(TEMPLATE.encode()), 1_000_000)
    self.assertNotIn("AWS::CloudWatch::Alarm", TEMPLATE)
    self.assertNotIn("sagemaker:InvokeEndpoint", TEMPLATE)
    self.assertNotIn("transcribe:StartTranscriptionJob", TEMPLATE)
    self.assertNotIn("ses:SendEmail", TEMPLATE)
    invoker = TEMPLATE.split("  WebsiteInvokerRole:\n")[1].split("\nOutputs:")[0]
    self.assertIn("project:website:environment:production", invoker)
    self.assertIn("Resource: !GetAtt Collector.Arn", invoker)
    self.assertNotIn("Writer.Arn", invoker)
    self.assertNotIn("Report.Arn", invoker)
    self.assertIn("Default: 'false'", TEMPLATE)


if __name__ == "__main__":
  main()
