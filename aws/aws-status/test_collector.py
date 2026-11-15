import copy
import http.client
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
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


def load_runtime(name, table, cloudwatch, sagemaker, home_gpu=False, transcribe=None):
  if transcribe is None:
    transcribe = Mock()
    transcribe.list_transcription_jobs.return_value = {"TranscriptionJobSummaries": []}
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
    return {"cloudwatch": cloudwatch, "sagemaker": sagemaker, "dynamodb": table, "transcribe": transcribe}[service]
  boto = modules["boto3"]
  boto.client = Mock(side_effect=client)
  boto.resource = Mock(return_value=Mock(Table=Mock(return_value=table)))
  modules["boto3.dynamodb.types"].TypeDeserializer = lambda: Mock(deserialize=lambda item: item["M"])
  modules["botocore.config"].Config = Mock()
  modules["botocore.exceptions"].ClientError = AwsError
  modules["botocore.exceptions"].BotoCoreError = OSError
  environment = {"HISTORY_TABLE": "fixture", "STACK_ACCOUNT": ACCOUNT, "SHORT_LINK_SLUG": "",
                 "HOME_GPU_MONITORING": "true" if home_gpu else "false"}
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


def transcription_job(name, state, created, finished=None):
  job = {"TranscriptionJobName": "site-transcribe-" + name * 64, "CreationTime": created,
         "TranscriptionJobStatus": state}
  if finished is not None:
    job["CompletionTime"] = finished
  if state == "FAILED":
    job["FailureReason"] = "private fixture reason"
  return job


def sample_report(identifier, state, source, checked=None, expires=None):
  return {"id": identifier, "state": state, "source": source,
          "lastChecked": (checked or NOW).isoformat(),
          "expiresAt": (expires or NOW + timedelta(hours=1)).isoformat()}


class CollectorTests(TestCase):
  def setUp(self):
    self.table, self.cloudwatch, self.sagemaker = MemoryTable(), Mock(), Mock()
    self.reader = load_runtime("Collector", self.table, self.cloudwatch, self.sagemaker)
    self.transcribe = Mock()
    self.transcribe.list_transcription_jobs.return_value = {"TranscriptionJobSummaries": []}
    self.writer = load_runtime("Writer", self.table, self.cloudwatch, self.sagemaker, transcribe=self.transcribe)
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
    self.assertTrue(hours[-3]["complete"])
    self.assertNotIn("complete", hours[-2])
    self.assertNotIn("complete", hours[-1])
    self.assertIn("incomplete", hours[-1]["issue"])
    self.assertNotIn("observedAt", hours[-2])
    self.assertEqual(hours[-1]["observedAt"], self.reader.iso(NOW))

  def test_legacy_probes_use_five_minute_coverage_and_new_hours_use_the_recorded_probe(self):
    day = date(2026, 11, 12)
    start, _ = self.reader.day_bounds(day)
    current = start + timedelta(hours=1, minutes=10)
    for slot in range(12):
      at = start + timedelta(minutes=5 * slot)
      self.writer.sample("cognito", 300, at, at, "o", self.context)
    self.writer.sample("cognito", 300, current, current, "o", self.context, hourly_probe=True)
    row = self.table.rows[("cognito", day.isoformat())]
    self.assertEqual(row["hourlyProbes"], "-h")
    self.assertEqual(row["samples"], "o" * 12 + "--o")
    self.assertEqual(self.reader.tile(day, row, 300, current + timedelta(minutes=15), "cognito")["state"], "operational")
    gap = start + timedelta(hours=2)
    now = gap + timedelta(hours=1, minutes=20)
    before_gap = self.reader.hour_tile(gap - timedelta(hours=1), day, row, 300, now, "cognito")
    self.assertEqual(before_gap["state"], "operational")
    self.assertTrue(before_gap["complete"])

    # A legacy point in the gap does not fill either the twelve legacy slots or an hourly observation.
    point = gap + timedelta(minutes=10)
    self.writer.sample("cognito", 300, point, point, "o", self.context)
    next_check = gap + timedelta(hours=1, minutes=10)
    self.writer.sample("cognito", 300, next_check, next_check, "u", self.context, hourly_probe=True)
    row = self.table.rows[("cognito", day.isoformat())]
    self.assertEqual(row["hourlyProbes"], "-h-h")
    missing = self.reader.hour_tile(gap, day, row, 300, now, "cognito")
    self.assertEqual(missing["state"], "unknown")
    self.assertNotIn("complete", missing)
    unknown = self.reader.hour_tile(gap + timedelta(hours=1), day, row, 300, now, "cognito")
    self.assertEqual(unknown["state"], "unknown")
    self.assertTrue(unknown["complete"])
    self.assertEqual(self.reader.tile(day, row, 300, now, "cognito")["state"], "unknown")

  def test_hourly_probes_use_distinct_mountain_fall_back_hours_and_report_recovery(self):
    day = date(2026, 11, 1)
    first = datetime(2026, 11, 1, 7, 10, tzinfo=timezone.utc)
    second = first + timedelta(hours=1)
    now = second + timedelta(minutes=2)
    for at, code in ((first, "f"), (second, "o")):
      self.writer.sample("short-links", 300, at, at, code, self.context, hourly_probe=True)
    row = self.table.rows[("short-links", day.isoformat())]
    failure, restored = (self.reader.hour_tile(at.replace(minute=0), day, row, 300, now, "short-links")
                         for at in (first, second))
    self.assertEqual((failure["state"], restored["state"]), ("outage", "operational"))
    self.assertTrue(failure["complete"])
    self.assertTrue(restored["complete"])
    self.assertEqual(failure["from"], self.reader.iso(first))
    self.assertEqual(failure["to"], self.reader.iso(second.replace(minute=0)))
    self.assertIn("Recovery was observed after this hour.", failure["issue"])
    self.assertEqual(row["hourlyProbes"], "-hh")

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
    self.assertEqual(snapshot["checks"][0]["expiresAt"], self.reader.iso(NOW + timedelta(minutes=75)))
    with patch.object(StubClock, "now", return_value=NOW + timedelta(hours=1)):
      self.assertEqual(self.read()["checks"][0]["id"], "project:website")
    with patch.object(StubClock, "now", return_value=NOW + timedelta(minutes=75)):
      self.assertEqual(self.read()["checks"], [])
    self.assertNotIn("ignore this", json.dumps(snapshot))
    self.assertNotIn(("short-links", "2026-11-12"), self.table.rows)

  def test_old_probe_schedule_delivery_is_dropped_before_making_requests(self):
    scheduled = NOW + timedelta(minutes=5)
    with patch.object(StubClock, "now", return_value=scheduled), patch.object(http.client, "HTTPSConnection") as connect:
      event = {"operation": "probes", "scheduledAt": self.writer.iso(scheduled)}
      self.assertEqual(self.writer.handler(event, self.context), {"ok": True, "collected": 0})
    connect.assert_not_called()
    self.assertEqual(self.table.write_count, 0)

  def test_optional_short_links_head_accepts_only_exact_location(self):
    response = Mock(status=302)
    response.getheader.return_value = self.writer.SHORT_TARGET
    self.assertTrue(self.writer.redirect_response(response))
    for bad in ("https://www.danielshort.me/tools/aws-status?tracking=1", "https://another.example/"):
      response.getheader.return_value = bad
      self.assertFalse(self.writer.redirect_response(response))

  def test_hourly_fixed_requests_missing_metrics_empty_transcription_and_idle_endpoints(self):
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
    self.assertEqual(self.cloudwatch.get_metric_statistics.call_count, 2)
    self.transcribe.list_transcription_jobs.assert_called_once_with(JobNameContains="site-transcribe-", MaxResults=100)
    for call in self.cloudwatch.get_metric_statistics.call_args_list:
      self.assertEqual(call.kwargs["Period"], 3600)
      self.assertEqual(call.kwargs["EndTime"], NOW.replace(minute=0))
      dimensions = {row["Name"]: row["Value"] for row in call.kwargs["Dimensions"]}
      self.assertEqual(call.kwargs["Namespace"], "AWS/SageMaker")
      self.assertEqual(set(dimensions), {"EndpointName", "VariantName"})
      self.assertIn(dimensions["EndpointName"], self.writer.ENDPOINTS.values())
      self.assertEqual(dimensions["VariantName"], "AllTraffic")
    self.assertEqual({call.kwargs["EndpointName"] for call in self.sagemaker.describe_endpoint.call_args_list},
                     set(self.writer.ENDPOINTS.values()))
    self.assertEqual(set(self.table.rows[("chatbot", "2026-11-12")]["samples"]) - {"-"}, {"i"})
    self.assertEqual(set(self.table.rows[("transcribe", "2026-11-12")]["samples"]) - {"-"}, {"u"})
    transcription = next(item for item in self.read()["connections"] if item["id"] == "transcribe")
    hour = transcription["hours"][-2]
    self.assertTrue(hour["complete"])
    self.assertEqual(hour["state"], "unknown")
    self.assertEqual(hour["transcription"], {"completed": 0, "failed": 0, "inProgress": 0})
    self.assertNotIn("transcription", transcription["days"][-1])
    self.assertNotIn(("home-gpu", "2026-11-12"), self.table.rows)

  def test_transcription_counts_terminal_times_and_hour_overlap_across_pages(self):
    end = NOW.replace(minute=0)
    start = end - timedelta(hours=1)
    self.transcribe.list_transcription_jobs.side_effect = [
      {"NextToken": "next", "TranscriptionJobSummaries": [
        transcription_job("a", "QUEUED", end),
        transcription_job("b", "QUEUED", start),
        transcription_job("c", "COMPLETED", start, end),
        {"TranscriptionJobName": "private-site-transcribe-" + "f" * 64},
      ]},
      {"TranscriptionJobSummaries": [
        transcription_job("d", "FAILED", start - timedelta(days=1), end - timedelta(seconds=1)),
        transcription_job("e", "COMPLETED", start - timedelta(hours=3), start),
        transcription_job("f", "COMPLETED", start - timedelta(hours=4), start - timedelta(seconds=1)),
      ]},
    ]
    self.assertEqual(self.writer.handler({"operation": "hourly"}, self.context), {"ok": True, "collected": 5})
    calls = self.transcribe.list_transcription_jobs.call_args_list
    self.assertEqual(len(calls), 2)
    self.assertEqual(calls[1].kwargs, {"JobNameContains": "site-transcribe-", "MaxResults": 100, "NextToken": "next"})
    row = self.table.rows[("transcribe", "2026-11-12")]
    slot, counts = next(iter(row["transcription"].items()))
    expected = {"completed": 1, "failed": 1, "inProgress": 2}
    self.assertEqual(counts, expected)
    self.assertNotIn("site-transcribe-", repr(row))
    self.assertNotIn("fixture reason", repr(row))
    row["transcription"][slot] = {key: Decimal(value) for key, value in counts.items()}

    transcription = next(item for item in self.read()["connections"] if item["id"] == "transcribe")
    hour, day = transcription["hours"][-2], transcription["days"][-1]
    self.assertNotIn("transcription", transcription["hours"][-1])
    for period in (hour, day):
      self.assertEqual(period["state"], "degraded")
      self.assertEqual(period["transcription"], expected)
      self.assertNotIn("Recovery", period["issue"])
    self.assertEqual(hour["from"], self.reader.iso(start))
    self.assertEqual(hour["to"], self.reader.iso(end))
    self.assertNotIn("site-transcribe-", json.dumps(transcription))

    self.transcribe.list_transcription_jobs.side_effect = None
    self.transcribe.list_transcription_jobs.return_value = {"TranscriptionJobSummaries": [
      transcription_job("c", "COMPLETED", start, end),
    ]}
    self.assertEqual(self.writer.read_transcription(end, end + timedelta(hours=1), self.context),
                     {"completed": 1, "failed": 0, "inProgress": 0})

  def test_transcription_unknown_during_incomplete_scan_can_be_replaced_by_verified_result(self):
    end = NOW.replace(minute=0)
    start = end - timedelta(hours=1)
    completed = transcription_job("a", "COMPLETED", start, NOW - timedelta(minutes=30))
    self.transcribe.list_transcription_jobs.side_effect = [
      {"NextToken": "next", "TranscriptionJobSummaries": [completed]}, AwsError("AccessDeniedException"),
    ]
    with self.assertLogs("Writer", level="WARNING") as logs:
      self.assertTrue(self.writer.handler({"operation": "hourly"}, self.context)["ok"])
    self.assertEqual(len(logs.output), 1)
    self.assertNotIn("site-transcribe-", logs.output[0])
    transcription = next(item for item in self.read()["connections"] if item["id"] == "transcribe")
    hour = transcription["hours"][-2]
    self.assertTrue(hour["complete"])
    self.assertEqual(hour["state"], "unknown")
    self.assertNotIn("transcription", hour)
    self.assertNotIn("transcription", self.table.rows[("transcribe", "2026-11-12")])

    self.transcribe.list_transcription_jobs.side_effect = None
    self.transcribe.list_transcription_jobs.return_value = {"TranscriptionJobSummaries": [completed]}
    prior_writes = self.table.write_count
    self.assertTrue(self.writer.handler({"operation": "hourly"}, self.context)["ok"])
    self.assertEqual(self.table.write_count, prior_writes + 1)
    self.transcribe.list_transcription_jobs.return_value = {"TranscriptionJobSummaries": []}
    self.assertTrue(self.writer.handler({"operation": "hourly"}, self.context)["ok"])
    self.assertEqual(self.table.write_count, prior_writes + 1)
    transcription = next(item for item in self.read()["connections"] if item["id"] == "transcribe")
    self.assertTrue(transcription["hours"][-2]["complete"])
    self.assertEqual(transcription["hours"][-2]["state"], "operational")
    self.assertEqual(transcription["hours"][-2]["transcription"], {"completed": 1, "failed": 0, "inProgress": 0})

  def test_unreliable_transcription_pages_are_unknown(self):
    end = NOW.replace(minute=0)
    start = end - timedelta(hours=1)
    malformed = transcription_job("a", "FAILED", start)
    duplicate = transcription_job("b", "QUEUED", start)
    empty = {"TranscriptionJobSummaries": []}
    cases = [
      [{"TranscriptionJobSummaries": [malformed]}],
      [{"NextToken": "same", **empty}, {"NextToken": "same", **empty}],
      [{"NextToken": str(index), **empty} for index in range(self.writer.TRANSCRIBE_MAX_PAGES)],
      [{"NextToken": "next", "TranscriptionJobSummaries": [duplicate]}, {"TranscriptionJobSummaries": [duplicate]}],
      [{}],
    ]
    for pages in cases:
      with self.subTest(pages=len(pages)), self.assertLogs("Writer", level="WARNING"):
        self.transcribe.list_transcription_jobs.side_effect = pages
        self.assertIsNone(self.writer.read_transcription(start, end, self.context))
    self.transcribe.list_transcription_jobs.reset_mock()
    self.context.get_remaining_time_in_millis.return_value = 9999
    with self.assertLogs("Writer", level="WARNING"):
      self.assertIsNone(self.writer.read_transcription(start, end, self.context))
    self.transcribe.list_transcription_jobs.assert_not_called()

  def test_daily_idle_requires_all_expected_hours_and_legacy_samples_remain_readable(self):
    day = date(2026, 11, 1)
    start, end = self.reader.day_bounds(day)
    now = end + timedelta(minutes=10)
    zero = {"completed": 0, "failed": 0, "inProgress": 0}
    mid_day = start + timedelta(hours=2, minutes=10)
    partial = {"samples": "uu", "checked": self.reader.iso(mid_day), "transcription": {"0": zero}}
    self.assertNotIn("transcription", self.reader.tile(day, partial, 3600, mid_day, "transcribe"))
    for slot in range(24):
      hour = start + timedelta(hours=slot)
      self.writer.sample("transcribe", 3600, hour, hour, "u", self.context, transcription=zero)
    row = self.table.rows[("transcribe", day.isoformat())]
    self.assertNotIn("transcription", self.reader.tile(day, row, 3600, now, "transcribe"))
    last = end - timedelta(hours=1)
    self.writer.sample("transcribe", 3600, last, last, "u", self.context, transcription=zero)
    row = self.table.rows[("transcribe", day.isoformat())]
    self.assertEqual(len(row["samples"]), 25)
    self.assertEqual(self.reader.tile(day, row, 3600, now, "transcribe")["transcription"], zero)
    self.assertNotIn("transcription", self.reader.hour_tile(last, day, row, 3600, now, "tracker"))

    legacy = {"samples": "bn", "checked": self.reader.iso(start + timedelta(hours=1))}
    accepted = self.reader.hour_tile(start, day, legacy, 3600, now, "transcribe")
    idle = self.reader.hour_tile(start + timedelta(hours=1), day, legacy, 3600, now, "transcribe")
    self.assertEqual(accepted["state"], "operational")
    self.assertIn("accepted batch requests", accepted["issue"])
    self.assertNotIn("transcription", accepted)
    self.assertEqual(idle["state"], "unknown")
    self.assertNotIn("transcription", idle)

  def test_home_gpu_distinguishes_missing_stale_fresh_future_and_invalid_check_ins(self):
    hour = int(NOW.timestamp()) // 3600
    begin = hour * 3600000
    now = int(NOW.timestamp() * 1000)
    key = ("home-gpu-heartbeat", str(hour))
    self.table.rows[(key[0], str(hour - 1))] = {"state": "ready", "received": begin - 1}
    self.assertEqual(self.writer.read_home_gpu(NOW, self.context), "p")
    for received, state, outcome in (
      (Decimal(begin), "ready", "g"),
      (now, "unavailable", "w"),
      (now, "unknown", "q"),
      (begin - 1, "ready", "p"),
      (now + 1, "ready", "q"),
      (begin + 3600000, "ready", "q"),
      (now, "invalid", "q"),
      (now, None, "q"),
      (str(now), "ready", "q"),
      (True, "ready", "q"),
      (Decimal(now) + Decimal("0.5"), "ready", "q"),
      (Decimal("NaN"), "ready", "q"),
    ):
      with self.subTest(received=received, state=state):
        self.table.rows[key] = {"received": received, "state": state}
        self.assertEqual(self.writer.read_home_gpu(NOW, self.context), outcome)
    self.table.rows[key] = {"state": "ready"}
    self.assertEqual(self.writer.read_home_gpu(NOW, self.context), "q")
    with patch.object(self.table, "get_item", side_effect=AwsError("InternalServerError")) as read:
      self.assertEqual(self.writer.read_home_gpu(NOW, self.context), "q")
      read.assert_called_once_with(Key={"pk": key[0], "sk": key[1]}, ConsistentRead=True)

  def test_hourly_poll_stores_only_one_home_gpu_sample_and_reader_keeps_it_separate(self):
    self.writer.HOME_GPU_MONITORING = True
    self.cloudwatch.describe_alarms.return_value = {"MetricAlarms": []}
    self.cloudwatch.get_metric_statistics.return_value = {"Datapoints": []}
    self.sagemaker.describe_endpoint.side_effect = lambda EndpointName: {
      "EndpointName": EndpointName, "EndpointStatus": "InService"
    }
    hour = int(NOW.timestamp()) // 3600
    heartbeat_key = ("home-gpu-heartbeat", str(hour))
    self.table.rows[heartbeat_key] = {
      "state": "ready", "received": Decimal(hour * 3600000 + 5000), "unused": "private fixture"
    }
    with patch.object(self.table, "get_item", wraps=self.table.get_item) as read:
      result = self.writer.handler({"operation": "hourly"}, self.context)
      self.assertEqual(result, {"ok": True, "collected": 6})
      reads = [call for call in read.call_args_list if call.kwargs["Key"]["pk"] == heartbeat_key[0]]
      self.assertEqual(len(reads), 1)
      self.assertEqual(reads[0].kwargs, {"Key": {"pk": heartbeat_key[0], "sk": heartbeat_key[1]},
                                       "ConsistentRead": True})
    first_write_count = self.table.write_count
    self.table.rows[heartbeat_key]["state"] = "unavailable"
    self.assertEqual(self.writer.handler({"operation": "hourly"}, self.context), {"ok": True, "collected": 6})
    self.assertEqual(self.table.write_count, first_write_count)
    self.assertNotIn("homeGpu", self.read())
    reader = load_runtime("Collector", self.table, self.cloudwatch, self.sagemaker, home_gpu=True)
    snapshot = reader.handler({}, self.context)
    self.assertNotIn("private fixture", json.dumps(snapshot))
    rows = {item["id"]: item["hours"][-1] for item in snapshot["connections"]}
    self.assertEqual(len(rows), 7)
    self.assertNotIn("home-gpu", rows)
    self.assertEqual(snapshot["homeGpu"]["id"], "home-gpu")
    latest = snapshot["homeGpu"]["hours"][-1]
    self.assertEqual(latest["state"], "operational")
    self.assertTrue(latest["complete"])
    self.assertIn("worker was ready", latest["issue"])
    self.assertEqual(rows["transcribe"]["state"], "unknown")

  def test_home_gpu_read_error_does_not_interrupt_existing_hourly_samples(self):
    self.writer.HOME_GPU_MONITORING = True
    self.cloudwatch.describe_alarms.return_value = {"MetricAlarms": []}
    self.cloudwatch.get_metric_statistics.return_value = {"Datapoints": []}
    self.sagemaker.describe_endpoint.side_effect = lambda EndpointName: {
      "EndpointName": EndpointName, "EndpointStatus": "InService"
    }
    read = self.table.get_item

    def failing_heartbeat(Key, ConsistentRead):
      if Key["pk"] == "home-gpu-heartbeat":
        raise AwsError("InternalServerError")
      return read(Key=Key, ConsistentRead=ConsistentRead)

    with patch.object(self.table, "get_item", side_effect=failing_heartbeat):
      self.assertEqual(self.writer.handler({"operation": "hourly"}, self.context), {"ok": True, "collected": 6})
    samples = {pk: set(row["samples"]) - {"-"} for (pk, _), row in self.table.rows.items()}
    self.assertEqual(samples["home-gpu"], {"q"})
    self.assertEqual(samples["transcribe"], {"u"})
    self.assertEqual(samples["chatbot"], {"s"})

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

  def test_named_hourly_alarm_details_are_fixed_and_legacy_open_transitions_do_not_extend(self):
    name = "website-job-tracker-api-5xx"
    self.cloudwatch.describe_alarms.return_value = {
      "MetricAlarms": [{"AlarmName": alarm, "StateValue": "ALARM" if alarm == name else "OK",
                       "StateReason": "private alarm reason"} for alarm in self.writer.ALARMS],
    }
    self.assertTrue(self.writer.handler({"operation": "hourly"}, self.context)["ok"])
    tracker = next(item for item in self.read()["connections"] if item["id"] == "tracker")
    latest = tracker["hours"][-1]
    self.assertIn("Tracker API 5xx alarm", latest["issue"])
    row = self.table.rows[("tracker", self.writer.mountain_day(NOW).isoformat())]
    self.assertEqual(list(row["alarmChecks"].values()), [["ta"]])
    self.assertNotIn("private alarm reason", repr(row))

    old = NOW.replace(minute=0) - timedelta(minutes=5)
    delayed = NOW + timedelta(hours=1, minutes=5)
    row["events"] = [[self.reader.iso(old), "ta", "ALARM"], [self.reader.iso(delayed), "ta", "ALARM"]]
    duration = "job-application-tracker-lambda-duration-high"
    self.cloudwatch.describe_alarms.return_value["MetricAlarms"] = [
      {"AlarmName": alarm, "StateValue": "ALARM" if alarm == duration else "OK"} for alarm in self.writer.ALARMS
    ]
    with patch.object(StubClock, "now", return_value=NOW + timedelta(hours=1)):
      self.assertTrue(self.writer.handler({"operation": "hourly"}, self.context)["ok"])
      tracker = next(item for item in self.read()["connections"] if item["id"] == "tracker")
    self.assertIn("Tracker Lambda duration alarm", tracker["days"][-1]["issue"])

    self.cloudwatch.describe_alarms.return_value["MetricAlarms"] = [
      {"AlarmName": alarm, "StateValue": "OK"} for alarm in self.writer.ALARMS
    ]
    with patch.object(StubClock, "now", return_value=NOW + timedelta(hours=2)):
      self.assertTrue(self.writer.handler({"operation": "hourly"}, self.context)["ok"])
      tracker = next(item for item in self.read()["connections"] if item["id"] == "tracker")
    self.assertEqual(tracker["hours"][-1]["state"], "operational")
    self.assertEqual(tracker["hours"][-2]["state"], "degraded")
    self.assertIn("Tracker Lambda duration alarm", tracker["hours"][-2]["issue"])
    self.assertIn("Tracker API 5xx alarm", tracker["hours"][-3]["issue"])

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
    services = (*self.reader.SERVICES, ("home-gpu", "Home GPU", 3600))
    for identifier, _label, cadence in services:
      for offset in range(90):
        day = today - timedelta(days=offset)
        start, end = self.reader.day_bounds(day)
        checked = min(NOW, end - timedelta(seconds=1))
        count = int((checked - start).total_seconds()) // cadence + 1
        start_text = self.reader.iso(start + timedelta(seconds=10))
        self.table.rows[(identifier, day.isoformat())] = {
          "pk": identifier, "sk": day.isoformat(), "samples": ("p" if identifier == "home-gpu" else "f") * count,
          "checked": self.reader.iso(checked),
          **({} if identifier == "home-gpu" else {"events": [[start_text, "ta", "ALARM"]], "truncated": True}),
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
    reader = load_runtime("Collector", self.table, self.cloudwatch, self.sagemaker, home_gpu=True)
    snapshot = reader.handler({}, self.context)
    self.assertTrue(snapshot["ok"])
    self.assertEqual(len(snapshot["homeGpu"]["hours"]), 72)
    self.assertEqual(len(snapshot["checks"]), 210)
    self.assertEqual(len(snapshot["catalogHistory"]["results"]), 70)
    self.assertEqual(snapshot["catalogHistory"]["results"][catalog_ids[0]], {"hours": "o" * 72, "days": "o" * 90})
    self.assertLess(len(json.dumps(snapshot).encode("utf-8")), 448 * 1024)

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

  def test_browser_reason_history_is_sparse_safe_and_chooses_the_latest_worst_state(self):
    self.context.invoked_function_arn += ":browser-ci"
    route, wait, unknown, legacy = "tool:text-compare", "tool:transcribe", "tool:watermark-remover", "project:website"
    earlier_at, later_at = NOW - timedelta(minutes=20), NOW - timedelta(minutes=8)
    earlier = [
      {**sample_report(route, "degraded", "browser", checked=earlier_at), "reasonCode": "r"},
      {**sample_report(wait, "degraded", "browser", checked=earlier_at), "reasonCode": "t"},
      {**sample_report(unknown, "unknown", "browser", checked=earlier_at), "reasonCode": "n"},
      sample_report(legacy, "operational", "browser", checked=earlier_at),
    ]
    later = [
      {**sample_report(wait, "degraded", "browser", checked=later_at), "reasonCode": "m"},
      sample_report(route, "operational", "browser", checked=later_at),
    ]
    for checks in (earlier, later):
      self.assertEqual(self.report.handler({"checks": checks}, self.context), {"ok": True, "accepted": len(checks)})
    tiles = self.read()["catalogHistory"]["results"]
    self.assertEqual(tiles[wait]["hours"][-2:], "dd")
    self.assertEqual(tiles[wait]["details"]["hours"][-2:], "tm")
    self.assertEqual(tiles[wait]["details"]["days"][-1], "m")
    self.assertEqual(tiles[route]["hours"][-2:], "do")
    self.assertEqual(tiles[route]["details"]["days"][-1], "r")
    self.assertEqual(tiles[unknown]["details"]["days"][-1], "n")
    self.assertNotIn("details", tiles[legacy])
    self.assertTrue(all(len(value) == 3 for value in self.table.rows[("catalog", "browser")]["results"].values()))

    # A newer report can omit details and must not inherit reasons from the superseded hour.
    bare = sample_report(wait, "degraded", "browser")
    self.assertEqual(self.report.handler({"checks": [bare]}, self.context), {"ok": True, "accepted": 1})
    details = self.read()["catalogHistory"]["results"][wait]["details"]
    self.assertEqual(details["hours"][-2:], "t-")
    self.assertEqual(details["days"][-1], "-")
    row = next(row for (pk, sk), row in self.table.rows.items() if pk == "catalog" and sk.startswith("report#"))
    self.assertEqual(set(row["reasons"]), {route, wait, unknown})
    row["reasons"][wait] = row["reasons"][wait].replace("t", "n")  # Invalid reason for a degraded observation.
    degraded = self.read()["catalogHistory"]["results"][wait]
    self.assertEqual(degraded["hours"][-2:], "dd")
    self.assertNotIn("details", degraded)

  def test_only_browser_alias_accepts_allowlisted_reason_state_pairs(self):
    base = sample_report("tool:text-compare", "degraded", "browser")
    with self.assertLogs("Report", level="WARNING"):
      no_alias = self.report.handler({"checks": [{**base, "reasonCode": "e"}]}, self.context)
    self.assertFalse(no_alias["ok"])
    self.context.invoked_function_arn += ":browser-ci"
    for malformed in (None, "", False, ["e"], "private output", "n", "rr"):
      with self.subTest(reason=malformed), self.assertLogs("Report", level="WARNING"):
        result = self.report.handler({"checks": [{**base, "reasonCode": malformed}]}, self.context)
        self.assertFalse(result["ok"])
    self.assertEqual(self.table.write_count, 0)
    result = self.report.handler({"checks": [{**base, "reasonCode": "e"}]}, self.context)
    self.assertEqual(result, {"ok": True, "accepted": 1})

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
    probes = TEMPLATE.split("  PeriodicProbes:\n")[1].split("  HourlyMetadata:")[0]
    transitions = TEMPLATE.split("  SelectedAlarmTransitions:\n")[1].split("  AllowPeriodicProbes:")[0]
    self.assertIn("ScheduleExpression: 'cron(10 * * * ? *)'", probes)
    self.assertIn("State: DISABLED", transitions)
    self.assertNotIn("sagemaker:InvokeEndpoint", TEMPLATE)
    self.assertNotIn("transcribe:StartTranscriptionJob", TEMPLATE)
    self.assertNotIn("ses:SendEmail", TEMPLATE)
    invoker = TEMPLATE.split("  WebsiteInvokerRole:\n")[1].split("\nOutputs:")[0]
    self.assertIn("project:website:environment:production", invoker)
    self.assertIn("Resource: !GetAtt Collector.Arn", invoker)
    self.assertNotIn("Writer.Arn", invoker)
    self.assertNotIn("Report.Arn", invoker)
    self.assertIn("Default: 'false'", TEMPLATE)
    reader_role = TEMPLATE.split("  CollectorRole:\n")[1].split("  Collector:\n")[0]
    writer_role = TEMPLATE.split("  WriterRole:\n")[1].split("  Writer:\n")[0]
    self.assertNotIn("home-gpu-heartbeat", reader_role)
    self.assertEqual(TEMPLATE.count("transcribe:ListTranscriptionJobs"), 1)
    self.assertIn(dedent('''\
              - Effect: Allow
                Action: transcribe:ListTranscriptionJobs
                Resource: '*'
                Condition:
                  StringEquals:
                    'aws:RequestedRegion': us-east-2
''').replace("\n", "\n              ").rstrip(), writer_role)
    self.assertIn("Action: dynamodb:GetItem\n", writer_role)
    self.assertEqual(writer_role.count("home-gpu-heartbeat"), 1)


if __name__ == "__main__":
  main()
