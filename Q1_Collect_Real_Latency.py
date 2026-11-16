import csv
import datetime
import math
import os
import statistics
import time
from pathlib import Path
import requests

path = Path(__file__).with_name("Q1_Raw_Timings.csv")
fields = ["utc_timestamp", "local_date", "service", "endpoint", "status", "elapsed_ms", "success", "error"]
if not path.exists():
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(fields)

location_id = os.environ["OPENAQ_LOCATION_ID"]
api_key = os.environ["OPENAQ_API_KEY"]
urls = [
    ("air", f"https://api.openaq.org/v3/locations/{location_id}/latest", {}, {"X-API-Key": api_key}),
    ("weather", "https://api.open-meteo.com/v1/forecast", {"latitude": 24.8607, "longitude": 67.0011, "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"}, {}),
]

def record(service, endpoint, status, elapsed_ms, success, error):
    now = datetime.datetime.now(datetime.timezone.utc)
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow([now.isoformat(), now.astimezone().date().isoformat(), service, endpoint, status, f"{elapsed_ms:.6f}", success, error])

def heat_index_c(t_c, rh):
    f = t_c * 9 / 5 + 32
    h = -42.379 + 2.04901523*f + 10.14333127*rh - .22475541*f*rh - .00683783*f*f - .05481717*rh*rh + .00122874*f*f*rh + .00085282*f*rh*rh - .00000199*f*f*rh*rh
    return (h - 32) * 5 / 9

advisories = ["WHO: avoid strenuous activity during extreme heat", "WHO: monitor exposure to fine particulate pollution"]

def lookup():
    return [item for item in advisories if "heat" in item.lower()]

for service, endpoint, params, headers in urls:
    for _ in range(10):
        start = time.perf_counter()
        status, success, error = "", False, ""
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=25)
            status = response.status_code
            response.raise_for_status()
            data = response.json()
            success = "results" in data if service == "air" else "current" in data
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        record(service, endpoint, status, 1000*(time.perf_counter()-start), success, error)

for service, task in [("heat_index", lambda:heat_index_c(38,70)), ("knowledge_base", lookup)]:
    for _ in range(20):
        start = time.perf_counter()
        task()
        record(service, "local", "", 1000*(time.perf_counter()-start), True, "")

with path.open(newline="", encoding="utf-8") as stream:
    entries = list(csv.DictReader(stream))
valid = [item for item in entries if item["success"] == "True"]
dates = {item["local_date"] for item in valid if item["service"] in {"air", "weather"}}
counts = {service: sum(item["service"] == service for item in valid) for service in ["air", "weather"]}
if len(dates) >= 2 and all(value >= 20 for value in counts.values()):
    durations = [float(item["elapsed_ms"]) for item in valid]
    median_tool_delay_seconds = statistics.median(durations) / 1000
    path.with_name("Q1_Median_Tool_Delay.txt").write_text(str(median_tool_delay_seconds), encoding="utf-8")
