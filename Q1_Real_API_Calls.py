import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API_KEY = "6f43f8d18bde7ad149a0f6e10a128ae0b7a352d7de0c9f4030c24c1a2ecd2e7e"
LATITUDE = 24.8607
LONGITUDE = 67.0011
REPETITIONS = 20
REQUEST_TIMEOUT = 15
REQUEST_PAUSE = 0.4
TIMEZONE = timezone(timedelta(hours=5))
OUTPUT = Path(__file__).with_name("Q1_Raw_Timings.txt")


def now_utc():
    return datetime.now(timezone.utc)


def request_json(base, parameters=None, headers=None):
    query = urllib.parse.urlencode(parameters or {})
    url = base + ("?" + query if query else "")
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "SaafHawa-Q1/1.0"})
    start = time.perf_counter()
    status = ""
    payload = {}
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "at": now_utc(),
        "status": status,
        "payload": payload,
        "error": error,
        "elapsed_ms": (time.perf_counter() - start) * 1000,
        "url": url,
    }


def field(name, value):
    return f"{name}={str(value).replace(chr(10), ' ').replace(chr(13), ' ').replace(' | ', ' ')}"


def line(**values):
    return " | ".join(field(key, value) for key, value in values.items())


def date_window(day):
    start = datetime.combine(day, datetime.min.time(), tzinfo=TIMEZONE)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def find_sensors():
    headers = {"X-API-Key": API_KEY, "User-Agent": "SaafHawa-Q1/1.0"}
    candidates = []
    searches = [
        ("https://api.openaq.org/v3/locations", {"coordinates": f"{LATITUDE},{LONGITUDE}", "radius": 25000, "limit": 100}),
        ("https://api.openaq.org/v3/locations", {"bbox": "66.6,24.5,67.6,25.4", "limit": 100}),
        ("https://api.openaq.org/v3/locations/6135302", {}),
    ]
    errors = []
    for url, parameters in searches:
        result = request_json(url, parameters, headers)
        if result["error"]:
            errors.append(result["error"])
            continue
        for location in result["payload"].get("results", []):
            sensors = {}
            for sensor in location.get("sensors", []):
                parameter = sensor.get("parameter") or {}
                name = str(parameter.get("name", "")).lower().replace(".", "").replace("_", "")
                if name in ("pm25", "pm10") and sensor.get("id") is not None:
                    sensors[name] = sensor["id"]
            if sensors:
                candidates.append((len(sensors), location.get("id"), location.get("name", ""), sensors))
        if any(item[0] == 2 for item in candidates):
            break
    if not candidates:
        return {}, "", "; ".join(errors) or "No nearby Karachi PM sensors were found"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][3], f"{candidates[0][2]} (OpenAQ location {candidates[0][1]})", ""


def get_air(day, index, sensors):
    if not sensors:
        response = request_json("https://api.openaq.org/v3/locations/6135302/latest", headers={"X-API-Key": API_KEY, "User-Agent": "SaafHawa-Q1/1.0"})
        if not response["error"]:
            response["error"] = "No suitable historical PM sensor was found; latest endpoint cannot substitute for dated measurements"
        return response, "", "", "", ""
    names = list(sensors)
    pollutant = names[index % len(names)]
    page = index // len(names) + 1
    date_from, date_to = date_window(day)
    url = f"https://api.openaq.org/v3/sensors/{sensors[pollutant]}/measurements"
    parameters = {"datetime_from": date_from, "datetime_to": date_to, "limit": 1, "page": page}
    result = request_json(url, parameters, {"X-API-Key": API_KEY, "User-Agent": "SaafHawa-Q1/1.0"})
    matches = result["payload"].get("results", [])
    if result["error"] or not matches:
        return result, pollutant, "", "", ""
    measurement = matches[0]
    parameter = measurement.get("parameter") or {}
    period = measurement.get("period") or {}
    stamp = ((period.get("datetimeFrom") or {}).get("local") or (period.get("datetimeFrom") or {}).get("utc") or "")
    value = measurement.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not stamp:
        return result, pollutant, "", "", ""
    measurement_day = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(TIMEZONE).date()
    if measurement_day != day:
        return result, pollutant, "", "", ""
    return result, pollutant, value, parameter.get("units", ""), stamp


def get_weather(day, index):
    url = "https://api.open-meteo.com/v1/forecast"
    parameters = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Karachi",
        "past_days": 1,
        "forecast_days": 1,
    }
    result = request_json(url, parameters, {"User-Agent": "SaafHawa-Q1/1.0"})
    hourly = result["payload"].get("hourly") or {}
    target = f"{day.isoformat()}T{index:02d}:00"
    stamps = hourly.get("time") or []
    if result["error"] or target not in stamps:
        return result, target, "", "", "", ""
    position = stamps.index(target)
    keys = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m")
    values = []
    for key in keys:
        series = hourly.get(key) or []
        value = series[position] if position < len(series) else None
        values.append(value if isinstance(value, (int, float)) else "")
    if any(value == "" for value in values):
        return result, target, "", "", "", ""
    kind = "hourly model data, historical" if day < now_utc().astimezone(TIMEZONE).date() else "hourly model forecast/estimate"
    return result, target, *values, kind


def heat_index_celsius(temperature, humidity):
    f = temperature * 9 / 5 + 32
    h = -42.379 + 2.04901523*f + 10.14333127*humidity - 0.22475541*f*humidity - 0.00683783*f*f - 0.05481717*humidity*humidity + 0.00122874*f*f*humidity + 0.00085282*f*humidity*humidity - 0.00000199*f*f*humidity*humidity
    return (h-32)*5/9


def knowledge_base_search():
    knowledge = ["WHO: reduce outdoor activity in extreme heat", "WHO: reduce exposure to high PM2.5", "Protect children and older adults during heatwaves"]
    return [entry for entry in knowledge if "heat" in entry.lower()]


def run():
    started = now_utc()
    today = started.astimezone(TIMEZONE).date()
    yesterday = today - timedelta(days=1)
    sensors, station, setup_error = find_sensors()
    lines = [line(type="run", request_started_utc=started.isoformat(), request_local_date=today, station=station, setup_error=setup_error)]
    counts = {}
    durations = []
    for day in (yesterday, today):
        for index in range(REPETITIONS):
            air, pollutant, value, unit, stamp = get_air(day, index, sensors)
            good = air["status"] == 200 and value != ""
            counts[(str(day), "OpenAQ")] = counts.get((str(day), "OpenAQ"), 0) + int(good)
            lines.append(line(type="request", source="OpenAQ", requested_data_date=day, actual_request_utc=air["at"].isoformat(), actual_request_local_date=air["at"].astimezone(TIMEZONE).date(), number=index+1, http_status=air["status"], success=good, latency_ms=f"{air['elapsed_ms']:.3f}", pollutant=pollutant, value=value, units=unit, measurement_timestamp=stamp, error=air["error"] or ("No matching reading for this period" if not good else "")))
            if good:
                durations.append(air["elapsed_ms"])
            time.sleep(REQUEST_PAUSE)
            weather, stamp, temp, humidity, wind, kind = get_weather(day, index)
            good = weather["status"] == 200 and temp != ""
            counts[(str(day), "Open-Meteo")] = counts.get((str(day), "Open-Meteo"), 0) + int(good)
            lines.append(line(type="request", source="Open-Meteo", requested_data_date=day, actual_request_utc=weather["at"].isoformat(), actual_request_local_date=weather["at"].astimezone(TIMEZONE).date(), number=index+1, http_status=weather["status"], success=good, latency_ms=f"{weather['elapsed_ms']:.3f}", temperature_c=temp, humidity_pct=humidity, wind_kmh=wind, measurement_timestamp=stamp, data_kind=kind, error=weather["error"] or ("No hourly value for this period" if not good else "")))
            if good:
                durations.append(weather["elapsed_ms"])
            time.sleep(REQUEST_PAUSE)
    for service, function in (("heat_index", lambda: heat_index_celsius(38, 70)), ("knowledge_base", knowledge_base_search)):
        for index in range(REPETITIONS):
            began = time.perf_counter()
            result = function()
            elapsed = (time.perf_counter()-began)*1000
            durations.append(elapsed)
            lines.append(line(type="local", source=service, actual_request_utc=now_utc().isoformat(), number=index+1, success=True, latency_ms=f"{elapsed:.6f}", result=str(result)[:160]))
    for day in (yesterday, today):
        for service in ("OpenAQ", "Open-Meteo"):
            lines.append(line(type="summary", requested_data_date=day, source=service, calls=REPETITIONS, calls_with_valid_data=counts.get((str(day), service), 0)))
    if durations:
        lines.append(line(type="summary", median_successful_tool_latency_seconds=f"{statistics.median(durations)/1000:.9f}"))
    lines.append(line(type="summary", actual_request_day=today, historical_data_does_not_count_as_yesterday_api_calls=True))
    with OUTPUT.open("a", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    run()
