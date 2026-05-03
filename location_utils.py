"""
GPS extraction (EXIF / video metadata) and Nominatim reverse geocoding.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

import requests
from PIL import Image
from PIL.ExifTags import GPSTAGS

logger = logging.getLogger(__name__)

NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

# Per https://operations.osmfoundation.org/policies/nominatim/ — identify the application.
NOMINATIM_HEADERS = {
    "User-Agent": "RoadSenseAI/1.0 (local road anomaly reporting; +https://example.local/roadsense)",
    "Accept-Language": "en",
}


def _first_nonempty(addr: dict[str, Any], keys: tuple[str, ...]) -> str:
    for k in keys:
        v = addr.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def map_nominatim_address(addr: dict[str, Any]) -> tuple[str, str]:
    """Map Nominatim `address` object to city, state."""
    city = _first_nonempty(
        addr,
        ("city", "town", "village", "municipality", "city_district", "county"),
    )
    state = _first_nonempty(addr, ("state", "region"))
    return city, state


def reverse_geocode(lat: float, lon: float, timeout: float = 15.0) -> dict[str, Any]:
    """
    Call Nominatim reverse API. Returns dict with:
      ok, city, state, error (if any).
    """
    out: dict[str, Any] = {
        "ok": False,
        "city": "",
        "state": "",
        "error": None,
    }
    try:
        r = requests.get(
            NOMINATIM_REVERSE,
            params={"lat": lat, "lon": lon, "format": "json"},
            headers=NOMINATIM_HEADERS,
            timeout=timeout,
        )
        if r.status_code == 429:
            out["error"] = "Nominatim rate limit; try again later."
            return out
        if not r.ok:
            out["error"] = f"Nominatim HTTP {r.status_code}"
            return out
        data = r.json()
        addr = data.get("address") or {}
        city, st = map_nominatim_address(addr)
        out["ok"] = True
        out["city"] = city
        out["state"] = st
        return out
    except requests.RequestException as e:
        logger.warning("Nominatim request failed: %s", e)
        out["error"] = str(e)
        return out


def forward_geocode_search(
    query: str,
    limit: int = 5,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """
    Nominatim forward search for a place name. Returns bounding boxes and optional
    GeoJSON polygons (same service as reverse_geocode).

    Each result: display_name, lat, lon, bbox (south, north, west, east), geojson (or None).
    """
    out: dict[str, Any] = {"ok": False, "results": [], "error": None}
    q = (query or "").strip()
    if len(q) < 2:
        out["error"] = "Query too short"
        return out
    limit = max(1, min(int(limit), 10))
    try:
        r = requests.get(
            NOMINATIM_SEARCH,
            params={
                "q": q,
                "format": "json",
                "limit": limit,
                "polygon_geojson": 1,
                "addressdetails": 0,
            },
            headers=NOMINATIM_HEADERS,
            timeout=timeout,
        )
        if r.status_code == 429:
            out["error"] = "Nominatim rate limit; try again later."
            return out
        if not r.ok:
            out["error"] = f"Nominatim HTTP {r.status_code}"
            return out
        data = r.json()
        if not isinstance(data, list):
            out["error"] = "Unexpected Nominatim response"
            return out
        results: list[dict[str, Any]] = []
        for row in data:
            bb = row.get("boundingbox")
            bbox_parsed: dict[str, float] | None = None
            if isinstance(bb, list) and len(bb) >= 4:
                try:
                    south, north, west, east = (
                        float(bb[0]),
                        float(bb[1]),
                        float(bb[2]),
                        float(bb[3]),
                    )
                    bbox_parsed = {
                        "south": south,
                        "north": north,
                        "west": west,
                        "east": east,
                    }
                except (TypeError, ValueError):
                    bbox_parsed = None
            lat_s, lon_s = row.get("lat"), row.get("lon")
            try:
                lat_f = float(lat_s) if lat_s is not None else None
                lon_f = float(lon_s) if lon_s is not None else None
            except (TypeError, ValueError):
                lat_f = lon_f = None
            gj = row.get("geojson")
            if not isinstance(gj, dict):
                gj = None
            results.append(
                {
                    "display_name": str(row.get("display_name") or "").strip(),
                    "lat": lat_f,
                    "lon": lon_f,
                    "bbox": bbox_parsed,
                    "geojson": gj,
                    "type": row.get("type") or row.get("class") or "",
                }
            )
        out["ok"] = True
        out["results"] = results
        return out
    except requests.RequestException as e:
        logger.warning("Nominatim search failed: %s", e)
        out["error"] = str(e)
        return out


def _convert_rational_to_float(rational) -> float:
    try:
        if hasattr(rational, "numerator"):
            return float(rational.numerator) / max(float(rational.denominator), 1e-12)
    except (TypeError, ValueError, AttributeError):
        pass
    if isinstance(rational, tuple) and len(rational) == 2:
        return float(rational[0]) / max(float(rational[1]), 1e-12)
    return float(rational)


def _dms_tuple_to_degrees(tup) -> float:
    if not tup or len(tup) < 3:
        return 0.0
    d = _convert_rational_to_float(tup[0])
    m = _convert_rational_to_float(tup[1])
    s = _convert_rational_to_float(tup[2])
    return d + m / 60.0 + s / 3600.0


def extract_gps_from_image_path(path: str) -> tuple[float | None, float | None, str | None]:
    """Read GPS from image EXIF. Returns (lat, lon, error)."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None, None, "No EXIF metadata in image"
            gps_ifd = exif.get_ifd(0x8825)
            if not gps_ifd:
                return None, None, "No GPS EXIF in image"
            gps_data: dict[str, Any] = {}
            for k, v in gps_ifd.items():
                tag = GPSTAGS.get(k, k)
                gps_data[tag] = v
            lat = gps_data.get("GPSLatitude")
            lon = gps_data.get("GPSLongitude")
            lat_ref = gps_data.get("GPSLatitudeRef")
            lon_ref = gps_data.get("GPSLongitudeRef")
            if lat is None or lon is None:
                return None, None, "Incomplete GPS EXIF in image"
            lat_deg = _dms_tuple_to_degrees(lat)
            lon_deg = _dms_tuple_to_degrees(lon)
            if str(lat_ref or "N").upper().startswith("S"):
                lat_deg = -abs(lat_deg)
            else:
                lat_deg = abs(lat_deg)
            if str(lon_ref or "E").upper().startswith("W"):
                lon_deg = -abs(lon_deg)
            else:
                lon_deg = abs(lon_deg)
            return lat_deg, lon_deg, None
    except Exception as e:
        logger.exception("EXIF GPS read failed")
        return None, None, str(e)


_ISO6709_RE = re.compile(
    r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)"
)


def _parse_iso6709(s: str) -> tuple[float | None, float | None]:
    s = s.strip()
    m = _ISO6709_RE.match(s)
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None, None


def _parse_location_tag(val: str) -> tuple[float | None, float | None]:
    if not val or not str(val).strip():
        return None, None
    val = str(val).strip()
    lat, lon = _parse_iso6709(val.replace(" ", ""))
    if lat is not None and lon is not None:
        return lat, lon
    parts = re.split(r"[\s,/]+", val)
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return None, None


def extract_gps_from_video_path(path: str) -> tuple[float | None, float | None, str | None]:
    """Read GPS from video container metadata via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None, None, "ffprobe not found (install ffmpeg for video GPS)"

    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            return None, None, "ffprobe failed to read video metadata"
        meta = json.loads(proc.stdout or "{}")
        tags: dict[str, str] = {}
        fmt = meta.get("format") or {}
        for k, v in (fmt.get("tags") or {}).items():
            if v is not None:
                tags[str(k).lower()] = str(v)
        for stream in meta.get("streams") or []:
            for k, v in (stream.get("tags") or {}).items():
                if v is not None:
                    tags[str(k).lower()] = str(v)

        def try_tags(tag_map: dict[str, str]) -> tuple[float | None, float | None]:
            candidates = [
                "com.apple.quicktime.location.iso6709",
                "location",
                "location-eng",
                "com.apple.quicktime.location",
            ]
            for key in candidates:
                lk = key.lower()
                for tk, tv in tag_map.items():
                    if lk in tk or tk == lk:
                        lat, lon = _parse_location_tag(tv)
                        if lat is not None and lon is not None:
                            return lat, lon
            for tv in tag_map.values():
                lat, lon = _parse_location_tag(tv)
                if lat is not None and lon is not None:
                    return lat, lon
            return None, None

        lat, lon = try_tags(tags)
        if lat is not None and lon is not None:
            return lat, lon, None

        return None, None, "No GPS metadata in video"
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Video GPS extraction failed: %s", e)
        return None, None, str(e)


def extract_gps_from_upload(
    file_storage,
    ext: str,
    allowed_images: set[str],
    allowed_videos: set[str],
) -> tuple[float | None, float | None, str, str | None]:
    """
    Save upload to a temp file and extract GPS (does not persist under uploads/).
    Returns: lat, lng, source_gps ('exif' | 'video_metadata'), error_message
    """
    ext = ext.lower()
    if ext in allowed_images:
        fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}")
        os.close(fd)
        try:
            file_storage.save(tmp_path)
            lat, lng, err = extract_gps_from_image_path(tmp_path)
            return lat, lng, "exif", err
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if ext in allowed_videos:
        fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}")
        os.close(fd)
        try:
            file_storage.save(tmp_path)
            lat, lng, err = extract_gps_from_video_path(tmp_path)
            return lat, lng, "video_metadata", err
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return None, None, "exif", "Unsupported file type for GPS extraction"


def enrich_location_strings(
    lat_str: str,
    lng_str: str,
) -> tuple[str, str, str | None]:
    """
    Given string lat/lng from form, return (city, state, reverse_error).
    Empty strings if coordinates invalid or geocode fails.
    """
    lat_str = (lat_str or "").strip()
    lng_str = (lng_str or "").strip()
    if not lat_str or not lng_str:
        return "", "", None
    try:
        lat_f = float(lat_str)
        lng_f = float(lng_str)
    except ValueError:
        return "", "", None
    geo = reverse_geocode(lat_f, lng_f)
    if not geo.get("ok"):
        return "", "", geo.get("error")
    return (
        geo.get("city") or "",
        geo.get("state") or "",
        geo.get("error"),
    )
