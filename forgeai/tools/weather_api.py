"""Keyless public Open-Meteo weather HTTP API tool."""

from __future__ import annotations

import re
from typing import Any, Callable

import httpx

from forgeai.ingest.parsers import DocumentInventory
from forgeai.tools.base import ToolResult

_WEATHER_RE = re.compile(
    r"\b(weather|forecast|temperature|rain|wind|humidity|storm|sea state|marine conditions|meteorolog\w*)\b",
    re.IGNORECASE,
)


class WeatherApiTool:
    """Fetch current weather from the keyless Open-Meteo HTTP API."""

    name = "weather_api"
    description = "Fetch current temperature and wind speed for latitude/longitude via Open-Meteo."
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
        },
        "required": ["latitude", "longitude"],
    }

    def __init__(self, http_get: Callable[..., Any] | None = None) -> None:
        self._http_get = http_get or httpx.get

    def is_eligible(self, inventory: DocumentInventory, brief: str) -> tuple[bool, str]:
        if _WEATHER_RE.search(brief):
            return True, "Brief explicitly asks for weather or meteorological conditions."
        return False, "No weather, forecast, or meteorological requirement in the brief."

    def run(self, args: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            lat = float(args["latitude"])
            lon = float(args["longitude"])
        except Exception as exc:
            return ToolResult(ok=False, output=f"Invalid coordinates: {exc}")

        try:
            resp = self._http_get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current_weather": "true"},
                timeout=10.0,
            )
            resp.raise_for_status()
            payload = resp.json()
            current = payload.get("current_weather", {})
            return ToolResult(
                ok=True,
                output=(
                    f"Weather at ({lat}, {lon}): temp={current.get('temperature')}C, "
                    f"windspeed={current.get('windspeed')}km/h"
                ),
                data=current,
            )
        except Exception as exc:
            return ToolResult(ok=False, output=f"Weather API request failed: {exc}")

