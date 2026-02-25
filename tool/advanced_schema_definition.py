# TODO Pydantic
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool


class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="城市名或坐标")
    # Literal["摄氏度", "华氏度"]表示units字段值的范围
    units: Literal["摄氏度", "华氏度"] = Field(
        default="摄氏度",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "摄氏度", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "摄氏度" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result

# TODO JSON
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}

@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "摄氏度", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "摄氏度" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result