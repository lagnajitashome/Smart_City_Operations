from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

TRAFFIC_RAW = RAW_DATA_DIR / "traffic"
WEATHER_RAW = RAW_DATA_DIR / "weather"
PARKING_RAW = RAW_DATA_DIR / "parking"
ROUTING_RAW = RAW_DATA_DIR / "routing"
INCIDENT_RAW = RAW_DATA_DIR / "incidents"

TRAFFIC_PROCESSED = PROCESSED_DATA_DIR / "traffic"
WEATHER_PROCESSED = PROCESSED_DATA_DIR / "weather"
PARKING_PROCESSED = PROCESSED_DATA_DIR / "parking"
ROUTING_PROCESSED = PROCESSED_DATA_DIR / "routing"
INCIDENT_PROCESSED = PROCESSED_DATA_DIR / "incidents"