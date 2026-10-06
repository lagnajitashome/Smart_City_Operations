import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Database
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "smart_city_db"
DB_USER = "postgres"
DB_PASSWORD = "Smartcity@123"