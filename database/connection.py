from sqlalchemy import create_engine
from urllib.parse import quote_plus

from sqlalchemy import create_engine

from config.settings import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)

password = quote_plus(DB_PASSWORD)

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{password}"
    f"@{DB_HOST}:{DB_PORT}"
    f"/{DB_NAME}"
)

engine = create_engine(
    DATABASE_URL,
    echo=True
)