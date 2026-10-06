from sqlalchemy import text
from database.connection import engine

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        print("=" * 60)
        print("DATABASE CONNECTED SUCCESSFULLY")
        print(result.fetchone()[0])
        print("=" * 60)

except Exception as e:
    print(e)