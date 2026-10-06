import json
import glob
from config.config import WEATHER_RAW

files = glob.glob(str(WEATHER_RAW / "*.json"))

print(f"Checking {len(files)} JSON files...\n")

bad_files = []

for file_path in files:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)

    except Exception as e:
        bad_files.append(file_path)

        print("=" * 70)
        print("BAD FILE:")
        print(file_path)
        print("ERROR:")
        print(e)
        print("=" * 70)

print()
print("CHECK COMPLETE")
print(f"Total files : {len(files)}")
print(f"Bad files   : {len(bad_files)}")