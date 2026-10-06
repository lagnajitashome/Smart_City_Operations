from sqlalchemy import text
from database.connection import engine


# ============================================================
# GET LATEST TRAFFIC DATA FOR A LOCATION
# ============================================================

def get_latest_traffic(location_id):

    query = text("""
        SELECT
            location_id,
            current_speed,
            free_flow_speed,
            traffic_delay,
            traffic_timestamp
        FROM traffic_weather_data
        WHERE location_id = :location_id
        ORDER BY traffic_timestamp DESC
        LIMIT 1
    """)

    with engine.connect() as conn:

        result = conn.execute(
            query,
            {
                "location_id": location_id
            }
        )

        row = result.mappings().first()

    return row


# ============================================================
# CALCULATE CONGESTION PERCENTAGE
# ============================================================

def calculate_congestion_percentage(
    current_speed,
    free_flow_speed
):

    if free_flow_speed is None:
        return None

    if free_flow_speed <= 0:
        return 0

    congestion = (
        (free_flow_speed - current_speed)
        / free_flow_speed
    ) * 100

    # Keep value between 0 and 100
    congestion = max(
        0,
        min(100, congestion)
    )

    return round(congestion, 2)


# ============================================================
# CONGESTION LEVEL
# ============================================================

def get_congestion_level(
    congestion_percentage
):

    if congestion_percentage is None:
        return "UNKNOWN"

    if congestion_percentage <= 20:
        return "LOW"

    elif congestion_percentage <= 40:
        return "MODERATE"

    elif congestion_percentage <= 60:
        return "HIGH"

    else:
        return "SEVERE"


# ============================================================
# ANALYZE ONE LOCATION
# ============================================================

def analyze_location(location_id):

    traffic = get_latest_traffic(
        location_id
    )

    if traffic is None:

        return {
            "location_id": location_id,
            "congestion_percentage": None,
            "congestion_level": "NO DATA"
        }

    congestion = calculate_congestion_percentage(
        traffic["current_speed"],
        traffic["free_flow_speed"]
    )

    level = get_congestion_level(
        congestion
    )

    return {

        "location_id":
            location_id,

        "current_speed":
            traffic["current_speed"],

        "free_flow_speed":
            traffic["free_flow_speed"],

        "traffic_delay":
            traffic["traffic_delay"],

        "congestion_percentage":
            congestion,

        "congestion_level":
            level,

        "traffic_timestamp":
            traffic["traffic_timestamp"]
    }


# ============================================================
# ANALYZE A ROUTE
# ============================================================

def analyze_route(
    route_locations
):

    results = []

    for location in route_locations:

        result = analyze_location(
            location
        )

        results.append(result)

    return results


# ============================================================
# CALCULATE ROUTE CONGESTION
# ============================================================

def calculate_route_congestion(
    route_locations
):

    location_results = analyze_route(
        route_locations
    )

    valid_values = [

        r["congestion_percentage"]

        for r in location_results

        if r["congestion_percentage"] is not None
    ]

    if not valid_values:

        return {

            "congestion_percentage":
                None,

            "congestion_level":
                "NO DATA",

            "locations":
                location_results
        }


    # Average congestion across locations
    average_congestion = (
        sum(valid_values)
        / len(valid_values)
    )

    average_congestion = round(
        average_congestion,
        2
    )


    return {

        "congestion_percentage":
            average_congestion,

        "congestion_level":
            get_congestion_level(
                average_congestion
            ),

        "locations":
            location_results
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("CONGESTION ANALYSIS")
    print("=" * 70)

    test_locations = [
        "RASHBEHARI",
        "SALT_LAKE",
        "ESPLANADE",
        "VIP_ROAD"
    ]

    for location in test_locations:

        result = analyze_location(
            location
        )

        print()
        print(
            f"Location: {location}"
        )

        print(
            f"Current Speed: "
            f"{result.get('current_speed')}"
        )

        print(
            f"Free Flow Speed: "
            f"{result.get('free_flow_speed')}"
        )

        print(
            f"Congestion: "
            f"{result.get('congestion_percentage')}%"
        )

        print(
            f"Level: "
            f"{result.get('congestion_level')}"
        )

    print()
    print("=" * 70)