import math

from sqlalchemy import text

from database.connection import engine
from graphhopper_routes import find_routes


# ============================================================
# CONFIGURATION
# ============================================================

# How many OSM geometry points to skip between samples.
#
# Example:
# sample_every = 15
#
# means we don't process every single geometry point.
# This keeps the calculation efficient.
#
SAMPLE_EVERY = 15


# Number of nearest monitored locations that can contribute
# to a route.
#
# We use the nearest location for each sampled OSM point.
#
MAX_MONITORED_LOCATIONS = 8


# Weight given to traffic and weather.
#
# Traffic is more important for route selection.
TRAFFIC_WEIGHT = 0.60
WEATHER_WEIGHT = 0.40


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distance between two latitude/longitude
    coordinates in kilometres.
    """

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    delta_lat = math.radians(
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# ============================================================
# GET LATEST CONDITIONS
# ============================================================

def get_latest_conditions():
    """
    Get the latest merged traffic + weather record
    for every monitored location.
    """

    query = text("""
        SELECT DISTINCT ON (location_id)

            location_id,

            latitude,
            longitude,

            traffic_timestamp,
            weather_timestamp,

            current_speed,
            free_flow_speed,

            current_travel_time,
            free_flow_travel_time,

            traffic_delay,
            speed_reduction,

            confidence,
            road_closure,
            frc,

            temperature,
            feels_like,
            humidity,

            wind_speed,
            wind_direction,

            cloudiness,
            rain_mm,

            weather_main,
            weather_description

        FROM traffic_weather_data

        ORDER BY
            location_id,
            traffic_timestamp DESC
    """)

    with engine.connect() as conn:

        result = conn.execute(
            query
        )

        rows = result.mappings().all()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# FIND NEAREST MONITORED LOCATION
# ============================================================

def find_nearest_location(
    latitude,
    longitude,
    locations
):
    """
    Find the monitored location closest to an OSM
    route coordinate.
    """

    nearest = None
    nearest_distance = float("inf")

    for location in locations:

        distance = haversine_distance(

            latitude,
            longitude,

            float(location["latitude"]),
            float(location["longitude"])

        )

        if distance < nearest_distance:

            nearest_distance = distance
            nearest = location

    return nearest, nearest_distance


# ============================================================
# SAMPLE OSM ROUTE
# ============================================================

def sample_route_geometry(
    route,
    locations
):
    """
    Take the actual OSM LineString geometry and associate
    each sampled road point with the nearest monitored
    traffic/weather location.
    """

    geometry_points = route.get(
        "geometry_points",
        []
    )

    if not geometry_points:

        return []


    samples = []


    # --------------------------------------------------------
    # Sample actual OSM geometry
    # --------------------------------------------------------

    sampled_points = geometry_points[
        ::SAMPLE_EVERY
    ]


    # Always include the final point
    if (
        geometry_points
        and
        geometry_points[-1]
        not in sampled_points
    ):

        sampled_points.append(
            geometry_points[-1]
        )


    # --------------------------------------------------------
    # Find nearest monitored location
    # --------------------------------------------------------

    for point in sampled_points:

        # GraphHopper geometry is:
        #
        # [longitude, latitude]
        #

        longitude = float(
            point[0]
        )

        latitude = float(
            point[1]
        )


        nearest, distance = (
            find_nearest_location(

                latitude,
                longitude,

                locations

            )
        )


        if nearest is not None:

            samples.append({

                "route_latitude":
                    latitude,

                "route_longitude":
                    longitude,

                "nearest_location":
                    nearest["location_id"],

                "distance_to_location_km":
                    round(
                        distance,
                        3
                    ),

                "condition":
                    nearest

            })


    return samples


# ============================================================
# CONGESTION PERCENTAGE
# ============================================================

def calculate_congestion_percentage(
    condition
):
    """
    Calculate congestion percentage:

        ((free flow speed - current speed)
         / free flow speed) * 100

    Result is restricted to 0-100.
    """

    current_speed = condition.get(
        "current_speed"
    )

    free_flow_speed = condition.get(
        "free_flow_speed"
    )


    if (
        current_speed is None
        or
        free_flow_speed is None
        or
        float(free_flow_speed) <= 0
    ):

        return 0.0


    congestion = (

        (
            float(free_flow_speed)
            -
            float(current_speed)
        )

        /

        float(free_flow_speed)

    ) * 100


    return round(

        max(
            0,
            min(
                100,
                congestion
            )
        ),

        2

    )


# ============================================================
# CONGESTION LABEL
# ============================================================

def congestion_label(
    congestion_percentage
):

    if congestion_percentage < 20:

        return "LOW"

    elif congestion_percentage < 50:

        return "MODERATE"

    elif congestion_percentage < 70:

        return "HIGH"

    else:

        return "SEVERE"


# ============================================================
# WEATHER RISK SCORE
# ============================================================

def calculate_weather_risk(
    condition
):
    """
    Calculate a numerical weather risk score from 0-100.

    0   = very low weather risk
    100 = very high weather risk
    """

    score = 0.0


    # --------------------------------------------------------
    # Rain
    # --------------------------------------------------------

    rain = condition.get(
        "rain_mm"
    )

    if rain is not None:

        rain = float(rain)

        if rain >= 10:

            score += 50

        elif rain >= 5:

            score += 35

        elif rain >= 2:

            score += 20

        elif rain > 0:

            score += 10


    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------

    wind = condition.get(
        "wind_speed"
    )

    if wind is not None:

        wind = float(wind)

        if wind >= 15:

            score += 20

        elif wind >= 10:

            score += 12

        elif wind >= 6:

            score += 5


    # --------------------------------------------------------
    # Weather condition
    # --------------------------------------------------------

    weather_main = condition.get(
        "weather_main"
    )


    if weather_main:

        weather_main = str(
            weather_main
        ).lower()


        if weather_main in [

            "thunderstorm",
            "tornado"

        ]:

            score += 30


        elif weather_main in [

            "rain",
            "drizzle",
            "snow"

        ]:

            score += 20


        elif weather_main in [

            "fog",
            "mist"

        ]:

            score += 15


    # --------------------------------------------------------
    # Humidity
    # --------------------------------------------------------

    humidity = condition.get(
        "humidity"
    )

    if humidity is not None:

        humidity = float(
            humidity
        )

        if humidity >= 90:

            score += 10

        elif humidity >= 80:

            score += 5


    return round(

        min(
            100,
            score
        ),

        2

    )


# ============================================================
# WEATHER LABEL
# ============================================================

def weather_label(
    weather_score
):

    if weather_score < 20:

        return "LOW"

    elif weather_score < 50:

        return "MODERATE"

    elif weather_score < 75:

        return "HIGH"

    else:

        return "SEVERE"


# ============================================================
# CALCULATE ROUTE CONDITIONS
# ============================================================

def evaluate_route(
    route,
    locations
):
    """
    Evaluate one complete OSM route using traffic and
    weather information along the route.
    """

    samples = sample_route_geometry(

        route,

        locations

    )


    if not samples:

        return None


    # ========================================================
    # REMOVE DUPLICATE MONITORED LOCATIONS
    #
    # Several OSM points may be closest to the same
    # monitored location.
    #
    # We don't want one location to dominate the score.
    # ========================================================

    unique_conditions = {}

    for sample in samples:

        location_id = sample[
            "nearest_location"
        ]


        unique_conditions[
            location_id
        ] = sample["condition"]


    conditions = list(
        unique_conditions.values()
    )


    # ========================================================
    # CONGESTION
    # ========================================================

    congestion_values = []


    for condition in conditions:

        congestion = (
            calculate_congestion_percentage(
                condition
            )
        )

        congestion_values.append(
            congestion
        )


    if congestion_values:

        average_congestion = round(

            sum(
                congestion_values
            )
            /
            len(
                congestion_values
            ),

            2

        )

        max_congestion = round(

            max(
                congestion_values
            ),

            2

        )

    else:

        average_congestion = 0
        max_congestion = 0


    # ========================================================
    # WEATHER
    # ========================================================

    weather_values = []


    for condition in conditions:

        weather_score = (
            calculate_weather_risk(
                condition
            )
        )

        weather_values.append(
            weather_score
        )


    if weather_values:

        average_weather = round(

            sum(
                weather_values
            )
            /
            len(
                weather_values
            ),

            2

        )

        max_weather = round(

            max(
                weather_values
            ),

            2

        )

    else:

        average_weather = 0
        max_weather = 0


    # ========================================================
    # ROAD CLOSURE
    # ========================================================

    road_closed = any(

        condition.get(
            "road_closure"
        ) is True

        for condition in conditions

    )


    # ========================================================
    # OVERALL ROUTE SCORE
    #
    # Higher = better
    #
    # Traffic contribution:
    #
    #     100 - congestion
    #
    # Weather contribution:
    #
    #     100 - weather risk
    # ========================================================

    traffic_score = (

        100
        -
        average_congestion

    )


    weather_score = (

        100
        -
        average_weather

    )


    overall_score = (

        TRAFFIC_WEIGHT
        *
        traffic_score

        +

        WEATHER_WEIGHT
        *
        weather_score

    )


    # --------------------------------------------------------
    # Road closure makes route unsuitable.
    # --------------------------------------------------------

    if road_closed:

        overall_score = 0


    overall_score = round(

        max(
            0,
            min(
                100,
                overall_score
            )
        ),

        2

    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "route_id":
            route["route_id"],

        "route_type":
            route["route_type"],

        "via":
            route["via"],

        "distance_km":
            route["distance_km"],

        "duration_minutes":
            route["duration_minutes"],

        "geometry":
            route["geometry"],

        "geometry_points":
            route["geometry_points"],

        "sample_count":
            len(samples),

        "locations_used":
            list(
                unique_conditions.keys()
            ),

        "average_congestion_percentage":
            average_congestion,

        "max_congestion_percentage":
            max_congestion,

        "congestion_level":
            congestion_label(
                average_congestion
            ),

        "average_weather_risk":
            average_weather,

        "max_weather_risk":
            max_weather,

        "weather_level":
            weather_label(
                average_weather
            ),

        "road_closed":
            road_closed,

        "traffic_score":
            round(
                traffic_score,
                2
            ),

        "weather_score":
            round(
                weather_score,
                2
            ),

        "overall_score":
            overall_score

    }


# ============================================================
# EVALUATE ALL ROUTES
# ============================================================

def evaluate_routes(
    routes
):

    print()
    print("=" * 75)
    print("ROUTE CONDITION ANALYSIS")
    print("=" * 75)


    # --------------------------------------------------------
    # Get latest traffic + weather records
    # --------------------------------------------------------

    locations = get_latest_conditions()


    if not locations:

        raise RuntimeError(

            "No traffic/weather "
            "conditions found in database."

        )


    evaluated_routes = []


    # --------------------------------------------------------
    # Evaluate each OSM route
    # --------------------------------------------------------

    for route in routes:

        result = evaluate_route(

            route,

            locations

        )


        if result is not None:

            evaluated_routes.append(
                result
            )


    # --------------------------------------------------------
    # Sort by overall score
    #
    # Highest score = best route
    # --------------------------------------------------------

    evaluated_routes.sort(

        key=lambda route:
        route["overall_score"],

        reverse=True

    )


    # --------------------------------------------------------
    # Assign ranking
    # --------------------------------------------------------

    for rank, route in enumerate(

        evaluated_routes,

        start=1

    ):

        route["rank"] = rank


    return evaluated_routes


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_route_recommendation(
    routes
):

    print()
    print("=" * 75)
    print("ROUTE RECOMMENDATIONS")
    print("=" * 75)


    for route in routes:

        print()

        print(
            f"Rank {route['rank']} "
            f"→ Route {route['route_id']}"
        )

        print("-" * 50)


        print(
            f"Distance       : "
            f"{route['distance_km']} km"
        )


        print(
            f"Travel time    : "
            f"{route['duration_minutes']} min"
        )


        if route["via"]:

            print(
                f"Via            : "
                f"{route['via']}"
            )

        else:

            print(
                "Via            : Direct"
            )


        print(
            f"Congestion     : "
            f"{route['average_congestion_percentage']}%"
            f" ({route['congestion_level']})"
        )


        print(
            f"Max congestion : "
            f"{route['max_congestion_percentage']}%"
        )


        print(
            f"Weather risk   : "
            f"{route['average_weather_risk']}/100"
            f" ({route['weather_level']})"
        )


        print(
            f"Traffic score  : "
            f"{route['traffic_score']}/100"
        )


        print(
            f"Weather score  : "
            f"{route['weather_score']}/100"
        )


        print(
            f"Overall score  : "
            f"{route['overall_score']}/100"
        )


        print(
            f"Locations used : "
            f"{', '.join(route['locations_used'])}"
        )


        print(
            f"OSM points     : "
            f"{len(route['geometry_points'])}"
        )


    # ========================================================
    # FINAL RECOMMENDATION
    # ========================================================

    if routes:

        best = routes[0]


        print()
        print("=" * 75)
        print("BEST ROUTE")
        print("=" * 75)


        print()

        print(
            f"Route {best['route_id']}"
        )


        print(
            f"Distance      : "
            f"{best['distance_km']} km"
        )


        print(
            f"Travel time   : "
            f"{best['duration_minutes']} min"
        )


        print(
            f"Congestion    : "
            f"{best['average_congestion_percentage']}%"
            f" ({best['congestion_level']})"
        )


        print(
            f"Weather risk  : "
            f"{best['average_weather_risk']}/100"
            f" ({best['weather_level']})"
        )


        print(
            f"Overall score : "
            f"{best['overall_score']}/100"
        )


        print()

        print(
            "RECOMMENDED ROUTE SELECTED"
        )


    print()
    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 75)
    print("SMART CITY ROUTE CONDITION SYSTEM")
    print("=" * 75)


    # --------------------------------------------------------
    # Get locations from database
    # --------------------------------------------------------

    locations = get_latest_conditions()


    if not locations:

        print(
            "No traffic/weather data available."
        )

        raise SystemExit


    print()

    print(
        "AVAILABLE LOCATIONS:"
    )


    for location in locations:

        print(
            f"- {location['location_id']}"
        )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    print()

    origin = input(
        "Enter origin: "
    ).strip().upper()


    destination = input(
        "Enter destination: "
    ).strip().upper()


    if origin == destination:

        print(
            "Origin and destination "
            "cannot be the same."
        )

        raise SystemExit


    # --------------------------------------------------------
    # Generate OSM routes
    # --------------------------------------------------------

    try:

        routes = find_routes(

            origin,

            destination

        )


        if not routes:

            print(
                "No routes generated."
            )

            raise SystemExit


        # ----------------------------------------------------
        # Evaluate routes
        # ----------------------------------------------------

        evaluated_routes = evaluate_routes(
            routes
        )


        # ----------------------------------------------------
        # Display recommendation
        # ----------------------------------------------------

        display_route_recommendation(
            evaluated_routes
        )


    except Exception as e:

        print()

        print("=" * 75)
        print("ERROR")
        print("=" * 75)

        print(e)