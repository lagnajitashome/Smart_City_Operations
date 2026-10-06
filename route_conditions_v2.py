import math         
 
from sqlalchemy import text
 
from database.connection import engine
from graphhopper_routes_v2 import find_routes
 
 
# ============================================================
# CONFIGURATION
# ============================================================
 
SAMPLE_EVERY = 15
 
# --------------------------------------------------------------
# Scoring weights. Must sum to 1.0.
#
# TRAFFIC_WEIGHT  -- how much congestion matters
# WEATHER_WEIGHT  -- how much weather risk matters
# TIME_WEIGHT     -- how much travel time matters, relative to the
#                    OTHER candidate routes in this specific request
#                    (not an absolute scale -- the fastest route in
#                    the set scores 100, the slowest scores 0).
#
# This is the fix for the "34km detour beats the 5km direct route"
# problem: previously travel time had NO weight in the score at all,
# so a route could win purely by sampling a marginally less congested
# sensor, regardless of how much longer it actually took to drive.
# --------------------------------------------------------------
TRAFFIC_WEIGHT = 0.60
WEATHER_WEIGHT = 0.30
TIME_WEIGHT = 0.10
 
assert abs(TRAFFIC_WEIGHT + WEATHER_WEIGHT + TIME_WEIGHT - 1.0) < 1e-6, \
    "Scoring weights must sum to 1.0"
 
 
# ============================================================
# HAVERSINE DISTANCE (unchanged)
# ============================================================
 
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
 
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
 
 
# ============================================================
# GET LATEST CONDITIONS (unchanged)
# ============================================================
 
def get_latest_conditions():
    query = text("""
        SELECT DISTINCT ON (location_id)
            location_id, latitude, longitude,
            traffic_timestamp, weather_timestamp,
            current_speed, free_flow_speed,
            current_travel_time, free_flow_travel_time,
            traffic_delay, speed_reduction,
            confidence, road_closure, frc,
            temperature, feels_like, humidity,
            wind_speed, wind_direction,
            cloudiness, rain_mm,
            weather_main, weather_description
        FROM traffic_weather_data
        ORDER BY location_id, traffic_timestamp DESC
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.mappings().all()
    return [dict(row) for row in rows]
 
 
# ============================================================
# FIND NEAREST MONITORED LOCATION (unchanged)
# ============================================================
 
def find_nearest_location(latitude, longitude, locations):
    nearest = None
    nearest_distance = float("inf")
 
    for location in locations:
        distance = haversine_distance(
            latitude, longitude,
            float(location["latitude"]), float(location["longitude"]),
        )
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = location
 
    return nearest, nearest_distance
 
 
# ============================================================
# SAMPLE OSM ROUTE (unchanged)
# ============================================================
 
def sample_route_geometry(route, locations):
    geometry_points = route.get("geometry_points", [])
    if not geometry_points:
        return []
 
    samples = []
    sampled_points = geometry_points[::SAMPLE_EVERY]
 
    if geometry_points and geometry_points[-1] not in sampled_points:
        sampled_points.append(geometry_points[-1])
 
    for point in sampled_points:
        # GraphHopper geometry is [longitude, latitude]
        longitude = float(point[0])
        latitude = float(point[1])
 
        nearest, distance = find_nearest_location(latitude, longitude, locations)
 
        if nearest is not None:
            samples.append({
                "route_latitude": latitude,
                "route_longitude": longitude,
                "nearest_location": nearest["location_id"],
                "distance_to_location_km": round(distance, 3),
                "condition": nearest,
            })
 
    return samples
 
 
# ============================================================
# CONGESTION PERCENTAGE (unchanged)
# ============================================================
 
def calculate_congestion_percentage(condition):
    current_speed = condition.get("current_speed")
    free_flow_speed = condition.get("free_flow_speed")
 
    if current_speed is None or free_flow_speed is None or float(free_flow_speed) <= 0:
        return 0.0
 
    congestion = (
        (float(free_flow_speed) - float(current_speed)) / float(free_flow_speed)
    ) * 100
 
    return round(max(0, min(100, congestion)), 2)
 
 
def congestion_label(congestion_percentage):
    if congestion_percentage < 20:
        return "LOW"
    elif congestion_percentage < 50:
        return "MODERATE"
    elif congestion_percentage < 70:
        return "HIGH"
    else:
        return "SEVERE"
 
 
# ============================================================
# WEATHER RISK SCORE (unchanged)
# ============================================================
 
def calculate_weather_risk(condition):
    score = 0.0
 
    rain = condition.get("rain_mm")
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
 
    wind = condition.get("wind_speed")
    if wind is not None:
        wind = float(wind)
        if wind >= 15:
            score += 20
        elif wind >= 10:
            score += 12
        elif wind >= 6:
            score += 5
 
    weather_main = condition.get("weather_main")
    if weather_main:
        weather_main = str(weather_main).lower()
        if weather_main in ["thunderstorm", "tornado"]:
            score += 30
        elif weather_main in ["rain", "drizzle", "snow"]:
            score += 20
        elif weather_main in ["fog", "mist"]:
            score += 15
 
    humidity = condition.get("humidity")
    if humidity is not None:
        humidity = float(humidity)
        if humidity >= 90:
            score += 10
        elif humidity >= 80:
            score += 5
 
    return round(min(100, score), 2)
 
 
def weather_label(weather_score):
    if weather_score < 20:
        return "LOW"
    elif weather_score < 50:
        return "MODERATE"
    elif weather_score < 75:
        return "HIGH"
    else:
        return "SEVERE"
 
 
# ============================================================
# TIME SCORE -- new
#
# Scored RELATIVE to the other candidate routes in this request:
# the fastest route among the candidates scores 100, the slowest
# scores 0, everything else scales linearly between them.
# ============================================================
 
def calculate_time_score(duration_minutes, min_duration, max_duration):
    if max_duration == min_duration:
        return 100.0
 
    score = 100 - (
        (duration_minutes - min_duration) / (max_duration - min_duration)
    ) * 100
 
    return round(max(0, min(100, score)), 2)
 
 
def time_label(time_score):
    if time_score >= 80:
        return "FAST"
    elif time_score >= 50:
        return "MODERATE"
    elif time_score >= 20:
        return "SLOW"
    else:
        return "VERY SLOW"
 
 
# ============================================================
# EVALUATE ONE ROUTE'S CONDITIONS (traffic + weather only --
# time score is added afterward in evaluate_routes, since it
# needs to know the min/max duration across ALL candidates)
# ============================================================
 
def evaluate_route(route, locations):
    samples = sample_route_geometry(route, locations)
 
    if not samples:
        return None
 
    # Remove duplicate monitored locations so one station doesn't
    # dominate the score just because more sampled points snapped to it.
    unique_conditions = {}
    for sample in samples:
        unique_conditions[sample["nearest_location"]] = sample["condition"]
 
    conditions = list(unique_conditions.values())
 
    congestion_values = [calculate_congestion_percentage(c) for c in conditions]
    average_congestion = round(sum(congestion_values) / len(congestion_values), 2) if congestion_values else 0
    max_congestion = round(max(congestion_values), 2) if congestion_values else 0
 
    weather_values = [calculate_weather_risk(c) for c in conditions]
    average_weather = round(sum(weather_values) / len(weather_values), 2) if weather_values else 0
    max_weather = round(max(weather_values), 2) if weather_values else 0
 
    road_closed = any(c.get("road_closure") is True for c in conditions)
 
    traffic_score = round(100 - average_congestion, 2)
    weather_score = round(100 - average_weather, 2)
 
    return {
        "route_id": route["route_id"],
        "route_type": route["route_type"],
        "via": route["via"],
        "distance_km": route["distance_km"],
        "duration_minutes": route["duration_minutes"],
        "geometry": route["geometry"],
        "geometry_points": route["geometry_points"],
        "instructions": route.get("instructions", []),
        "sample_count": len(samples),
        "locations_used": list(unique_conditions.keys()),
        "average_congestion_percentage": average_congestion,
        "max_congestion_percentage": max_congestion,
        "congestion_level": congestion_label(average_congestion),
        "average_weather_risk": average_weather,
        "max_weather_risk": max_weather,
        "weather_level": weather_label(average_weather),
        "road_closed": road_closed,
        "traffic_score": traffic_score,
        "weather_score": weather_score,
        # time_score / overall_score / rank are filled in by evaluate_routes()
    }
 
 
# ============================================================
# EVALUATE ALL ROUTES
# ============================================================
 
def evaluate_routes(routes):
    print()
    print("=" * 75)
    print("ROUTE CONDITION ANALYSIS")
    print("=" * 75)
 
    locations = get_latest_conditions()
    if not locations:
        raise RuntimeError("No traffic/weather conditions found in database.")
 
    evaluated_routes = []
    for route in routes:
        result = evaluate_route(route, locations)
        if result is not None:
            evaluated_routes.append(result)
 
    if not evaluated_routes:
        return []
 
    # --------------------------------------------------------
    # Compute the relative time score now that we know the full
    # spread of durations across all candidate routes.
    # --------------------------------------------------------
    durations = [r["duration_minutes"] for r in evaluated_routes]
    min_duration = min(durations)
    max_duration = max(durations)
 
    for r in evaluated_routes:
        time_score = calculate_time_score(r["duration_minutes"], min_duration, max_duration)
        r["time_score"] = time_score
        r["time_level"] = time_label(time_score)
 
        overall_score = (
            TRAFFIC_WEIGHT * r["traffic_score"]
            + WEATHER_WEIGHT * r["weather_score"]
            + TIME_WEIGHT * time_score
        )
 
        if r["road_closed"]:
            overall_score = 0
 
        r["overall_score"] = round(max(0, min(100, overall_score)), 2)
 
    evaluated_routes.sort(key=lambda route: route["overall_score"], reverse=True)
 
    for rank, route in enumerate(evaluated_routes, start=1):
        route["rank"] = rank
 
    return evaluated_routes
 
 
# ============================================================
# DISPLAY RESULTS
# ============================================================
 
def display_route_recommendation(routes):
    print()
    print("=" * 75)
    print("ROUTE RECOMMENDATIONS")
    print("=" * 75)
 
    for route in routes:
        print()
        print(f"Rank {route['rank']} -> Route {route['route_id']}")
        print("-" * 50)
        print(f"Distance       : {route['distance_km']} km")
        print(f"Travel time    : {route['duration_minutes']} min")
 
        if route["via"]:
            print(f"Via            : {route['via']}")
        else:
            print("Via            : Direct")
 
        print(f"Congestion     : {route['average_congestion_percentage']}% ({route['congestion_level']})")
        print(f"Max congestion : {route['max_congestion_percentage']}%")
        print(f"Weather risk   : {route['average_weather_risk']}/100 ({route['weather_level']})")
        print(f"Traffic score  : {route['traffic_score']}/100")
        print(f"Weather score  : {route['weather_score']}/100")
        print(f"Time score     : {route['time_score']}/100 ({route['time_level']})")
        print(f"Overall score  : {route['overall_score']}/100")
        print(f"Locations used : {', '.join(route['locations_used'])}")
        print(f"OSM points     : {len(route['geometry_points'])}")
 
    if routes:
        best = routes[0]
 
        print()
        print("=" * 75)
        print("BEST ROUTE")
        print("=" * 75)
        print()
        print(f"Route {best['route_id']}")
        print(f"Distance      : {best['distance_km']} km")
        print(f"Travel time   : {best['duration_minutes']} min")
        print(f"Congestion    : {best['average_congestion_percentage']}% ({best['congestion_level']})")
        print(f"Weather risk  : {best['average_weather_risk']}/100 ({best['weather_level']})")
        print(f"Time score    : {best['time_score']}/100 ({best['time_level']})")
        print(f"Overall score : {best['overall_score']}/100")
        print()
        print("RECOMMENDED ROUTE SELECTED")
 
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
 
    locations = get_latest_conditions()
    if not locations:
        print("No traffic/weather data available.")
        raise SystemExit
 
    print()
    print("AVAILABLE LOCATIONS:")
    for location in locations:
        print(f"- {location['location_id']}")
 
    print()
    origin = input("Enter origin: ").strip().upper()
    destination = input("Enter destination: ").strip().upper()
 
    if origin == destination:
        print("Origin and destination cannot be the same.")
        raise SystemExit
 
    try:
        routes = find_routes(origin, destination)
 
        if not routes:
            print("No routes generated.")
            raise SystemExit
 
        evaluated_routes = evaluate_routes(routes)
        display_route_recommendation(evaluated_routes)
 
    except Exception as e:
        print()
        print("=" * 75)
        print("ERROR")
        print("=" * 75)
        print(e)
 