import os
import math
import requests

from dotenv import load_dotenv
from sqlalchemy import text

from database.connection import engine


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GRAPHHOPPER_API_KEY = os.getenv("GRAPHHOPPER_API_KEY")
GRAPHHOPPER_URL = "https://graphhopper.com/api/1/route"

if not GRAPHHOPPER_API_KEY:
    raise RuntimeError("GRAPHHOPPER_API_KEY not found in .env file.")

# How many alternative routes to try to get from GraphHopper's
# native algorithm before falling back to offset waypoints.
NATIVE_MAX_PATHS = 3

# How much longer (in GraphHopper's weight units, ~time) an
# alternative is allowed to be vs. the best route.
# Raise this if you're consistently getting 0-1 alternatives back.
NATIVE_MAX_WEIGHT_FACTOR = 1.8

# Max fraction of the route an alternative is allowed to share
# with the best route. Lower = more forced diversity.
NATIVE_MAX_SHARE_FACTOR = 0.6

DEDUPE_OVERLAP_THRESHOLD = 0.8
OVERLAP_TOLERANCE_KM = 0.05


# ============================================================
# DB LOOKUPS (unchanged)
# ============================================================

def get_location_data(location_id):
    query = text("""
        SELECT location_id, latitude, longitude
        FROM traffic_weather_data
        WHERE location_id = :location_id
        ORDER BY traffic_timestamp DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"location_id": location_id})
        return result.mappings().first()


def get_all_locations():
    query = text("""
        SELECT DISTINCT location_id
        FROM traffic_weather_data
        ORDER BY location_id
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]


# ============================================================
# SINGLE ROUTE REQUEST (unchanged, still used for offset via-points)
# ============================================================

def get_route(origin_lat, origin_lon, destination_lat, destination_lon, via_points=None):
    points = [f"{origin_lat},{origin_lon}"]

    if via_points:
        for lat, lon in via_points:
            points.append(f"{lat},{lon}")

    points.append(f"{destination_lat},{destination_lon}")

    params = {
        "key": GRAPHHOPPER_API_KEY,
        "profile": "car",
        "point": points,
        "instructions": "true",
        "calc_points": "true",
        "points_encoded": "false",
    }

    response = requests.get(GRAPHHOPPER_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "paths" not in data or not data["paths"]:
        raise RuntimeError(f"GraphHopper did not return a route: {data}")

    return data["paths"][0]


# ============================================================
# NATIVE ALTERNATIVE ROUTES (new — replaces the sensor via-point loop)
# ============================================================

def get_native_alternative_routes(origin_lat, origin_lon, destination_lat, destination_lon):
    """
    Asks GraphHopper's own alternative_route algorithm for distinct
    routes, instead of forcing detours through arbitrary sensor
    locations. This is the correct tool for the job — GraphHopper
    explores the actual road graph for viable, competitive alternatives.
    """
    params = {
        "key": GRAPHHOPPER_API_KEY,
        "profile": "car",
        "point": [f"{origin_lat},{origin_lon}", f"{destination_lat},{destination_lon}"],
        "algorithm": "alternative_route",
        "alternative_route.max_paths": NATIVE_MAX_PATHS,
        "alternative_route.max_weight_factor": NATIVE_MAX_WEIGHT_FACTOR,
        "alternative_route.max_share_factor": NATIVE_MAX_SHARE_FACTOR,
        "instructions": "true",
        "calc_points": "true",
        "points_encoded": "false",
    }

    response = requests.get(GRAPHHOPPER_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    raw_paths = data.get("paths", [])

    # GraphHopper normally returns each route directly as a path object.
    # Normalize wrapped {"path": {...}} responses too.
    normalized_paths = []
    for item in raw_paths:
        if isinstance(item, dict) and isinstance(item.get("path"), dict):
            normalized_paths.append(item["path"])
        elif isinstance(item, dict):
            normalized_paths.append(item)

    return normalized_paths


# ============================================================
# GEOMETRY-BASED DEDUPE (new — replaces the distance/duration signature)
# ============================================================

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(d_lon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _overlap_ratio(coords_a, coords_b):
    """Fraction of sampled points in A that have a near-match in B."""
    if not coords_a or not coords_b:
        return 0.0

    sample_a = coords_a[:: max(1, len(coords_a) // 40)]
    sample_b = coords_b[:: max(1, len(coords_b) // 40)]

    matches = 0
    for lon_a, lat_a in sample_a:
        for lon_b, lat_b in sample_b:
            if _haversine_km(lat_a, lon_a, lat_b, lon_b) <= OVERLAP_TOLERANCE_KM:
                matches += 1
                break
    return matches / len(sample_a)


def remove_duplicate_routes(candidate_routes):
    """
    Keeps a route only if it's not geometrically near-identical to a
    route already kept. Assumes candidate_routes[0] is the shortest/
    direct route (sort by distance before calling this).
    """
    if not candidate_routes:
        return []

    kept = [candidate_routes[0]]
    for candidate in candidate_routes[1:]:
        cand_coords = candidate.get("points", {}).get("coordinates", [])
        too_similar = any(
            _overlap_ratio(cand_coords, kept_route.get("points", {}).get("coordinates", [])) >= DEDUPE_OVERLAP_THRESHOLD
            for kept_route in kept
        )
        if not too_similar:
            kept.append(candidate)
    return kept


# ============================================================
# GEOMETRY VALIDATION (unchanged)
# ============================================================

def get_route_geometry(path):
    geometry = path.get("points")
    if not geometry:
        raise ValueError("GraphHopper route does not contain geometry.")

    coordinates = geometry.get("coordinates", [])
    if not coordinates:
        raise ValueError("GraphHopper route geometry contains no coordinates.")

    return geometry


# ============================================================
# GENERATE CANDIDATE ROUTES
# ============================================================

def find_routes(origin, destination):
    print()
    print('=' * 70)
    print('SMART CITY ROUTE GENERATION')
    print('=' * 70)
    print(f'\nOrigin      : {origin}')
    print(f'Destination : {destination}')

    origin_data = get_location_data(origin)
    if origin_data is None:
        raise ValueError(f'Origin location not found: {origin}')

    destination_data = get_location_data(destination)
    if destination_data is None:
        raise ValueError(f'Destination location not found: {destination}')

    o_lat, o_lon = origin_data['latitude'], origin_data['longitude']
    d_lat, d_lon = destination_data['latitude'], destination_data['longitude']

    print(f'\nORIGIN COORDINATES\nLatitude  : {o_lat}\nLongitude : {o_lon}')
    print(f'\nDESTINATION COORDINATES\nLatitude  : {d_lat}\nLongitude : {d_lon}')

    print('\n' + '-' * 70)
    print('Requesting genuine alternative routes from GraphHopper...')
    print('-' * 70)

    paths = get_native_alternative_routes(o_lat, o_lon, d_lat, d_lon)

    if not paths:
        print('\nGraphHopper returned no routes.')
        return []

    # Validate and remove near-identical OSM geometries.
    valid_paths = []
    for path in paths:
        get_route_geometry(path)
        valid_paths.append(path)

    valid_paths.sort(key=lambda p: p.get('distance', float('inf')))
    valid_paths = remove_duplicate_routes(valid_paths)

    # This is only display/order information. Travel time is NOT used
    # anywhere in the route score; route_conditions.py handles scoring.
    valid_paths.sort(key=lambda p: p.get('time', float('inf')))

    routes = []
    for index, path in enumerate(valid_paths, start=1):
        geometry = get_route_geometry(path)
        coordinates = geometry.get('coordinates', [])

        routes.append({
            'route_id': index,
            'route_type': 'DIRECT' if index == 1 else 'GRAPHHOPPER_ALTERNATIVE',
            'via': None,
            'distance_km': round(path['distance'] / 1000, 2),
            'duration_minutes': round(path['time'] / 60000, 1),
            'geometry': geometry,
            'geometry_points': coordinates,
            'geometry_point_count': len(coordinates),
            'instructions': path.get('instructions', [])
        })

    print('\n' + '=' * 70)
    print('CANDIDATE OSM ROUTES')
    print('=' * 70)

    for route in routes:
        print(f"\nRoute {route['route_id']}")
        print(f"Type     : {route['route_type']}")
        print(f"Distance : {route['distance_km']} km")
        print(f"Duration : {route['duration_minutes']} min")
        print(f"OSM geometry points : {route['geometry_point_count']}")

    print(f"\n{'=' * 70}\nTotal genuine OSM routes: {len(routes)}\n{'=' * 70}")
    return routes


# MAIN PROGRAM (unchanged)
# ============================================================

if __name__ == "__main__":
    locations = get_all_locations()

    print("\n" + "=" * 70)
    print("AVAILABLE LOCATIONS")
    print("=" * 70)
    for location in locations:
        print(f"- {location}")

    origin = input("\nEnter origin: ").strip().upper()
    destination = input("Enter destination: ").strip().upper()

    if origin == destination:
        print("\nOrigin and destination cannot be the same.")
    elif origin not in locations:
        print(f"\nInvalid origin: {origin}")
    elif destination not in locations:
        print(f"\nInvalid destination: {destination}")
    else:
        try:
            routes = find_routes(origin, destination)

            print("\n" + "=" * 70)
            print("ROUTE GEOMETRY CHECK")
            print("=" * 70)

            for route in routes:
                print(f"\nRoute {route['route_id']}:")
                print(f"Geometry type : {route['geometry'].get('type')}")
                print(f"Geometry points : {route['geometry_point_count']}")
                if route["geometry_points"]:
                    print(f"First point : {route['geometry_points'][0]}")
                    print(f"Last point  : {route['geometry_points'][-1]}")

            print(f"\n{'=' * 70}\nROUTING COMPLETED\n{'=' * 70}")

        except Exception as e:
           print()
           print("=" * 70)
           print("ERROR")
           print("=" * 70)
           print(e)