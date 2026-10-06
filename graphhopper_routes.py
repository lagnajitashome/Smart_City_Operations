import os
import requests

from dotenv import load_dotenv
from sqlalchemy import text

from database.connection import engine


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GRAPHHOPPER_API_KEY = os.getenv(
    "GRAPHHOPPER_API_KEY"
)

GRAPHHOPPER_URL = (
    "https://graphhopper.com/api/1/route"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not GRAPHHOPPER_API_KEY:

    raise RuntimeError(
        "GRAPHHOPPER_API_KEY not found in .env file."
    )


# ============================================================
# GET LATEST LOCATION COORDINATES
# ============================================================

def get_location_data(location_id):

    query = text("""
        SELECT
            location_id,
            latitude,
            longitude
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
# GET ALL AVAILABLE MONITORING LOCATIONS
# ============================================================

def get_all_locations():

    query = text("""
        SELECT DISTINCT
            location_id
        FROM traffic_weather_data
        ORDER BY location_id
    """)

    with engine.connect() as conn:

        result = conn.execute(query)

        return [
            row[0]
            for row in result.fetchall()
        ]


# ============================================================
# REQUEST ROUTE FROM GRAPHHOPPER
#
# GraphHopper uses OpenStreetMap road data.
#
# The returned "points" field contains the actual route
# geometry.
# ============================================================

def get_route(
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    via_points=None
):

    # --------------------------------------------------------
    # Build GraphHopper points
    #
    # Format:
    #
    # latitude,longitude
    # --------------------------------------------------------

    points = [

        f"{origin_lat},{origin_lon}"

    ]


    # --------------------------------------------------------
    # Add optional intermediate waypoint
    # --------------------------------------------------------

    if via_points:

        for lat, lon in via_points:

            points.append(

                f"{lat},{lon}"

            )


    # --------------------------------------------------------
    # Add destination
    # --------------------------------------------------------

    points.append(

        f"{destination_lat},{destination_lon}"

    )


    # --------------------------------------------------------
    # GraphHopper parameters
    # --------------------------------------------------------

    params = {

        "key":
            GRAPHHOPPER_API_KEY,

        "profile":
            "car",

        "point":
            points,

        "instructions":
            "true",

        "calc_points":
            "true",

        "points_encoded":
            "false"

    }


    # --------------------------------------------------------
    # Send request
    # --------------------------------------------------------

    response = requests.get(

        GRAPHHOPPER_URL,

        params=params,

        timeout=30

    )


    # --------------------------------------------------------
    # Check HTTP response
    # --------------------------------------------------------

    response.raise_for_status()


    data = response.json()


    # --------------------------------------------------------
    # Check GraphHopper response
    # --------------------------------------------------------

    if "paths" not in data:

        raise RuntimeError(

            f"GraphHopper did not return "
            f"a route: {data}"

        )


    if not data["paths"]:

        raise RuntimeError(

            "GraphHopper returned "
            "an empty route list."

        )


    # We requested one route, so return the
    # first path.

    return data["paths"][0]


# ============================================================
# CHECK ROUTE GEOMETRY
# ============================================================

def get_route_geometry(path):

    geometry = path.get(
        "points"
    )


    if not geometry:

        raise ValueError(
            "GraphHopper route does not "
            "contain geometry."
        )


    coordinates = geometry.get(
        "coordinates",
        []
    )


    if not coordinates:

        raise ValueError(
            "GraphHopper route geometry "
            "contains no coordinates."
        )


    return geometry


# ============================================================
# REMOVE DUPLICATE ROUTES
# ============================================================

def remove_duplicate_routes(
    candidate_routes
):

    unique_routes = []

    seen = set()


    for candidate in candidate_routes:

        path = candidate["path"]


        # ----------------------------------------------------
        # Distance and duration signature
        # ----------------------------------------------------

        distance = round(

            path["distance"] / 1000,

            1

        )


        duration = round(

            path["time"] / 60000,

            1

        )


        signature = (

            distance,
            duration

        )


        if signature not in seen:

            seen.add(
                signature
            )

            unique_routes.append(
                candidate
            )


    return unique_routes


# ============================================================
# GENERATE CANDIDATE ROUTES
# ============================================================

def find_routes(
    origin,
    destination
):

    print()
    print("=" * 70)
    print("SMART CITY ROUTE GENERATION")
    print("=" * 70)


    print()

    print(
        f"Origin      : {origin}"
    )

    print(
        f"Destination : {destination}"
    )


    # ========================================================
    # GET ORIGIN
    # ========================================================

    origin_data = get_location_data(
        origin
    )


    if origin_data is None:

        raise ValueError(

            f"Origin location "
            f"not found: {origin}"

        )


    # ========================================================
    # GET DESTINATION
    # ========================================================

    destination_data = get_location_data(
        destination
    )


    if destination_data is None:

        raise ValueError(

            f"Destination location "
            f"not found: {destination}"

        )


    # ========================================================
    # DISPLAY COORDINATES
    # ========================================================

    print()

    print(
        "ORIGIN COORDINATES"
    )

    print(
        f"Latitude  : "
        f"{origin_data['latitude']}"
    )

    print(
        f"Longitude : "
        f"{origin_data['longitude']}"
    )


    print()

    print(
        "DESTINATION COORDINATES"
    )

    print(
        f"Latitude  : "
        f"{destination_data['latitude']}"
    )

    print(
        f"Longitude : "
        f"{destination_data['longitude']}"
    )


    # ========================================================
    # GET ALL MONITORING LOCATIONS
    #
    # These are used ONLY to create candidate routing
    # corridors.
    #
    # The actual route geometry still comes from OSM.
    # ========================================================

    all_locations = get_all_locations()


    candidate_locations = [

        location

        for location in all_locations

        if location != origin

        and location != destination

    ]


    # ========================================================
    # STORE CANDIDATE ROUTES
    # ========================================================

    candidate_routes = []


    # ========================================================
    # ROUTE 1 — DIRECT OSM ROUTE
    # ========================================================

    print()
    print("-" * 70)

    print(
        "Generating direct OSM route..."
    )

    print("-" * 70)


    try:

        path = get_route(

            origin_data["latitude"],
            origin_data["longitude"],

            destination_data["latitude"],
            destination_data["longitude"]

        )


        # Validate geometry

        get_route_geometry(
            path
        )


        candidate_routes.append({

            "route_type":
                "DIRECT",

            "via":
                None,

            "path":
                path

        })


    except Exception as e:

        print()

        print(
            f"Direct route failed: {e}"
        )


    # ========================================================
    # ALTERNATIVE CANDIDATE ROUTES
    #
    # We deliberately route through other monitoring
    # locations to explore different OSM road corridors.
    #
    # GraphHopper then determines the actual road path.
    # ========================================================

    for via_location in candidate_locations:


        via_data = get_location_data(
            via_location
        )


        if via_data is None:

            continue


        print()

        print(
            f"Generating OSM route via "
            f"{via_location}..."
        )


        try:

            path = get_route(

                origin_data["latitude"],
                origin_data["longitude"],

                destination_data["latitude"],
                destination_data["longitude"],

                via_points=[

                    (

                        via_data["latitude"],

                        via_data["longitude"]

                    )

                ]

            )


            # Validate geometry

            get_route_geometry(
                path
            )


            candidate_routes.append({

                "route_type":
                    "VIA_LOCATION",

                "via":
                    via_location,

                "path":
                    path

            })


        except Exception as e:

            print(

                f"Could not generate "
                f"route via "
                f"{via_location}: {e}"

            )


    # ========================================================
    # REMOVE DUPLICATE ROUTES
    # ========================================================

    candidate_routes = (

        remove_duplicate_routes(

            candidate_routes

        )

    )


    # ========================================================
    # SORT ROUTES BY TRAVEL TIME
    # ========================================================

    candidate_routes.sort(

        key=lambda route:
        route["path"]["time"]

    )


    # ========================================================
    # FORMAT ROUTES
    # ========================================================

    routes = []


    for index, candidate in enumerate(

        candidate_routes,

        start=1

    ):


        path = candidate["path"]


        geometry = get_route_geometry(
            path
        )


        coordinates = geometry.get(
            "coordinates",
            []
        )


        routes.append({

            "route_id":
                index,

            "route_type":
                candidate["route_type"],

            "via":
                candidate["via"],

            "distance_km":
                round(

                    path["distance"] / 1000,

                    2

                ),

            "duration_minutes":
                round(

                    path["time"] / 60000,

                    1

                ),

            # =================================================
            # IMPORTANT
            #
            # This is the actual OSM route geometry.
            # =================================================

            "geometry":
                geometry,

            "geometry_points":
                coordinates,

            "geometry_point_count":
                len(coordinates),

            "instructions":
                path.get(
                    "instructions",
                    []
                )

        })


    # ========================================================
    # DISPLAY ROUTES
    # ========================================================

    print()
    print("=" * 70)
    print("CANDIDATE OSM ROUTES")
    print("=" * 70)


    if not routes:

        print()

        print(
            "No routes were generated."
        )

        return []


    for route in routes:

        print()

        print(
            f"Route {route['route_id']}"
        )


        print(
            f"Type     : "
            f"{route['route_type']}"
        )


        if route["via"]:

            print(
                f"Via      : "
                f"{route['via']}"
            )


        print(
            f"Distance : "
            f"{route['distance_km']} km"
        )


        print(
            f"Duration : "
            f"{route['duration_minutes']} min"
        )


        print(
            f"OSM geometry points : "
            f"{route['geometry_point_count']}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        f"Total candidate routes: "
        f"{len(routes)}"
    )

    print("=" * 70)


    return routes


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # GET AVAILABLE LOCATIONS
    # ========================================================

    locations = get_all_locations()


    print()

    print("=" * 70)

    print(
        "AVAILABLE LOCATIONS"
    )

    print("=" * 70)


    for location in locations:

        print(
            f"- {location}"
        )


    print()


    # ========================================================
    # USER INPUT
    # ========================================================

    origin = input(

        "Enter origin: "

    ).strip().upper()


    destination = input(

        "Enter destination: "

    ).strip().upper()


    # ========================================================
    # VALIDATION
    # ========================================================

    if origin == destination:

        print()

        print(

            "Origin and destination "
            "cannot be the same."

        )


    elif origin not in locations:

        print()

        print(

            f"Invalid origin: "
            f"{origin}"

        )


    elif destination not in locations:

        print()

        print(

            f"Invalid destination: "
            f"{destination}"

        )


    # ========================================================
    # RUN ROUTING
    # ========================================================

    else:

        try:

            routes = find_routes(

                origin,

                destination

            )


            # =================================================
            # GEOMETRY PREVIEW
            # =================================================

            print()

            print("=" * 70)

            print(
                "ROUTE GEOMETRY CHECK"
            )

            print("=" * 70)


            for route in routes:

                print()

                print(
                    f"Route "
                    f"{route['route_id']}:"
                )


                print(

                    f"Geometry type : "
                    f"{route['geometry'].get('type')}"

                )


                print(

                    f"Geometry points : "
                    f"{route['geometry_point_count']}"

                )


                if route["geometry_points"]:

                    print(

                        f"First point : "
                        f"{route['geometry_points'][0]}"

                    )


                    print(

                        f"Last point  : "
                        f"{route['geometry_points'][-1]}"

                    )


            print()

            print("=" * 70)

            print(
                "ROUTING COMPLETED"
            )

            print("=" * 70)


        except Exception as e:

            print()

            print("=" * 70)

            print(
                "ERROR"
            )

            print("=" * 70)

            print(e)