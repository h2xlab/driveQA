import carla
import os
import time
import numpy as np
import cv2
from collections import defaultdict
import random
import math
from carla import VehicleLightState
from itertools import product, permutations
import json


# WEATHER = {
#     # 'ClearNoon':carla.WeatherParameters.ClearNoon,
#     'WetNoon': carla.WeatherParameters.WetNoon,
#     'HardRainNoon': carla.WeatherParameters.HardRainNoon,
#     # 'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
#     'ClearMorning': carla.WeatherParameters(precipitation=0.0, sun_altitude_angle=20.0),
#     'HardRainMorning': carla.WeatherParameters(precipitation=80.0, precipitation_deposits=80.0, sun_altitude_angle=20.0),
#     'ClearSunset': carla.WeatherParameters.ClearSunset,
#     'WetSunset': carla.WeatherParameters.WetSunset,
#     'HardRainSunset': carla.WeatherParameters.HardRainSunset,
#     # 'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
#     'ClearNight': carla.WeatherParameters(precipitation=0.0, sun_altitude_angle=-5.0),
#     'WetNight': carla.WeatherParameters(precipitation=30.0, wetness=50.0, sun_altitude_angle=-5.0),
#     'HardRainNight': carla.WeatherParameters(precipitation=80.0, precipitation_deposits=80.0, sun_altitude_angle=-5.0),
#     # 'WetCloudyNight': carla.WeatherParameters(cloudiness=70.0, precipitation=30.0, sun_altitude_angle=-60.0),
# }

WEATHER = {
    'HardRainTwilight': carla.WeatherParameters(
        cloudiness=90.0, precipitation=60.0, precipitation_deposits=100.0, 
        wind_intensity=1.0, sun_azimuth_angle=315.0, sun_altitude_angle=0.0
    ),
    'ClearSunset': carla.WeatherParameters(
        cloudiness=15.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=45.0, sun_altitude_angle=15.0
    ),
    'WetNight': carla.WeatherParameters(
        cloudiness=20.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=225.0, sun_altitude_angle=-7.5
    ),
    'ClearNoon': carla.WeatherParameters(
        cloudiness=15.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=0.0, sun_altitude_angle=75.0
    ),
    'CloudyTwilight': carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=0.0, sun_altitude_angle=0.0
    ),
    'WetMorning': carla.WeatherParameters(
        cloudiness=20.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=180.0, sun_altitude_angle=35.0
    ),
    'WetDawn': carla.WeatherParameters(
        cloudiness=20.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=0.0, sun_altitude_angle=5.0
    ),
    'WetCloudyTwilight': carla.WeatherParameters(
        cloudiness=90.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=0.0, sun_altitude_angle=0.0
    ),
    'WetCloudyNight': carla.WeatherParameters(
        cloudiness=90.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=225.0, sun_altitude_angle=-7.5
    ),
    'CloudySunset': carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=270.0, sun_altitude_angle=15.0
    ),
    'MidRainSunset': carla.WeatherParameters(
        cloudiness=80.0, precipitation=30.0, precipitation_deposits=50.0, 
        wind_intensity=0.4, sun_azimuth_angle=270.0, sun_altitude_angle=15.0
    ),
    'SoftRainTwilight': carla.WeatherParameters(
        cloudiness=90.0, precipitation=15.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=90.0, sun_altitude_angle=0.0
    ),
    'MidRainNoon': carla.WeatherParameters(
        cloudiness=80.0, precipitation=30.0, precipitation_deposits=50.0, 
        wind_intensity=0.4, sun_azimuth_angle=90.0, sun_altitude_angle=75.0
    ),
    'ClearTwilight': carla.WeatherParameters(
        cloudiness=15.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=45.0, sun_altitude_angle=0.0
    ),
    'HardRainDawn': carla.WeatherParameters(
        cloudiness=90.0, precipitation=60.0, precipitation_deposits=100.0, 
        wind_intensity=1.0, sun_azimuth_angle=225.0, sun_altitude_angle=5.0
    ),
    'HardRainNight': carla.WeatherParameters(
        cloudiness=90.0, precipitation=60.0, precipitation_deposits=100.0, 
        wind_intensity=1.0, sun_azimuth_angle=225.0, sun_altitude_angle=-7.5
    ),
    'WetCloudyNoon': carla.WeatherParameters(
        cloudiness=90.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=180.0, sun_altitude_angle=75.0
    ),
    'CloudyDawn': carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=0.0, sun_altitude_angle=5.0
    ),
    'WetCloudyDawn': carla.WeatherParameters(
        cloudiness=90.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=45.0, sun_altitude_angle=5.0
    ),
    'SoftRainMorning': carla.WeatherParameters(
        cloudiness=90.0, precipitation=15.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=315.0, sun_altitude_angle=35.0
    ),
    'SoftRainSunset': carla.WeatherParameters(
        cloudiness=90.0, precipitation=15.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=270.0, sun_altitude_angle=15.0
    ),
    'WetCloudyMorning': carla.WeatherParameters(
        cloudiness=90.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=180.0, sun_altitude_angle=35.0
    ),
    'ClearMorning': carla.WeatherParameters(
        cloudiness=15.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=45.0, sun_altitude_angle=35.0
    ),
    'MidRainNight': carla.WeatherParameters(
        cloudiness=80.0, precipitation=30.0, precipitation_deposits=50.0, 
        wind_intensity=0.4, sun_azimuth_angle=135.0, sun_altitude_angle=-7.5
    ),
    'SoftRainNoon': carla.WeatherParameters(
        cloudiness=90.0, precipitation=15.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=315.0, sun_altitude_angle=75.0
    ),
    'SoftRainDawn': carla.WeatherParameters(
        cloudiness=90.0, precipitation=15.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=315.0, sun_altitude_angle=5.0
    ),
    'SoftRainNight': carla.WeatherParameters(
        cloudiness=90.0, precipitation=15.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=270.0, sun_altitude_angle=-7.5
    ),
    'WetCloudySunset': carla.WeatherParameters(
        cloudiness=90.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=0.0, sun_altitude_angle=15.0
    ),
    'WetNoon': carla.WeatherParameters(
        cloudiness=20.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=45.0, sun_altitude_angle=75.0
    ),
    'HardRainMorning': carla.WeatherParameters(
        cloudiness=90.0, precipitation=60.0, precipitation_deposits=100.0, 
        wind_intensity=1.0, sun_azimuth_angle=45.0, sun_altitude_angle=35.0
    ),
    'MidRainTwilight': carla.WeatherParameters(
        cloudiness=80.0, precipitation=30.0, precipitation_deposits=50.0, 
        wind_intensity=0.4, sun_azimuth_angle=0.0, sun_altitude_angle=0.0
    ),
    'ClearDawn': carla.WeatherParameters(
        cloudiness=15.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=90.0, sun_altitude_angle=5.0
    ),
    'MidRainDawn': carla.WeatherParameters(
        cloudiness=80.0, precipitation=30.0, precipitation_deposits=50.0, 
        wind_intensity=0.4, sun_azimuth_angle=135.0, sun_altitude_angle=5.0
    ),
    'WetTwilight': carla.WeatherParameters(
        cloudiness=20.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=225.0, sun_altitude_angle=0.0
    ),
    'HardRainNoon': carla.WeatherParameters(
        cloudiness=90.0, precipitation=60.0, precipitation_deposits=100.0, 
        wind_intensity=1.0, sun_azimuth_angle=90.0, sun_altitude_angle=75.0
    ),
    'ClearNight': carla.WeatherParameters(
        cloudiness=15.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=0.0, sun_altitude_angle=-7.5
    ),
    'WetSunset': carla.WeatherParameters(
        cloudiness=20.0, precipitation=0.0, precipitation_deposits=50.0, 
        wind_intensity=0.35, sun_azimuth_angle=270.0, sun_altitude_angle=15.0
    ),
    'CloudyNight': carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=45.0, sun_altitude_angle=-7.5
    ),
    'CloudyNoon': carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=45.0, sun_altitude_angle=75.0
    ),
    'CloudyMorning': carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, 
        wind_intensity=0.35, sun_azimuth_angle=90.0, sun_altitude_angle=35.0
    ),
    'MidRainMorning': carla.WeatherParameters(
        cloudiness=80.0, precipitation=30.0, precipitation_deposits=50.0, 
        wind_intensity=0.4, sun_azimuth_angle=0.0, sun_altitude_angle=35.0
    ),
    'HardRainSunset': carla.WeatherParameters(
        cloudiness=90.0, precipitation=60.0, precipitation_deposits=100.0, 
        wind_intensity=1.0, sun_azimuth_angle=0.0, sun_altitude_angle=15.0
    )
}


def save_metadata(filename, camera_params, vehicle_infos, town_name, weather):
    """
    Save metadata including camera parameters, vehicle info, town name, and weather to a JSON file.
    
    Parameters:
    - filename (str): The path to save the metadata JSON file.
    - camera_params (dict): Dictionary containing camera parameters.
    - vehicle_infos (list): List of dictionaries containing vehicle information.
    - town_name (str): The name of the town/map.
    - weather (str): The name of the weather condition.
    """
    metadata = {
        "town": town_name,
        "weather": weather,
        "camera_params": camera_params,
        "vehicles": vehicle_infos
    }
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)


def get_traffic_manager(client, port=4500):
    tm = client.get_trafficmanager(port)
    tm.set_global_distance_to_leading_vehicle(1.0)
    tm.set_hybrid_physics_mode(True)
    tm.set_synchronous_mode(True)
    return tm

def add_camera_sensor(world, location, rotation, fov='90', attach_to=None):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', fov)  # Wide field of view for larger coverage
    camera_transform = carla.Transform(location, rotation)
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=None)
    return camera


def save_image(image, image_name):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))  # BGRA format
    rgb_image = array[:, :, :3]  # Remove the alpha channel
    cv2.imwrite(image_name, rgb_image)


def find_intersections(world):
    """Calculate and return a list of averaged intersection locations in the CARLA world."""
    carla_map = world.get_map()
    topology = carla_map.get_topology()

    intersections = []

    # Loop through the topology and identify waypoints within junctions
    for segment in topology:
        start_wp = segment[0]  # Starting waypoint of each road segment

        if start_wp.is_junction:
            junction_id = start_wp.junction_id
            location = (start_wp.transform.location.x,
                        start_wp.transform.location.y,
                        start_wp.transform.location.z)
            intersections.append((junction_id, location))

    # Group locations by junction_id
    unique_intersections = defaultdict(list)
    for junction_id, location in intersections:
        unique_intersections[junction_id].append(location)

    # Calculate average location for each junction_id
    unique_sorted_intersections = []
    for junction_id, locations in unique_intersections.items():
        avg_x = sum(loc[0] for loc in locations) / len(locations)
        avg_y = sum(loc[1] for loc in locations) / len(locations)
        avg_z = sum(loc[2] for loc in locations) / len(locations)
        unique_sorted_intersections.append((junction_id, carla.Location(avg_x, avg_y, avg_z))) # id + location
    
    print(f"Number of intersections: {len(unique_sorted_intersections)}")
    return unique_sorted_intersections


def find_pre_turn_waypoints(world, min_distance=4.0, max_distance=6.0, step_size=0.5):
    """Calculate and return a list of pre-turn locations for intersections in the CARLA world."""
    carla_map = world.get_map()
    topology = carla_map.get_topology()
    preturn_waypoints = []

    # Loop through the topology to identify waypoints at junctions
    for segment in topology:
        start_wp = segment[0]  # Starting waypoint of each road segment

        if start_wp.is_junction:
            junction_id = start_wp.junction_id

            # Trace back from each junction waypoint to find pre-turn waypoints
            wp = start_wp
            while wp and wp.is_junction:
                junction_id = start_wp.junction_id
                
                distance = min_distance
                while distance <= max_distance:
                    previous_wps = wp.previous(distance)
                    if previous_wps:
                        wp = previous_wps[0]
                        break
                    distance += step_size  # Decrease distance and try again
                
                if wp:  # Waypoint found within the maximum distance
                    preturn_waypoints.append((junction_id, wp))
                    break  # Stop after finding one pre-turn location for this junction

    return preturn_waypoints


def assign_vehicle_directions_based_on_waypoints(vehicle_waypoint, distance=15.0, angle_thres=45):
    """
    Assigns a feasible driving direction ('straight', 'left', 'right') 
    based on the next waypoints available at an intersection.
    """
    next_waypoints = vehicle_waypoint.next(distance)  # get next waypoints

    directions = []
    for next_wp in next_waypoints:
        # print(vehicle_waypoint)
        # print(next_wp)
        # Calculate the angle between the current waypoint's yaw and the next waypoint
        angle_diff = next_wp.transform.rotation.yaw - vehicle_waypoint.transform.rotation.yaw
        angle_diff = (angle_diff + 180) % 360 - 180  # Normalize angle to -180 to 180 range
        # print(angle_diff)
        # Classify direction based on angle difference
        if abs(angle_diff) < angle_thres:
            direction = "straight"
        elif angle_diff < -angle_thres:
            direction = "left"
        elif angle_diff > angle_thres:
            direction = "right"
        else:
            direction = "unknown"

        directions.append(direction)

    # Filter out duplicates to avoid redundant directions
    directions = list(set(directions))

    # Assign a single direction if multiple are available (optional, based on need)
    return directions if directions else ["straight"]  # Default to "straight" if no directions found


def remove_duplicate_orders(unique_order):
    """
    Removes duplicate orders by standardizing the order within "and" parts.
    """
    results = set()
    for order in unique_order:
        # Split on the first occurrence of " -> " to handle multiple segments gracefully
        if " -> " in order:
            left, right = order.split(" -> ", 1)
        else:
            left, right = order, ""

        # Sort each part within "and" to ensure consistency
        left_parts = " and ".join(sorted(left.split(" and ")))
        right_parts = " and ".join(sorted(right.split(" and ")))

        # Reconstruct standardized order and add to results
        standardized_order = f"{left_parts} -> {right_parts}" if right_parts else left_parts
        results.add(standardized_order)

    return results


def replace_or_add_order(order, unique_orders):
    """
    Checks if a given order with potentially different "and" term order exists.
    If it exists, replaces it with the given order; otherwise, adds it to the set.
    """
    # print(f'order: {order}')
    # print(f'unique_orders2: {unique_orders}')

    # Split order on the first " -> " to handle multiple segments
    if " -> " in order:
        left, right = order.split(" -> ", 1)
    else:
        left, right = order, ""

    # Sort each part within "and" for standardization
    left_parts = " and ".join(sorted(left.split(" and ")))
    right_parts = " and ".join(sorted(right.split(" and ")))
    standardized_order = f"{left_parts} -> {right_parts}" if right_parts else left_parts

    # Check each existing order in unique_orders
    for existing_order in unique_orders:
        # Split the existing order similarly
        if " -> " in existing_order:
            existing_left, existing_right = existing_order.split(" -> ", 1)
        else:
            existing_left, existing_right = existing_order, ""

        # Standardize both parts for comparison
        existing_left_parts = " and ".join(sorted(existing_left.split(" and ")))
        existing_right_parts = " and ".join(sorted(existing_right.split(" and ")))
        existing_standardized = f"{existing_left_parts} -> {existing_right_parts}" if existing_right_parts else existing_left_parts

        # If a match is found, replace it with the original format of the order
        if existing_standardized == standardized_order:
            unique_orders.remove(existing_order)
            unique_orders.add(order)  # Add in original format
            return unique_orders

    # No match found, add the new order
    unique_orders.add(order)
    return unique_orders