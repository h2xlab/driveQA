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
from utils import *

seed = 2025
random.seed(seed)

# vehicle color mapping
color_mapping = {
    '42,61,214': 'Blue',
    '38,38,38': 'Gray',
    '79,33,85': 'Purple',
    '155,0,0': 'Red',
    '255,250,122': 'Yellow',
    '0,0,0': 'Black'
}


def is_conflicting_at_T_intersection(vehicle1, vehicle2):
    """
    Determine if two vehicles have a conflict at the uncontrolled T intersection based on their driving directions.
    Returns True if they conflict, False otherwise.
    """
    # Extract directions and road types
    direction1, road_type1 = vehicle1['direction'], vehicle1['road_type']
    direction2, road_type2 = vehicle2['direction'], vehicle2['road_type']
    
    # Rule 1: Any straight and left turn combination causes conflict
    if (direction1 == 'straight' and direction2 == 'left') or (direction1 == 'left' and direction2 == 'straight'):
        return True

    # Rule 2: main_left straight and stem right turn causes conflict
    if (road_type1 == 'main_left' and direction1 == 'straight' and road_type2 == 'stem' and direction2 == 'right') or \
       (road_type2 == 'main_left' and direction2 == 'straight' and road_type1 == 'stem' and direction1 == 'right'):
        return True

    # Rule 3: Right turn and right turn never causes conflict
    if direction1 == 'right' and direction2 == 'right':
        return False

    # Rule 4: main_right left turn and stem left turn causes conflict
    if (road_type1 == 'main_right' and direction1 == 'left' and road_type2 == 'stem' and direction2 == 'left') or \
       (road_type2 == 'main_right' and direction2 == 'left' and road_type1 == 'stem' and direction1 == 'left'):
        return True

    # Rule 5: main_right left turn and main_left right turn causes conflict
    if (road_type1 == 'main_right' and direction1 == 'left' and road_type2 == 'main_left' and direction2 == 'right') or \
       (road_type2 == 'main_right' and direction2 == 'left' and road_type1 == 'main_left' and direction1 == 'right'):
        return True

    # no conflict
    return False


def create_direct_questions(vehicle_info_list):
    """
    Generates right-of-way questions based on all possible combinations of vehicle directions.
    """
    # Extract possible directions for each vehicle
    direction_options = [vehicle['directions'] for vehicle in vehicle_info_list]

    # Generate all possible combinations of directions
    combinations_list = [combo for combo in product(*direction_options)]
    questions = []

    for i, combo in enumerate(combinations_list):
        # print(combo)
        vehicle_infos = {}
        for idx, direction in enumerate(combo):
            # print(vehicle_info_list[idx])
            if 'straight' in vehicle_info_list[idx]['directions'] and 'right' in vehicle_info_list[idx]['directions']:
                road_type = 'main_left'
            elif 'straight' in vehicle_info_list[idx]['directions'] and 'left' in vehicle_info_list[idx]['directions']:
                road_type = 'main_right'
            else:
                road_type = 'stem'

            vehicle_infos[idx] = {
                'color': vehicle_info_list[idx]['color'],
                'direction': direction,
                'location': vehicle_info_list[idx]['location'],
                'yaw': vehicle_info_list[idx]['yaw'],
                'road_type': road_type
            }

        # Generate question text
        descriptions = [
            f"The {color_mapping[vehicle['color']]} car will {'go straight' if vehicle['direction'] == 'straight' else 'turn ' + vehicle['direction']}."
            for vehicle in vehicle_infos.values()
        ]
        question_text = f"Question {i+1}: Assuming several cars arrive at an uncontrolled T-intersection simultaneously. " + " ".join(descriptions) + " Who has the right-of-way at the intersection? (Note: ignore the cars which have no conflict with other cars)"

        # Format options based on vehicle colors
        option_labels = ['A', 'B', 'C', 'D']
        options = [f"{option_labels[i]}. {color_mapping[vehicle['color']]} car" for i, vehicle in vehicle_infos.items()]
        options.append(f"{option_labels[len(vehicle_infos)]}. No one needs to yield because there is no conflict.")
        if len(options) < 4:
            options.append('D. None of the above.')
        formatted_options = "\n".join(options)

        # Determine the right-of-way answer for this scenario
        right_of_way_vehicle = determine_right_of_way(vehicle_infos)
        
        # Get the reason for the right-of-way decision
        reason = get_right_of_way_reason(vehicle_infos, right_of_way_vehicle)
        
        if right_of_way_vehicle:
            vehicle_index = list(vehicle_infos.values()).index(right_of_way_vehicle)
            # answer = f"Correct Answer: {option_labels[vehicle_index]}. {color_mapping[right_of_way_vehicle['color']]} car."
            answer = f"Correct Answer: {option_labels[vehicle_index]}. {color_mapping[right_of_way_vehicle['color']]} car. {reason}"
        else:
            # answer = f"Correct Answer: {option_labels[len(vehicle_infos)]}. No one needs to yield because there is no conflict."
            answer = f"Correct Answer: {option_labels[len(vehicle_infos)]}. No one needs to yield because there is no conflict. {reason}"

        # Combine question, options, and answer
        full_question = f"{question_text}\n{formatted_options}\n\n{answer}\n"

        questions.append(full_question)

    return questions


def determine_right_of_way(vehicle_infos):
    """
    Determine the right-of-way based on vehicle positions and road priority rules.
    Main road straight traffic has priority over other movements at T-intersections.
    Returns the vehicle with the right-of-way, or indicates if there is no conflict.
    """
    vehicles = list(vehicle_infos.values())
    num_vehicles = len(vehicles)
    conflicting_vehicles = []

    # Check for conflicts among all pairs of vehicles
    for i in range(num_vehicles):
        for j in range(i + 1, num_vehicles):
            if is_conflicting_at_T_intersection(vehicles[i], vehicles[j]):
                if vehicles[i] not in conflicting_vehicles:
                    conflicting_vehicles.append(vehicles[i])
                if vehicles[j] not in conflicting_vehicles:
                    conflicting_vehicles.append(vehicles[j])

    # If there are no conflicting vehicles, return that no one needs to yield
    if not conflicting_vehicles:
        return None  # No conflict, so "No one needs to yield"

    # First priority: Main road straight
    main_straight = [v for v in conflicting_vehicles if v['road_type'] in ['main_left', 'main_right'] and v['direction'] == 'straight']
    if main_straight:
        if len(main_straight) == 1:
            return main_straight[0]
        else:
            return get_right_most_vehicle(main_straight)

    # Then follow standard priority
    right_turn_vehicles = [v for v in conflicting_vehicles if v['direction'] == 'right']
    straight_vehicles = [v for v in conflicting_vehicles if v['direction'] == 'straight']
    left_turn_vehicles = [v for v in conflicting_vehicles if v['direction'] == 'left']

    # Check right turn vehicles
    if right_turn_vehicles:
        if len(right_turn_vehicles) == 1:
            return right_turn_vehicles[0]
        else:
            return get_right_most_vehicle(right_turn_vehicles)

    # Check straight vehicles
    if straight_vehicles:
        if len(straight_vehicles) == 1:
            return straight_vehicles[0]
        else:
            return get_right_most_vehicle(straight_vehicles)

    # Check left turn vehicles
    if left_turn_vehicles:
        if len(left_turn_vehicles) == 1:
            return left_turn_vehicles[0]
        else:
            return get_right_most_vehicle(left_turn_vehicles)

    return None


def get_right_most_vehicle(vehicles):
    """
    Determine the vehicle with the right-of-way based on fixed road_type priority.
    """
    # Define road_type priority
    road_type_priority = {'main_right': 1, 'stem': 2, 'main_left': 3}
    
    # Sort vehicles based on road_type priority
    vehicles_sorted = sorted(vehicles, key=lambda v: road_type_priority[v['road_type']])
    return vehicles_sorted[0] if vehicles_sorted else None


def create_order_questions(vehicle_info_list):
    """
    Generates questions asking for the order of right-of-way for scenarios with potentially non-conflicting vehicles.
    """
    direction_options = [vehicle['directions'] for vehicle in vehicle_info_list]
    combinations_list = [combo for combo in product(*direction_options)]
    questions = []

    for question_id, combo in enumerate(combinations_list):
        # print(f'--- Question {question_id+1} ---')
        vehicle_infos = {}
        for idx, direction in enumerate(combo):
            if 'straight' in vehicle_info_list[idx]['directions'] and 'right' in vehicle_info_list[idx]['directions']:
                road_type = 'main_left'
            elif 'straight' in vehicle_info_list[idx]['directions'] and 'left' in vehicle_info_list[idx]['directions']:
                road_type = 'main_right'
            else:
                road_type = 'stem'

            vehicle_infos[idx] = {
                'color': vehicle_info_list[idx]['color'],
                'direction': direction,
                'location': vehicle_info_list[idx]['location'],
                'yaw': vehicle_info_list[idx]['yaw'],
                'road_type': road_type
            }

        # Get the right-of-way order including simultaneous vehicles
        result_order = determine_right_of_way_order(vehicle_infos)
        # Get the reason for the right-of-way order
        order_reason = get_right_of_way_order_reason(result_order)
        
        # print(f'result_order: {result_order}')
        # Generate question text
        descriptions = [
            f"The {color_mapping[vehicle['color']]} car will {'go straight' if vehicle['direction'] == 'straight' else 'turn ' + vehicle['direction']}."
            for vehicle in vehicle_infos.values()
        ]
        question_text = f"Question {question_id+1}: Assuming several cars arrive at an uncontrolled T-intersection simultaneously. " + " ".join(descriptions) + " In which order should they proceed?"

        option_labels = ['A', 'B', 'C', 'D']

        # print(f'result_order: {result_order}')
        # print(f'vehicle_infos: {vehicle_infos}')
        if result_order is None:
            possible_orders = [
                " -> ".join([f"{color_mapping[vehicle['color']]} car" for vehicle in order])
                for order in permutations(vehicle_infos.values())
            ]
            # print(f'possible_orders: {possible_orders}')
            correct_answer_text = "No one needs to yield because there is no conflict"

            possible_orders = possible_orders[:min(3, len(possible_orders))] + [correct_answer_text]

            if len(possible_orders) < 4:
                possible_orders.append('None of the above.')

            correct_index = possible_orders.index(correct_answer_text)
            options = [f"{option_labels[i]}. {possible_orders[i]}" for i in range(len(possible_orders))]
            # answer = f"Correct Answer: {option_labels[correct_index]}. {correct_answer_text}."
            answer = f"Correct Answer: {option_labels[correct_index]}. {correct_answer_text}. {order_reason}"
        else:
            correct_order = " -> ".join(
                " and ".join(f"{color_mapping[vehicle['color']]} car" for vehicle in group) for group in result_order
            )
            
            unique_orders = set()
            if 'and' in correct_order:
                # Flatten the result_order to get all vehicles
                all_vehicles = [vehicle for group in result_order for vehicle in group]
                # Generate all permutations of the vehicles
                all_vehicle_permutations = list(permutations(all_vehicles))

                for perm in all_vehicle_permutations:
                    groupings = [
                        [ [perm[0]], list(perm[1:]) ],
                        [ list(perm[:-1]), [perm[-1]] ]
                    ]
                    for grouping in groupings:
                        # Generate the order text
                        order_text = " -> ".join(
                            " and ".join(f"{color_mapping[vehicle['color']]} car" for vehicle in group)
                            for group in grouping
                        )
                        unique_orders.add(order_text)
                        unique_orders = remove_duplicate_orders(unique_orders)
                        unique_orders = replace_or_add_order(correct_order, unique_orders)
            else:
                for perm in permutations(result_order):
                    for subgroup_permutation in product(*[permutations(group) for group in perm]):
                        order_text = " -> ".join(
                            " and ".join(sorted(f"{color_mapping[vehicle['color']]} car" for vehicle in subgroup))
                            for subgroup in subgroup_permutation
                        )
                        unique_orders.add(order_text)

            unique_orders = list(unique_orders)
            random.shuffle(unique_orders)
            unique_orders = unique_orders[:4]  # 4 options

            # Ensure correct_order is in the options
            if correct_order not in unique_orders:
                unique_orders[random.randint(0, len(unique_orders) - 1)] = correct_order

            if len(unique_orders) < 4:
                unique_orders.append('No one needs to yield because there is no conflict.')
            if len(unique_orders) < 4:
                unique_orders.append('None of the above.')

            options = [f"{option_labels[i]}. {unique_orders[i]}" for i in range(len(unique_orders))]

            # Identify the Correct Answer index
            correct_index = unique_orders.index(correct_order)
            # answer = f"Correct Answer: {option_labels[correct_index]}. {correct_order}"
            answer = f"Correct Answer: {option_labels[correct_index]}. {correct_order}. {order_reason}"

        # Combine question, options, and answer
        formatted_options = "\n".join(options)
        full_question = f"{question_text}\n{formatted_options}\n\n{answer}\n"

        questions.append(full_question)

    return questions


def determine_right_of_way_order(vehicle_infos):
    """
    Determine the order in which vehicles should proceed based on right-of-way rules.
    Handles cases where vehicles can proceed simultaneously if there is no conflict.
    """
    vehicles = list(vehicle_infos.values())
    num_vehicles = len(vehicles)
    conflicting_vehicles = []
    non_conflicting_pairs = {i: [] for i in range(num_vehicles)}

    # Check conflicts among vehicles
    for i in range(num_vehicles):
        for j in range(i + 1, num_vehicles):
            if is_conflicting_at_T_intersection(vehicles[i], vehicles[j]):
                conflicting_vehicles.append((vehicles[i], vehicles[j]))
            else:
                non_conflicting_pairs[i].append(j)
                non_conflicting_pairs[j].append(i)

    # Separate vehicles into conflicting and non-conflicting groups
    all_conflicting = []
    for pair in conflicting_vehicles:
        for vehicle in pair:
            if vehicle not in all_conflicting:
                all_conflicting.append(vehicle)

    # If all vehicles are non-conflicting, return None to indicate no specific order is required
    if not all_conflicting:
        return None

    # Order vehicles by priority
    ordered_conflicting = []
    
    # First priority: Main road straight
    main_straight = [v for v in all_conflicting if v['road_type'] in ['main_left', 'main_right'] and v['direction'] == 'straight']
    if main_straight:
        ordered_conflicting.extend(sorted(main_straight, key=lambda v: get_road_type_priority(v['road_type'])))

    # Then add remaining vehicles by standard priority
    right_turn_vehicles = [v for v in all_conflicting if v['direction'] == 'right' and v not in ordered_conflicting]
    straight_vehicles = [v for v in all_conflicting if v['direction'] == 'straight' and v not in ordered_conflicting]
    left_turn_vehicles = [v for v in all_conflicting if v['direction'] == 'left']

    if right_turn_vehicles:
        ordered_conflicting.extend(sorted(right_turn_vehicles, key=lambda v: get_road_type_priority(v['road_type'])))
    if straight_vehicles:
        ordered_conflicting.extend(sorted(straight_vehicles, key=lambda v: get_road_type_priority(v['road_type'])))
    if left_turn_vehicles:
        ordered_conflicting.extend(sorted(left_turn_vehicles, key=lambda v: get_road_type_priority(v['road_type'])))

    # Rest of the function remains the same
    result_order = []
    if ordered_conflicting:
        first_vehicle_index = vehicles.index(ordered_conflicting[0])
        non_conflicting = []

        # Check if the index of the first vehicle is in non_conflicting_pairs
        if first_vehicle_index in non_conflicting_pairs:
            non_conflicting_indices = non_conflicting_pairs[first_vehicle_index]
            non_conflicting = [vehicles[first_vehicle_index]] + [vehicles[idx] for idx in non_conflicting_indices]

        # If a non-conflicting pair is found, add it to result_order
        if non_conflicting:
            result_order.append(non_conflicting)
            # Remove both the first_vehicle and its pair from ordered_conflicting
            ordered_conflicting = [v for v in ordered_conflicting if v not in non_conflicting]
        else:
            # If no pair is found, treat it as a single vehicle group
            result_order.append([ordered_conflicting[0]])
            # Remove only the first vehicle from ordered_conflicting
            ordered_conflicting = ordered_conflicting[1:]

    # Add the ordered conflicting vehicles to the result order
    for vehicle in ordered_conflicting:
        result_order.append([vehicle])

    return result_order


def get_road_type_priority(road_type):
    """
    Returns a priority value for the road type, where lower values indicate higher priority.
    """
    road_type_priority = {'main_right': 1, 'stem': 2, 'main_left': 3}
    return road_type_priority.get(road_type, 4)  # Default to 4 if road_type is unknown


def spawn_vehicles_near_intersection_v3(client, world, preturn_waypoints, traffic_manager, car_lights_on=False):
    def select_vehicle_blueprint(world, target_colors):
        """Selects a vehicle blueprint with the specified colors."""
        vehicles_bp = world.get_blueprint_library().filter('vehicle.*')
        vehicles_bp_4 = [bp for bp in vehicles_bp if int(bp.get_attribute('number_of_wheels')) == 4]
        
        for bp in vehicles_bp_4:
            supported_colors = bp.get_attribute('color').recommended_values if bp.has_attribute('color') else []
            if all(color in supported_colors for color in target_colors):
                # print(f"Selected vehicle blueprint: {bp.id}")
                return bp
        print("No vehicle blueprint found with all specified colors.")
        return None

    global successful_vehicle_info_by_junction
    successful_vehicle_info_by_junction.clear()
    spawn_points = world.get_map().get_spawn_points()
    
    vehicle_ids = []
    used_colors_by_junction = defaultdict(set)
    waypoints_grouped_by_junction = defaultdict(list)
    questions = defaultdict(list)

    # target_colors = ['42,61,214', '38,38,38', '79,33,85', '155,0,0', '255,250,122', '0,0,0']  # Blue, Gray, Purple, Red, Yellow, Black
    target_colors = ['42,61,214', '79,33,85', '155,0,0', '255,250,122']  # Blue, Purple, Red, Yellow

    target_vehicle_bp = select_vehicle_blueprint(world, target_colors)
    if not target_vehicle_bp:
        return []

    for junction_id, waypoint in preturn_waypoints:
        waypoints_grouped_by_junction[junction_id].append(waypoint)
        # print(f'junction_id: {junction_id}, waypoint: {waypoint}')

    for junction_id, waypoints in waypoints_grouped_by_junction.items():
        batch = []
        vehicle_info_list = []
        max_spawn = 3
        # max_spawn = len(waypoints)
        vehicle_count = random.randint(2, min(max_spawn, len(waypoints)))
        selected_waypoints = random.sample(waypoints, vehicle_count)

        used_yaws = set()
        # Spawn vehicles
        for waypoint in selected_waypoints:
            transform = waypoint.transform
            transform.location.z = spawn_points[0].location.z
            
            if transform.rotation.yaw in used_yaws:
                continue  # Skip this waypoint if yaw is already used
            used_yaws.add(transform.rotation.yaw)

            # Choose a unique color from target_colors
            available_colors = [color for color in target_colors if color not in used_colors_by_junction[junction_id]]
            color = random.choice(available_colors)
            used_colors_by_junction[junction_id].add(color)

            # print(f'available_colors: {available_colors}, used_colors_by_junction: {used_colors_by_junction[junction_id]}')
            # print(f'color: {color}')

            target_vehicle_bp.set_attribute('color', color)
            target_vehicle_bp.set_attribute('role_name', 'autopilot')

            directions = assign_vehicle_directions_based_on_waypoints(waypoint, distance=25)

            vehicle_info_list.append({
                'junction_id': junction_id,
                'color': color, 
                'directions': directions,
                'location': transform.location, 
                'yaw': transform.rotation.yaw,
                'vehicle_id': target_vehicle_bp.id,
            })

            batch.append(
                carla.command.SpawnActor(target_vehicle_bp, transform)
                .then(carla.command.SetAutopilot(carla.command.FutureActor, False, traffic_manager.get_port()))
            )

        successful_vehicle_info_list = []

        for response, vehicle_info in zip(client.apply_batch_sync(batch, True), vehicle_info_list):
            if response.error:
                # print(f"Error spawning vehicle: {response.error}")
                pass
            else:
                vehicle_actor = world.get_actor(response.actor_id)
                if car_lights_on:
                    light_state = carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position
                    vehicle_actor.set_light_state(carla.VehicleLightState(light_state))
                vehicle_ids.append((junction_id, response.actor_id)) 
                successful_vehicle_info_list.append(vehicle_info)

        successful_vehicle_info_by_junction[junction_id] = successful_vehicle_info_list

        # print(f'vehicle_info_list: {vehicle_info_list}')
        # Generate questions
        questions_list1 = create_direct_questions(successful_vehicle_info_list)
        questions[junction_id].extend(questions_list1)
        questions_list2 = create_order_questions(successful_vehicle_info_list)
        questions[junction_id].extend(questions_list2)

    return vehicle_ids, questions



def add_top_camera_sensor(world, waypoint, distance, z_offset, pitch_offset, yaw_offset=0):
    # adjust camera location. backward: distance < 0, forward: distance > 0
    yaw_rad = math.radians(waypoint.transform.rotation.yaw)
    
    dx = -distance * math.cos(yaw_rad)
    dy = -distance * math.sin(yaw_rad)

    adjusted_location = carla.Location(
        x=waypoint.transform.location.x + dx,
        y=waypoint.transform.location.y + dy,
        z=waypoint.transform.location.z + z_offset + random.uniform(-5, 5)
    )
    
    adjusted_rotation = carla.Rotation(
        pitch = pitch_offset + random.uniform(-2, 2),
        yaw=waypoint.transform.rotation.yaw + yaw_offset + random.uniform(-5, 5)
        )
    
    camera = add_camera_sensor(world, adjusted_location, adjusted_rotation)
    return camera


def add_front_camera_sensor(world, vehicle, height=1.4, pitch=-1):
    """
    Attach a camera to a vehicle to create a first-person view.
    """
    # Get the vehicle's yaw to determine relative position
    vehicle_transform = vehicle.get_transform()
    # print(vehicle_transform)

    relative_location = carla.Location(
        x=vehicle_transform.location.x + 0.7 * math.cos(math.radians(vehicle_transform.rotation.yaw)),
        y=vehicle_transform.location.y + 0.7 * math.sin(math.radians(vehicle_transform.rotation.yaw)),
        z=vehicle_transform.location.z + height
    )

    # Define camera location relative to vehicle position
    # relative_location = carla.Location(
    #     x=vehicle_transform.location.x,
    #     y=vehicle_transform.location.y,
    #     z=vehicle_transform.location.z + height
    # )

    # Define camera rotation with specified pitch
    relative_rotation = carla.Rotation(
        pitch=pitch,
        yaw=vehicle_transform.rotation.yaw - 3
    )
    
    # Use the base function to create and attach the camera to the vehicle
    camera = add_camera_sensor(world, relative_location, relative_rotation, fov='130', attach_to=vehicle)
    
    return camera


def get_right_of_way_reason(vehicle_infos, right_of_way_vehicle):
    """Generate detailed reason for why a vehicle has right-of-way at a T-intersection."""
    if not right_of_way_vehicle:
        return "There is no conflict between any of the vehicles at this intersection. All vehicles can proceed simultaneously because their intended paths do not intersect or overlap at any point."
    
    vehicles = list(vehicle_infos.values())
    conflicting_pairs = []
    
    # Identify all conflicting pairs
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            if is_conflicting_at_T_intersection(vehicles[i], vehicles[j]):
                conflicting_pairs.append((vehicles[i], vehicles[j]))
    
    # Generate conflict description
    conflict_desc = "The following vehicles have conflicting paths: "
    conflict_parts = []
    for v1, v2 in conflicting_pairs:
        conflict_parts.append(f"the {color_mapping[v1['color']]} car conflicts with the {color_mapping[v2['color']]} car")
    
    if conflict_parts:
        conflict_desc += ", and ".join(conflict_parts) + "."
    else:
        conflict_desc = "There are no direct conflicts between vehicles."

    # Build a comprehensive explanation
    explanation_parts = []

    # Check if vehicle has right-of-way due to being on main road going straight
    if right_of_way_vehicle['road_type'] in ['main_left', 'main_right'] and right_of_way_vehicle['direction'] == 'straight':
        stem_vehicles = [v for v in vehicles if v['road_type'] == 'stem' and v != right_of_way_vehicle]
        if stem_vehicles:
            stem_desc = ", ".join([color_mapping[v['color']] for v in stem_vehicles])
            explanation_parts.append(f"The {color_mapping[right_of_way_vehicle['color']]} car going straight on the main road has priority over the {stem_desc} car(s) on the stem road")

    # Check straight vs left priority
    if right_of_way_vehicle['direction'] == 'straight':
        left_vehicles = [v for v in vehicles if v['direction'] == 'left' and v != right_of_way_vehicle]
        if left_vehicles:
            left_desc = ", ".join([color_mapping[v['color']] for v in left_vehicles])
            explanation_parts.append(f"The {color_mapping[right_of_way_vehicle['color']]} car going straight has priority over the {left_desc} car(s) turning left because going straight have priority over turning left")

    # Check right turn vs left turn priority
    elif right_of_way_vehicle['direction'] == 'right':
        left_vehicles = [v for v in vehicles if v['direction'] == 'left' and v != right_of_way_vehicle]
        if left_vehicles:
            left_desc = ", ".join([color_mapping[v['color']] for v in left_vehicles])
            explanation_parts.append(f"The {color_mapping[right_of_way_vehicle['color']]} car turning right has priority over the {left_desc} car(s) turning left because turning right have priority over turning left")

    # Combine all explanations
    if explanation_parts:
        return f"{conflict_desc} {'; '.join(explanation_parts)}."
    
    # Default reason with more detail
    return f"{conflict_desc} The {color_mapping[right_of_way_vehicle['color']]} car has the right-of-way based on the standard right-of-way rules at an uncontrolled T-intersection: main road traffic has priority over stem road, then turning right over turning left and going straight over turning left"

def get_right_of_way_order_reason(result_order):
    """Generate detailed reason for the order in which vehicles should proceed at a T-intersection."""
    if result_order is None:
        return "There is no conflict between any of the vehicles at this intersection. All vehicles can proceed simultaneously because their intended paths do not intersect or overlap at any point."
    
    if len(result_order) == 1:
        vehicle_colors = [color_mapping[v['color']] for v in result_order[0]]
        return f"These vehicles ({', '.join(vehicle_colors)}) don't have conflicting paths, so they can all proceed simultaneously without risk of collision."
    
    reasons = []
    
    # First group reason with comprehensive explanation
    first_group = result_order[0]
    if len(first_group) > 1:
        first_vehicle = first_group[0]
        other_vehicles = first_group[1:]
        
        priority_reasons = []
        # Check main road priority
        if first_vehicle['road_type'] in ['main_left', 'main_right'] and first_vehicle['direction'] == 'straight':
            if any(v['road_type'] == 'stem' for group in result_order[1:] for v in (group if isinstance(group, list) else [group])):
                priority_reasons.append("main road straight traffic has priority over stem road")
        
        # Check movement priority
        if first_vehicle['direction'] == 'right':
            left_turning = any(v['direction'] == 'left' for group in result_order[1:] for v in (group if isinstance(group, list) else [group]))
            if left_turning:
                priority_reasons.append("turning right have priority over turning left")
        elif first_vehicle['direction'] == 'straight':
            left_turning = any(v['direction'] == 'left' for group in result_order[1:] for v in (group if isinstance(group, list) else [group]))
            if left_turning:
                priority_reasons.append("going straight have priority over turning left")
        
        priority_explanation = " and ".join(priority_reasons) if priority_reasons else "T-intersection priority rules (first main road over stem road, then right turn over left turn and straight over left turn)"
        reasons.append(f"The {color_mapping[first_vehicle['color']]} car has the right-of-way because {priority_explanation}. "
                      f"The {', '.join([color_mapping[v['color']] for v in other_vehicles])} car(s) can proceed simultaneously with the {color_mapping[first_vehicle['color']]} car "
                      f"as they don't have conflicting paths.")
    else:
        first_vehicle = first_group[0]
        remaining_vehicles = [v for group in result_order[1:] for v in (group if isinstance(group, list) else [group])]
        
        priority_reasons = []
        # Check main road priority
        if first_vehicle['road_type'] in ['main_left', 'main_right'] and first_vehicle['direction'] == 'straight':
            if any(v['road_type'] == 'stem' for v in remaining_vehicles):
                priority_reasons.append("main road straight traffic has priority over stem road")
        
        # Check movement priority
        if first_vehicle['direction'] == 'right':
            if any(v['direction'] == 'left' for v in remaining_vehicles):
                priority_reasons.append("turning right have priority over turning left")
        elif first_vehicle['direction'] == 'straight':
            if any(v['direction'] == 'left' for v in remaining_vehicles):
                priority_reasons.append("going straight have priority over turning left")
        
        priority_explanation = " and ".join(priority_reasons) if priority_reasons else "T-intersection priority rules (first main road over stem road, then right turn over left turn and straight over left turn)"
        reasons.append(f"The {color_mapping[first_vehicle['color']]} car should proceed first because {priority_explanation}.")

    # Middle groups with detailed explanations
    for i in range(1, len(result_order) - 1):
        group = result_order[i]
        remaining_vehicles = [v for g in result_order[i+1:] for v in (g if isinstance(g, list) else [g])]
        
        if len(group) > 1:
            vehicle_colors = [color_mapping[v['color']] for v in group]
            reasons.append(f"After the {color_mapping[result_order[i-1][0]['color']]} car{'s' if len(result_order[i-1]) > 1 else ''} "
                         f"clear{'s' if len(result_order[i-1]) == 1 else ''} the intersection, "
                         f"the group of {', '.join(vehicle_colors)} cars can proceed simultaneously as they don't have "
                         f"conflicting paths with each other.")
        else:
            vehicle = group[0]
            priority_reasons = []
            
            if vehicle['direction'] == 'right':
                if any(v['direction'] == 'left' for v in remaining_vehicles):
                    priority_reasons.append("turning right have priority over turning left")
            elif vehicle['direction'] == 'straight':
                if any(v['direction'] == 'left' for v in remaining_vehicles):
                    priority_reasons.append("going straight have priority over turning left")
            
            priority_explanation = " and ".join(priority_reasons) if priority_reasons else "T-intersection priority rules (first main road over stem road, then right turn over left turn and straight over left turn)"
            reasons.append(f"The {color_mapping[vehicle['color']]} car should proceed next because {priority_explanation}.")

    # Last group with specific explanation
    if len(result_order) > 1:
        last_group = result_order[-1]
        if len(last_group) > 1:
            vehicle_colors = [color_mapping[v['color']] for v in last_group]
            reasons.append(f"Finally, the {', '.join(vehicle_colors)} cars can proceed simultaneously after all other vehicles "
                         f"have cleared the intersection, as they don't have conflicting paths with each other.")
        else:
            vehicle = last_group[0]
            if vehicle['direction'] == 'left':
                reasons.append(f"The {color_mapping[vehicle['color']]} car must proceed last because left turns have the lowest "
                             f"priority at an uncontrolled T-intersection, yielding to both straight-moving traffic and right turn.")
            else:
                reasons.append(f"The {color_mapping[vehicle['color']]} car proceeds last as it had the lowest priority based on "
                             f"the intersection rules: main road over stem road, then turning right over turning left, and straight over turning left.")
    
    return " ".join(reasons)


if __name__ == '__main__':
    camera_views = ['top', 'front']
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    # available_maps = client.get_available_maps()
    available_maps = ['Town01', 'Town02', 'Town04', 'Town05', 'Town07', 'Town10HD']
    # available_maps = ['Town10HD']

    T_intersection_id = {
        'Town03': [655, 1038, 1352, 1441, 1696, 1794],
        'Town04': [278, 1008, 1103, 1197],
        'Town05': [359, 1882, 2014, 2240],
        'Town07': [123, 146, 260, 288, 355, 402, 430, 466, 543, 566, 879, 907],
        'Town10HD': [23, 719]
    }
    weathers = list(WEATHER.keys())
    # random.shuffle(weathers)

    for map_name in sorted(available_maps):
        map_name = os.path.basename(map_name)
        try:
            world = client.load_world(map_name)
            print(f'------------ Map: {map_name} ------------')
            time.sleep(2)
        except Exception as e:
            print(e)
            print(f'Fail to load {map_name}')
        world = client.get_world()

        weather_count = 0
        # while weather_count < min(8, len(weathers)):
        while weather_count < len(weathers):
            weather = weathers[weather_count]
            world.set_weather(WEATHER[weather])
            weather_count += 1
            time.sleep(0.5)

            if map_name == 'Town10HD':
                min_distance = 2.0
            else:
                min_distance = 2.0
            # intersections = find_intersections(world)  # location of intersection center
            preturn_waypoints = find_pre_turn_waypoints(world, min_distance=min_distance, max_distance=5.0, step_size=0.5)
            preturn_waypoints.sort(key=lambda x: x[0])  # sort by junction id

            # filter duplicates
            unique_preturn_waypoints = []
            seen = set()
            for junction_id, waypoint in preturn_waypoints:
                if map_name == 'Town01' and junction_id == 222:
                    continue
                if map_name in T_intersection_id.keys():
                    if junction_id not in T_intersection_id[map_name]:
                        continue
                waypoint_id = (
                    junction_id,
                    waypoint.transform.location.x,
                    waypoint.transform.location.y,
                    waypoint.transform.location.z,
                    waypoint.transform.rotation.pitch,
                    waypoint.transform.rotation.yaw,
                    waypoint.transform.rotation.roll,
                )
                if waypoint_id not in seen:
                    # print(waypoint_id)
                    seen.add(waypoint_id)
                    unique_preturn_waypoints.append((junction_id, waypoint))
            # print(unique_preturn_waypoints)
            preturn_waypoints = unique_preturn_waypoints
            # print(f"Number of pre-turn locations: {len(preturn_waypoints)}")

            traffic_manager = get_traffic_manager(client, port=4500)
            synchronous_master = True

            car_lights_on = False
            if 'night' in weather.lower():
                car_lights_on = True

            successful_vehicle_info_by_junction = {}
            vehicle_ids, questions = spawn_vehicles_near_intersection_v3(client, world, preturn_waypoints, traffic_manager, car_lights_on=car_lights_on)

            if len(vehicle_ids) == 0:
                break

            # Loop through camera views
            for camera_view in camera_views:
                print(f"Collecting {camera_view} view data...")
                
                output_image_dir = f"T_intersection/seed{seed}/{camera_view}_view/{map_name}/images/"
                output_question_dir = f"T_intersection/seed{seed}/{camera_view}_view/{map_name}/questions/"
                output_metadata_dir = f"T_intersection/seed{seed}/{camera_view}_view/{map_name}/metadata/"

                os.makedirs(output_image_dir, exist_ok=True)
                os.makedirs(output_question_dir, exist_ok=True)
                os.makedirs(output_metadata_dir, exist_ok=True)
                
                image_count = 0

                if camera_view == 'top':
                    for junction_id, waypoint in preturn_waypoints:
                        if (map_name == 'Town02' and junction_id == 20) or (map_name == 'Town01' and junction_id == 87):
                            camera = add_top_camera_sensor(world, waypoint, distance=-10, z_offset=35, pitch_offset=-87, yaw_offset=0)
                        elif map_name == 'Town10HD':
                            camera = add_top_camera_sensor(world, waypoint, distance=8, z_offset=30, pitch_offset=-50, yaw_offset=0)
                        else:
                            camera = add_top_camera_sensor(world, waypoint, distance=8, z_offset=25, pitch_offset=-52, yaw_offset=0)

                        def save_image_callback(image):
                            global image_count
                            global successful_vehicle_info_by_junction
                            image_name = f"{output_image_dir}/seed{seed}_{map_name}_{weather}_intersection{junction_id}_{camera_view}view_{image_count:04d}.jpg"
                            save_image(image, image_name)
                            print(f"Saved image {image_name}")
                            # Capture camera parameters
                            camera_transform = camera.get_transform()
                            camera_params = {
                                "location": {
                                    "x": camera_transform.location.x,
                                    "y": camera_transform.location.y,
                                    "z": camera_transform.location.z
                                },
                                "rotation": {
                                    "pitch": camera_transform.rotation.pitch,
                                    "yaw": camera_transform.rotation.yaw,
                                    "roll": camera_transform.rotation.roll
                                }
                            }
                            
                            vehicle_infos = [
                                {
                                    "vehicle_id": vehicle['vehicle_id'],
                                    "color": vehicle['color'],
                                    "location": {
                                        "x": vehicle['location'].x,
                                        "y": vehicle['location'].y,
                                        "z": vehicle['location'].z
                                    },
                                    "yaw": vehicle['yaw']
                                }
                                for vehicle in successful_vehicle_info_by_junction.get(junction_id, [])
                            ]
                            
                            # Extract full weather parameters
                            weather_obj = world.get_weather()
                            weather_params = {
                                "cloudiness": weather_obj.cloudiness,
                                "precipitation": weather_obj.precipitation,
                                "precipitation_deposits": weather_obj.precipitation_deposits,
                                "wind_intensity": weather_obj.wind_intensity,
                                "sun_azimuth_angle": weather_obj.sun_azimuth_angle,
                                "sun_altitude_angle": weather_obj.sun_altitude_angle,
                                "fog_density": weather_obj.fog_density,
                                "fog_distance": weather_obj.fog_distance,
                                "fog_falloff": weather_obj.fog_falloff,
                                "wetness": weather_obj.wetness,
                            }
                            # Save metadata with town name and weather
                            metadata_filename = f"{output_metadata_dir}/seed{seed}_{map_name}_{weather}_intersection{junction_id}_{camera_view}view_{image_count:04d}.json"
                            save_metadata(metadata_filename, camera_params, vehicle_infos, map_name, weather_params)
                            print(f"Saved metadata {metadata_filename}")
                            camera.stop()

                        camera.listen(save_image_callback)

                        world.tick()
                        time.sleep(0.3)

                        txt_name = f"{output_question_dir}/seed{seed}_{map_name}_{weather}_intersection{junction_id}_{camera_view}view_{image_count:04d}.txt"
                        with open(txt_name, 'w') as f:
                            for question in questions[junction_id]:
                                if camera_view == 'front':
                                    question = question.replace(f'The {color_mapping[vehicle_color]} car', 'The ego car').replace(f'{color_mapping[vehicle_color]} car', 'The ego car')
                                f.write(question + "\n")

                        camera.destroy()

                        image_count += 1
                elif camera_view == 'front':
                    print(f"vehicle_ids: {vehicle_ids}")
                    for junction_id, vehicle_id in vehicle_ids:
                        # Add front camera sensor
                        vehicle = world.get_actor(vehicle_id)
                        camera = add_front_camera_sensor(world, vehicle)
                        vehicle_color = vehicle.attributes.get('color', None)

                        def save_image_callback(image):
                            global image_count
                            global successful_vehicle_info_by_junction
                            image_name = f"{output_image_dir}/seed{seed}_{map_name}_{weather}_intersection{junction_id}_{camera_view}view_{image_count:04d}.jpg"
                            save_image(image, image_name)
                            print(f"Saved image {image_name}")
                            # Capture camera parameters
                            camera_transform = camera.get_transform()
                            camera_params = {
                                "location": {
                                    "x": camera_transform.location.x,
                                    "y": camera_transform.location.y,
                                    "z": camera_transform.location.z
                                },
                                "rotation": {
                                    "pitch": camera_transform.rotation.pitch,
                                    "yaw": camera_transform.rotation.yaw,
                                    "roll": camera_transform.rotation.roll
                                }
                            }
                            
                            vehicle_infos = [
                                {
                                    "vehicle_id": vehicle['vehicle_id'],
                                    "color": vehicle['color'],
                                    "location": {
                                        "x": vehicle['location'].x,
                                        "y": vehicle['location'].y,
                                        "z": vehicle['location'].z
                                    },
                                    "yaw": vehicle['yaw']
                                }
                                for vehicle in successful_vehicle_info_by_junction.get(junction_id, [])
                            ]
                            
                            # Extract full weather parameters
                            weather_obj = world.get_weather()
                            weather_params = {
                                "cloudiness": weather_obj.cloudiness,
                                "precipitation": weather_obj.precipitation,
                                "precipitation_deposits": weather_obj.precipitation_deposits,
                                "wind_intensity": weather_obj.wind_intensity,
                                "sun_azimuth_angle": weather_obj.sun_azimuth_angle,
                                "sun_altitude_angle": weather_obj.sun_altitude_angle,
                                "fog_density": weather_obj.fog_density,
                                "fog_distance": weather_obj.fog_distance,
                                "fog_falloff": weather_obj.fog_falloff,
                                "wetness": weather_obj.wetness,
                            }
                            # Save metadata with town name and weather
                            metadata_filename = f"{output_metadata_dir}/seed{seed}_{map_name}_{weather}_intersection{junction_id}_{camera_view}view_{image_count:04d}.json"
                            save_metadata(metadata_filename, camera_params, vehicle_infos, map_name, weather_params)
                            print(f"Saved metadata {metadata_filename}")
                            camera.stop()

                        camera.listen(save_image_callback)

                        world.tick()
                        time.sleep(0.3)

                        txt_name = f"{output_question_dir}/seed{seed}_{map_name}_{weather}_intersection{junction_id}_{camera_view}view_{image_count:04d}.txt"
                        with open(txt_name, 'w') as f:
                            for question in questions[junction_id]:
                                if camera_view == 'front':
                                    question = question.replace(f'The {color_mapping[vehicle_color]} car', 'The ego car').replace(f'{color_mapping[vehicle_color]} car', 'The ego car')
                                f.write(question + "\n")

                        camera.destroy()

                        image_count += 1

            for _, vehicle_id in vehicle_ids:
                # print(f"Destroying vehicle {vehicle_id}")
                world.get_actor(vehicle_id).destroy()

        print(f"Images captured and saved for all intersections for {map_name}.")
