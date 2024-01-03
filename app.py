import streamlit as st
import pandas as pd
from typing import List, Tuple, Dict, Callable
from copy import deepcopy
import pandas as pd
import numpy as np


MOVES = [(0,-1), (1,0), (0,1), (-1,0)]

COSTS = { '🌾': 1, '🌲': 3, '🪨': 5, '🐊': 7}

small_world = [
['🌾', '🪨', '🌲', '🌲', '🌲', '🌲', '🌲'],
['🌾', '🪨', '🪨', '🗻', '🌲', '🌲', '🌲'],
['🌾', '🌲', '🌲', '🗻', '🌲', '🐊', '🗻'],
['🌾', '🪨', '🌾', '🌾', '🐊', '🌾', '🗻'],
['🌲', '🌲', '🌲', '🌲', '🐊', '🌲', '🌾'],
['🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
['🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾']
]

big_world = [
['🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
['🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌲', '🌲'],
['🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🌲', '🌲', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌲'],
['🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🐊'],
['🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻'],
['🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻'],
['🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🌾'],
['🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🪨', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾'],
['🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🌾', '🐊', '🐊', '🐊'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🌲', '🌲', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🪨', '🪨', '🪨', '🪨', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🪨']
]

def heuristic(position, goal):
    """
    Heuristic function that estimates the distance between two positions using the Manhattan distance.

    Args:
        position (Tuple[int, int]): The current position (x, y).
        goal (Tuple[int, int]): The goal position (x, y).

    Returns:
        int: The estimated distance between the two positions.
    """
    x1, y1 = position
    x2, y2 = goal
    return abs(x2 - x1) + abs(y2 - y1)

def get_neighbors(position, world, moves):
    """
    Retrieves the valid neighboring positions from a given position in the world.

    Args:
        position (Tuple[int, int]): The current position (x, y).
        world (List[List[str]]): The world map represented as a 2D list of strings.
        moves (List[Tuple[int, int]]): The valid moves that can be made from a position.

    Returns:
        List[Tuple[int, int]]: The neighboring positions.
    """
    x, y = position
    neighbors = [(x + dx, y + dy) for dx, dy in moves]
    neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < len(world) and 0 <= ny < len(world[0]) and world[nx][ny] != '🗻']
    return neighbors

def calculate_cost(parent_cost, position, neighbor, world, costs):
    """
    Calculates the cost to move from a parent position to a neighbor position.

    Args:
        parent_cost (int): The cost to reach the parent position.
        position (Tuple[int, int]): The current position (x, y).
        neighbor (Tuple[int, int]): The neighbor position (x, y).
        world (List[List[str]]): The world map represented as a 2D list of strings.
        costs (Dict[str, int]): The costs associated with different types of cells in the world.

    Returns:
        int: The total cost to move from the parent position to the neighbor position.
    """
    x1, y1 = position
    x2, y2 = neighbor
    terrain_cost = costs[world[x2][y2]]
    move_cost = abs(x2 - x1) + abs(y2 - y1)
    return parent_cost + terrain_cost + move_cost



def a_star_search( world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    """
    Performs A* search algorithm to find the shortest path from start to goal in the given world.

    Args:
        world (List[List[str]]): The world map represented as a 2D list of strings.
        start (Tuple[int, int]): The starting position (row, column).
        goal (Tuple[int, int]): The goal position (row, column).
        costs (Dict[str, int]): The costs associated with different types of cells in the world.
        moves (List[Tuple[int, int]]): The valid moves that can be made from a cell (up, down, left, right).
        heuristic (Callable): The heuristic function to estimate the distance from a cell to the goal.

    Returns:
        List[Tuple[int, int]]: The path from start to goal as a list of positions (row, column).
    """
    #initilize data structures
    frontier = [(start, 0)]
    explored = set()
    g = {start: 0}
    f = {start: heuristic(start, goal)}
    parents = {}
    
    while frontier:
        current, current_cost = min(frontier, key=lambda x: x[1] + f[x[0]])
        
        if current == goal:
            path = []
            while current in parents:
                path.insert(0, current)
                current = parents[current]
            path.insert(0, start)
            return path
        
        frontier.remove((current, current_cost))
        explored.add(current)
        
        for neighbor in get_neighbors(current, world, moves):
            if neighbor in explored:
                continue
            
            cost = calculate_cost(current_cost, current, neighbor, world, costs)
            
            if neighbor not in frontier or cost < g[neighbor]:
                g[neighbor] = cost
                f[neighbor] = cost + heuristic(neighbor, goal)
                parents[neighbor] = current
                
                if neighbor not in frontier:
                    frontier.append((neighbor, cost))
    return [] 


def pretty_print_path(world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    """
    Prints the path, path cost, and full map with visual symbols.

    Args:
        world (List[List[str]]): The world map represented as a 2D list of strings.
        path (List[Tuple[int, int]]): The path from start to goal as a list of positions (row, column).
        start (Tuple[int, int]): The starting position (row, column).
        goal (Tuple[int, int]): The goal position (row, column).
        costs (Dict[str, int]): The costs associated with different types of cells in the world.

    Returns:
        int: The total cost of the path.
    """
    path_cost = 0
    map_data = []
    
    progress_bar = st.progress(0)  # Create a progress bar
    status_text = st.empty()

    for i in range(len(world)):
        row_data = []
        for j in range(len(world[0])):
            position = (i, j)

            if position == goal:
                row_data.append("🎁")
            elif position == start:
                row_data.append("🚀")  # Start position
            elif position in path:
                if position[0] < path[path.index(position) + 1][0]:
                    row_data.append("⏬")  # Move down
                elif position[0] > path[path.index(position) + 1][0]:
                    row_data.append("⏪")  # Move left
                elif position[1] < path[path.index(position) + 1][1]:
                    row_data.append("⏩")  # Move right
                elif position[1] > path[path.index(position) + 1][1]:
                    row_data.append("⏫")  # Move up
                path_cost += costs[world[i][j]]
            else:
                row_data.append(world[i][j])
        
        map_data.append(row_data)

        status_text.text(f"Progress: {int((i + 1) / len(world) * 100)}%")  # Update the progress tex
        progress = progress_bar.progress((i + 1) / len(world))
        
    path_df = pd.DataFrame(map_data)

    return path_cost, path_df


#App Parameters
st.title("A* Search Algorithm")
#st.set_page_config(layout="wide")
st.subheader("EN.605.645 Artificial Intelligence")
st.write("Frank Zhao")

## #start/goal inputs
# st.header("Start")
# # Input 1 (Set 1)
# start_set1 = st.number_input("Input 1 (Set 1)", value=0, step=1)
# # Input 2 (Set 1)
# start_set2 = st.number_input("Input 2 (Set 1)", value=0, step=1)

# # Set 2
# st.header("Goal")

# # Input 1 (Set 2)
# input_1_set2 = st.number_input("Input 1 (Set 2)", value=0, step=1)
# # Input 2 (Set 2)
# input_2_set2 = st.number_input("Input 2 (Set 2)", value=0, step=1)


#Search Map Section
st.header("Search Maps")
options = st.selectbox("Which Map Would You Like To Search?",["Small World","Big World"])


col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
with col1:
    view_world_button = st.button("Display Map")
with col2:
    search_world_button = st.button("Run Search")

start = (0, 0)

if view_world_button:
    if options == "Small World":
        st.write("Map:")
        st.table(small_world)
    elif options == "Big World":
        st.write("Map:")
        st.table(big_world)

if search_world_button:
    if options == "Small World":       
        goal = (len(small_world[0]) - 1, len(small_world) - 1)
        path = a_star_search(small_world, start, goal, COSTS, MOVES, heuristic=heuristic)
        path_cost, path_df = pretty_print_path(small_world, path, start, goal, COSTS)
        st.write("Path:")
        st.table(path_df)
        st.write("Path Cost:", path_cost)
    elif options == "Big World":
        goal = (len(big_world[0]) - 1, len(big_world) - 1)
        path = a_star_search(big_world, start, goal, COSTS, MOVES, heuristic=heuristic)
        path_cost, path_df = pretty_print_path(big_world, path, start, goal, COSTS)
        st.write("Path:")
        st.table(path_df)
        st.write("Path Cost:", path_cost)