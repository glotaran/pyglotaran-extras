from typing import List
from collections import deque

import networkx as nx
from networkx.readwrite.json_graph import cytoscape_data

from glotaran.model.model import Model
from glotaran.parameter.parameters import Parameters
from glotaran.model.item import fill_item


def round_and_convert(value_in_ps_inverse):
    value_in_ns_inverse = value_in_ps_inverse * 1e3
    return round(value_in_ns_inverse) if value_in_ns_inverse >= 1 else round(value_in_ns_inverse, 2)

def build_all_transitions(megacomplex_k_matrices, omitted_rate_constants):
    transitions = []
    idx = 1
    total_decay_rates = set()

    for system in megacomplex_k_matrices.values():
        for (state_from, state_to), param in system.matrix.items():
            if param.label not in omitted_rate_constants:
                rate_constant_value = round_and_convert(param.value)
                extra_edge_attribute = {'weight': rate_constant_value}
                if state_from != state_to:
                    transitions.append((state_to, state_from, extra_edge_attribute))
                elif (state_to, rate_constant_value) not in total_decay_rates:
                    transitions.append((state_to, f'GS{idx}', extra_edge_attribute))
                    total_decay_rates.add((state_to, rate_constant_value))
                    idx += 1
    return transitions

def get_filled_megacomplex_k_matrices(megacomplexes: List[str], model: Model, parameters: Parameters):
    k_matrices = {}
    for mc in megacomplexes:
        if mc not in model.megacomplex:
            raise ValueError(f"Megacomplex {mc} not found.")
        if model.megacomplex[mc].type != 'decay':
            continue
        filled_megacomplex = fill_item(model.megacomplex[mc], model, parameters)
        k_matrices[mc] = filled_megacomplex.get_k_matrix()
    return k_matrices

def apply_some_adjustments(graph):
    for node in graph:
        ground_state_neighbors = [neighbor for neighbor in graph[node] if 'GS' in neighbor]

        if len(ground_state_neighbors) > 1:
            total_rate_constant = 0
            first_neighbor = ground_state_neighbors[0]

            for neighbor in ground_state_neighbors[1:]:
                total_rate_constant += graph[node][neighbor]['weight']
                graph.remove_edge(node, neighbor)

            graph[node][first_neighbor]['weight'] += total_rate_constant

    return graph

def dump_cytpscape_json_data(transitions):
    graph = nx.DiGraph()
    graph.add_edges_from(transitions)
    graph = apply_some_adjustments(graph)
    return cytoscape_data(graph)

def is_directed_acyclic(graph):
    return nx.is_directed_acyclic_graph(graph)

def layout_directed_acyclic_graph(graph, visualization_options):
    topological_order = list(nx.topological_sort(graph))

    x_pos = 0
    y_pos = 0
    layer_width = {}
    node_positions = {}

    # Start positioning from the first node in topological order
    root_node = topological_order[0]
    update_position_in_directed_acyclic_graph(graph, root_node, x_pos, y_pos, node_positions, layer_width)

    # Adjust the width of nodes with multiple predecessors
    for node in topological_order:
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            pred_y_levels = [node_positions[p][1] for p in predecessors]
            pred_x_levels = [node_positions[p][0] for p in predecessors]
            if len(set(pred_y_levels)) == 1:  # All predecessors at the same horizontal level
                max_predecessor_x = max(pred_x_levels)
                new_pos = (max_predecessor_x, node_positions[predecessors[0]][1] - 1)
            elif len(set(pred_x_levels)) == 1:  # All predecessors at the same vertical level
                min_predecessor_y = min(pred_y_levels)
                new_pos = (node_positions[predecessors[0]][0] + 1, min_predecessor_y)
            else:
                max_predecessor_x = max(pred_x_levels)
                new_pos = (max_predecessor_x + 1, node_positions[node][1])
            shift_x = new_pos[0] - node_positions[node][0]
            shift_y = new_pos[1] - node_positions[node][1]
            # Shift the node and all its successors
            nodes_to_shift = [node]
            while nodes_to_shift:
                current_node = nodes_to_shift.pop()
                current_x, current_y = node_positions[current_node]
                node_positions[current_node] = (current_x + shift_x, current_y + shift_y)
                nodes_to_shift.extend(graph.successors(current_node))
    visualization_options.plot_graph_edge_connection_style = 'arc3'
    return graph, node_positions, visualization_options

# Function to update position recursively
def update_position_in_directed_acyclic_graph(graph, node, x, y, pos, layer_width):
    pos[node] = (x, y)
    successors = list(graph.successors(node))
    num_successors = len(successors)
    if num_successors == 1:
        update_position_in_directed_acyclic_graph(graph, successors[0], x, y - 1, pos, layer_width)
    elif num_successors > 1:
        for i, successor in enumerate(successors):
            if i == 0:
                update_position_in_directed_acyclic_graph(graph, successor, x + 1, y, pos, layer_width)
            else:
                update_position_in_directed_acyclic_graph(graph, successor, x, y - 1, pos, layer_width)
    layer_width[x] = max(layer_width.get(x, 0), y)

def layout_directed_cyclic_graph(graph, visualization_options):
    node_positions = {}
    degree_dict = dict(graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)

    # Center position for the starting node
    center = (0, 0)
    node_positions[sorted_nodes[0]] = center

    directions = [(1, 0), (0, 1), (0, -1), (-1, 0)]

    used_positions = set()
    used_positions.add(center)

    queue = deque([sorted_nodes[0]])

    corner_position = (10, 10)

    while queue:
        current_node = queue.popleft()
        current_pos = node_positions[current_node]
        neighbors = list(graph.neighbors(current_node)) + list(graph.predecessors(current_node))
        neighbor_pos_index = 0

        for neighbor in neighbors:
            if neighbor not in node_positions:
                attempts = 0
                while True:
                    direction = directions[neighbor_pos_index % 4]
                    new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                    neighbor_pos_index += 1
                    attempts += 1
                    if new_pos not in used_positions:
                        node_positions[neighbor] = new_pos
                        used_positions.add(new_pos)
                        queue.append(neighbor)
                        break
                    if attempts >= 4:  # If can't place in four attempts, place as isolated
                        node_positions[neighbor] = corner_position
                        used_positions.add(corner_position)
                        corner_position = (corner_position[0] + 1, corner_position[1] + 1)
                        break
    visualization_options.plot_graph_edge_connection_style = 'arc3,rad=0.1'
    visualization_options.plot_graph_node_size = 5000
    return graph, node_positions, visualization_options
