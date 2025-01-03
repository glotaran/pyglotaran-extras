from typing import Dict, List, Union, Optional
import json

from pydantic import BaseModel, ConfigDict
from glotaran.model.model import Model
from glotaran.parameter.parameters import Parameters

from .utils import get_filled_megacomplex_k_matrices, build_all_transitions, dump_cytpscape_json_data
from .widget import GraphWidget

class Node(BaseModel):
    alternate_name: Optional[str] = None
    width: Optional[int] = 80
    height: Optional[int] = 30

class VisualizationOptions(BaseModel):
    nodes: Dict[str, Node] = {}
    colour_node_mapping: Dict[str, List[str]] = {}
    omitted_rate_constants: List[str] = []

    model_config = ConfigDict(extra='allow')

def visualize_megacomplex(megacomplex: Union[str, List[str]], model: Model, parameter: Parameters, visualization_options: VisualizationOptions = VisualizationOptions()) -> GraphWidget:
    if isinstance(megacomplex, str):
        megacomplexes = [megacomplex]
    else:
        megacomplexes = megacomplex

    k_matrices = get_filled_megacomplex_k_matrices(megacomplexes, model, parameter)

    transitions = build_all_transitions(k_matrices, visualization_options.omitted_rate_constants)
    
    graph_data = dump_cytpscape_json_data(transitions)

    widget = GraphWidget(graph_data, visualization_options=visualization_options.__dict__)

    return widget

def visualize_dataset_model(dataset_model: str, model: Model, parameter: Parameters, exclude_megacomplexes: Optional[List[str]] = None, visualization_options: VisualizationOptions = VisualizationOptions()) -> GraphWidget:
    if dataset_model not in model.dataset:
        raise ValueError(f"Dataset model {dataset_model} not found in the model.")
    
    associated_megacomplexes = model.dataset[dataset_model].megacomplex
    if exclude_megacomplexes:
        megacomplexes = [mc for mc in associated_megacomplexes if mc not in exclude_megacomplexes]
    else:
        megacomplexes = associated_megacomplexes

    k_matrices = get_filled_megacomplex_k_matrices(megacomplexes, model, parameter)

    transitions = build_all_transitions(k_matrices, visualization_options.omitted_rate_constants)
    
    graph_data = dump_cytpscape_json_data(transitions)

    widget = GraphWidget(graph_data=json.dumps(graph_data), visualization_options=visualization_options.model_dump_json())

    return widget
