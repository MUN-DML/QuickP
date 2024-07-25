from enum import Enum
import networkx as nx


class NodeWeightFunction(Enum):
    SUM_COMP_COST = 'sum_comp_cost'
    AVE_COMP_COST = 'ave_comp_cost'
    AVE_COMP_COST_WITH_IN_DEGREE = 'ave_comp_cost_with_in_degree'


class EdgeWeightFunction(Enum):
    MOCK_COMMUNICATION_COST = 'source_output_tensor'
    SOURCE_OUTPUT_TENSOR_WITH_COMP = 'source_output_tensor_with_comp'
