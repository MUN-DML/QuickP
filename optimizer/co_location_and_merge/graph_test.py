# Test function to verify the correctness of has_single_disjoint_path
import networkx as nx

from optimizer.co_location_and_merge.group_algorithm import is_edge_mergable


def test_has_single_disjoint_path():
    # Test case 1: Graph with a single path
    G1 = nx.DiGraph()
    G1.add_edges_from([(1, 2), (2, 3)])
    assert is_edge_mergable(1, 2, G1) == True, "Test case 1 failed"

    # Test case 2: Graph with two node-disjoint paths
    G2 = nx.DiGraph()
    G2.add_edges_from([(1, 3), (1, 4), (4, 3)])
    assert is_edge_mergable(1, 3, G2) == False, "Test case 2 failed"

    # Test case 3: Graph with no path between source and target
    G3 = nx.DiGraph()
    G3.add_edges_from([(1, 2), (3, 4)])
    assert is_edge_mergable(1, 2, G3) == True, "Test case 3 failed"  # No path means only one disjoint path

    # Test case 4: Graph with a single node-disjoint path but multiple edges between nodes
    G4 = nx.DiGraph()
    G4.add_edges_from([(1, 2), (2, 3), (1, 2)])  # Multiple edges between 1 and 2
    assert is_edge_mergable(1, 2, G4) == True, "Test case 4 failed"

    print("All test cases passed!")

test_has_single_disjoint_path()