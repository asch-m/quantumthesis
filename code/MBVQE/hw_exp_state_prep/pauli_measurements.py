import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

deco_letter_to_unitary = {'Y': np.array([[ 0.70710678+3.38813179e-21j,  0.70710678+3.38813179e-21j],
                                        [-0.70710678+3.38813179e-21j,  0.70710678+3.38813179e-21j]]),
                        'A': np.array([[ 0.70710678+3.38813179e-21j, -0.70710678+3.38813179e-21j],
                                        [ 0.70710678+3.38813179e-21j,  0.70710678+3.38813179e-21j]]), 
                        'Z': np.array([[ 1.+0.j,  0.+0.j],
                                           [ 0.+0.j, -1.+0.j]]),
                        'P': np.array([[0.70710678+0.70710678j, 0.        +0.j        ],
                                    [0.        +0.j        , 0.70710678-0.70710678j]]),
                        'M': np.array([[0.70710678-0.70710678j, 0.        +0.j        ],
                                    [0.        +0.j        , 0.70710678+0.70710678j]])             
                        }     

class ClusterState:
    """Class for representing a cluster state graph
    attributes:
        G: networkx graph
        decorations: dictionary of decorations for each node. 
                    Note that the keys are tuples (i,j) where i is the layer and j is the ansatz index.
                    Also, the leftmost operator is applied first.
                    Note: Z is actual Z operation, Y is sqrt(iY), A is sqrt(-iY),  P is sqrt(+iZ) and M is sqrt(-iZ)


    """
    def __init__(self):
        self.decorations = None
        self.G = None
      
    def initialize_decorations_grid(self, n_layers, n_ansatz):
        decorations = dict()
        for i in range(n_layers):
            for j in range(n_ansatz):
                decorations[(i,j)] = ''
                
        return decorations
    
    def initialize_decorations(self):
        decorations = dict()
        for i in self.G.nodes():
            decorations[i] = ''
        return decorations
    
    def remove_edge(self, edge):
        self.G.remove_edge(edge[0], edge[1])

    def remove_node(self, node):
        self.G.remove_node(node)

    def initialize_2D_graph(self, n_layers, n_ansatz):
        self.G = nx.grid_2d_graph(n_layers, n_ansatz)
        self.decorations = self.initialize_decorations_grid(n_layers, n_ansatz)
    
    def initialize_graph_from_edge_list(self, edge_list):
        self.G = nx.Graph()
        self.G.add_edges_from(edge_list)
        self.decorations = self.initialize_decorations()

def local_complementation(G, node):
    H = G.subgraph(G.neighbors(node))
    H_complement = nx.complement(H)
    G.remove_edges_from(H.edges())
    G.add_edges_from(H_complement.edges())

def bare_X_measurement(clusterstate, node, b0, sign):
    # implement the plain X measurement protocol without considering decorations
    neighbor_list = list(clusterstate.G.neighbors(node))
    b0_neigbor_list = list(clusterstate.G.neighbors(b0))
    local_complementation(clusterstate.G, b0)
    local_complementation(clusterstate.G, node)
    clusterstate.G.remove_node(node)
    local_complementation(clusterstate.G, b0)

    # add decorations to neighboring nodes
    if sign == 1:
        clusterstate.decorations[b0] += 'Y'
        join = b0_neigbor_list + list([b0])
        Z_nodes = list(set(neighbor_list) - set(join))
        for Z_node in Z_nodes:
            clusterstate.decorations[Z_node] += 'Z'
    if sign == -1:
        clusterstate.decorations[b0] += 'A'
        join = neighbor_list + list([node])
        Z_nodes = list(set(b0_neigbor_list) - set(join))
        for Z_node in Z_nodes:
            clusterstate.decorations[Z_node] += 'Z'


def bare_Z_measurement(clusterstate, node, sign):
    neighbor_list = list(clusterstate.G.neighbors(node))
    clusterstate.G.remove_node(node)
    if sign == -1:
        for neighbor in neighbor_list:
            clusterstate.decorations[neighbor] += 'Z'

def bare_Y_measurement(clusterstate, node, sign):
    neighbor_list = list(clusterstate.G.neighbors(node))
    local_complementation(clusterstate.G, node)
    clusterstate.G.remove_node(node)
    if sign == -1:
        for neighbor in neighbor_list:
            clusterstate.decorations[neighbor] += 'P'
    if sign == 1:
        for neighbor in neighbor_list:
            clusterstate.decorations[neighbor] += 'M'

def get_meas_given_decorations(measurement_axis, sign, decorations):
    for operator in decorations:
        if operator == 'Y' and measurement_axis == 'X':
            measurement_axis = 'Z'
            sign *= -1
        elif operator == 'Y' and measurement_axis == 'Z':
            measurement_axis = 'X'
        elif operator == 'Y' and measurement_axis == 'Y':
            pass

        if operator == 'A' and measurement_axis == 'X':
            measurement_axis = 'Z'
        elif operator == 'A' and measurement_axis == 'Z':
            measurement_axis = 'X'
        elif operator == 'A' and measurement_axis == 'Y':
            pass

        elif operator == 'Z' and measurement_axis == 'X':
            sign *= -1
        elif operator == 'Z' and measurement_axis == 'Z':
            pass
        elif operator == 'Z' and measurement_axis == 'Y':
            sign *= -1

        elif operator == 'P' and measurement_axis == 'X':
            measurement_axis = 'Y'
        elif operator == 'P' and measurement_axis == 'Z':
            pass
        elif operator == 'P' and measurement_axis == 'Y':
            measurement_axis = 'X'
            sign *= -1
        
        elif operator == 'M' and measurement_axis == 'X':
            measurement_axis = 'Y'
            sign *= -1
        elif operator == 'M' and measurement_axis == 'Z':
            pass
        elif operator == 'M' and measurement_axis == 'Y':
            measurement_axis = 'X'
    return measurement_axis, sign



def X_measurement(clusterstate, node, special_neighbor=None, meas_sign=1, verbose=False):
    """Performs an X measurement on the given node in the graph G.
        Note that the added decorations are technically not correct - they just serve as a aid to apply the correct operators later.

    Args:   
        clusterstate: ClusterState object
        node: the node (qubit) to be measured. Indexing starts at 1.
        special_neighbor: in the graphical meausrement algorithm, a special neighbor b_0 is chosen. Specificy this neighbor here.
    
    Returns:
        None
    """

    try:
        first, *rest = clusterstate.G.neighbors(node)
    except ValueError:
        if verbose:
            print("Node {} has no neighbors".format(node))
        clusterstate.G.remove_node(node)
        return

    #check if sign is 1 or -1
    assert meas_sign == 1 or meas_sign == -1, "meas_sign must be 1 or -1"
    
    if isinstance(special_neighbor, tuple) or isinstance(special_neighbor, int):
        # check if special neighbor is a node of the graph
        if special_neighbor not in clusterstate.G.neighbors(node):
            raise ValueError("Special neighbor {} is not a neighbor of node {}".format(special_neighbor, node))
        b0 = special_neighbor
    elif special_neighbor == 'same layer':
        lst = [(x,y) for x,y in clusterstate.G.neighbors(node) if x == node[0]]
        try:
            b0 = lst[0]
        except IndexError:
            if verbose:
                print("Node {} has no neighbors in the same layer. Chose {} instead.".format(node, first))
            # choose neighbor with smalles index sum
            
            b0 = first
    else:
        b0 = first

    #neighbor_list = list(clusterstate.G.neighbors(node))
    #b0_neigbor_list = list(clusterstate.G.neighbors(b0))
    if clusterstate.decorations[node] == '':
        bare_X_measurement(clusterstate, node, b0,  meas_sign)
        # print('no decorations')
    else:
        
        measurement_axis = 'X'
        sign =  meas_sign

        measurement_axis, sign = get_meas_given_decorations(measurement_axis, sign, clusterstate.decorations[node])
            
        if verbose:
            print('measuremetn axis: ', measurement_axis)
            print('sign: ', sign)
        if measurement_axis == 'Z':
            bare_Z_measurement(clusterstate, node, sign)
        elif measurement_axis == 'X':
            bare_X_measurement(clusterstate, node, b0, sign)
        elif measurement_axis == 'Y':
            bare_Y_measurement(clusterstate, node, sign)

def Z_measurement(clusterstate, node,  meas_sign=1, verbose=False):
    if clusterstate.decorations[node] == '':
        bare_Z_measurement(clusterstate, node,  meas_sign)
        # print('no decorations')
    else:
        measurement_axis = 'Z'
        sign =  meas_sign

        measurement_axis, sign = get_meas_given_decorations(measurement_axis, sign, clusterstate.decorations[node])
            
        if verbose:
            print('measuremetn axis: ', measurement_axis)
            print('sign: ', sign)
        if measurement_axis == 'Z':
            bare_Z_measurement(clusterstate, node, sign)
        elif measurement_axis == 'X':    
            try:
                first, *rest = clusterstate.G.neighbors(node)
            except ValueError:
                if verbose:
                    print("Node {} has no neighbors".format(node))
                clusterstate.G.remove_node(node)
            bare_X_measurement(clusterstate, node, first, sign)
        elif measurement_axis == 'Y':
            bare_Y_measurement(clusterstate, node, sign)

def Y_measurement(clusterstate, node, meas_sign = 1, verbose=False):
    if clusterstate.decorations[node] == '':
        bare_Y_measurement(clusterstate, node,  meas_sign)
        # print('no decorations')
    else:
        measurement_axis = 'Y'
        sign =  meas_sign

        measurement_axis, sign = get_meas_given_decorations(measurement_axis, sign, clusterstate.decorations[node])
            
        if verbose:
            print('measuremetn axis: ', measurement_axis)
            print('sign: ', sign)
        if measurement_axis == 'Z':
            bare_Z_measurement(clusterstate, node, sign)
        elif measurement_axis == 'X':
            try:
                first, *rest = clusterstate.G.neighbors(node)
            except ValueError:
                if verbose:
                    print("Node {} has no neighbors".format(node))
                clusterstate.G.remove_node(node)
            bare_X_measurement(clusterstate, node, first, sign)
        elif measurement_axis == 'Y':
            bare_Y_measurement(clusterstate, node, sign)


def old_X_measurement(clusterstate, node, special_neighbor=None, verbose=False):
    """Performs an X measurement on the given node in the graph G.
        Note that the added decorations are technically not correct - they just serve as a aid to apply the correct operators later.

    Args:   
        clusterstate: ClusterState object
        node: the node (qubit) to be measured. Indexing starts at 1.
        special_neighbor: in the graphical meausrement algorithm, a special neighbor b_0 is chosen. Specificy this neighbor here.
    
    Returns:
        None
    """

    try:
        first, *rest = clusterstate.G.neighbors(node)
    except ValueError:
        if verbose:
            print("Node {} has no neighbors".format(node))
        clusterstate.G.remove_node(node)
        return

    if isinstance(special_neighbor, tuple):
        # check if special neighbor is a node of the graph
        if special_neighbor not in clusterstate.G.neighbors(node):
            raise ValueError("Special neighbor {} is not a neighbor of node {}".format(special_neighbor, node))
        b0 = special_neighbor
    elif special_neighbor == 'same layer':
        lst = [(x,y) for x,y in grid_graph.G.neighbors(node) if x == node[0]]
        try:
            b0 = lst[0]
        except IndexError:
            if verbose:
                print("Node {} has no neighbors in the same layer. Chose {} instead.".format(node, first))
            # choose neighbor with smalles index sum
            
            b0 = first
    else:
        b0 = first

    neighbor_list = list(clusterstate.G.neighbors(node))
    b0_neigbor_list = list(clusterstate.G.neighbors(b0))
    if clusterstate.decorations[node] == '':
        local_complementation(clusterstate.G, b0)
        local_complementation(clusterstate.G, node)
        clusterstate.G.remove_node(node)
        local_complementation(clusterstate.G, b0)

        # add decorations to neighboring nodes
        clusterstate.decorations[b0] += 'Y'

        join = b0_neigbor_list + list([b0])
        Z_nodes = list(set(neighbor_list) - set(join))

        for Z_node in Z_nodes:
            clusterstate.decorations[Z_node] += 'Z'
    else:
        
        measurement_axis = 'X'
        sign = 1

        for operator in clusterstate.decorations[node]:
            if verbose:
                print('operator: ', operator)
                print('measurement axis: ', measurement_axis)
                print('logical and: ',  operator == 'Y' and measurement_axis == 'X' )
            if operator == 'Y' and measurement_axis == 'X':
                measurement_axis = 'Z'
            elif operator == 'Y' and measurement_axis == 'Z':
                measurement_axis = 'X'
            elif operator == 'Z' and measurement_axis == 'X':
                sign *= -1
            elif operator == 'Z' and measurement_axis == 'Z':
                pass
        if verbose:
            print('measuremetn axis: ', measurement_axis)
            print('sign: ', sign)
        if measurement_axis == 'Z':
            clusterstate.G.remove_node(node)
            if sign == -1:
                for neighbor in neighbor_list:
                    clusterstate.decorations[neighbor] += 'Z'
        elif measurement_axis == 'X':
            local_complementation(clusterstate.G, b0)
            local_complementation(clusterstate.G, node)
            clusterstate.G.remove_node(node)
            local_complementation(clusterstate.G, b0)
            
            # add decorations to neighboring nodes
            if sign == 1:
                clusterstate.decorations[b0] += 'Y'
                join = b0_neigbor_list + list([b0])
                Z_nodes = list(set(neighbor_list) - set(join))
                for Z_node in Z_nodes:
                    clusterstate.decorations[Z_node] += 'Z'
            if sign == -1:
                clusterstate.decorations[b0] += 'Y'
                join = neighbor_list + list([node])
                Z_nodes = list(set(b0_neigbor_list) - set(join))
                for Z_node in Z_nodes:
                    clusterstate.decorations[Z_node] += 'Z'

def draw_grid_graph(clusterstate, layout='skewed'):
    """Draws the graph G with decorations

    Args:
        clusterstate: ClusterState object

    Returns:
        None
    """
    if layout == 'skewed':
        pos = {(x,y):(y+0.1*x**2,-x+0.1*y**2) for x,y in clusterstate.G.nodes()}
        pos_labels = {(x,y):(y+0.1*x**2,-x+0.1*y**2+0.2) for x,y in clusterstate.G.nodes()}
    
    elif layout == 'grid':
        pos = {(x,y):(x,y) for x,y in clusterstate.G.nodes()}
        pos_labels = {(x,y):(x,y+0.2) for x,y in clusterstate.G.nodes()}

    plt.figure()
    nx.draw_networkx(clusterstate.G, with_labels=True, pos=pos, alpha=0.8)
    updated_decorations = {k: v for k, v in clusterstate.decorations.items() if k in clusterstate.G.nodes()}
    nx.draw_networkx_labels(clusterstate.G, pos=pos_labels, labels=updated_decorations)
    plt.show()

def draw_graph(clusterstate):
    """Draws the graph G with decorations

    Args:
        clusterstate: ClusterState object

    Returns:
        None
    """



    plt.figure()
    nx.draw_networkx(clusterstate.G, with_labels=True, alpha=0.8)
    updated_decorations = {k: v for k, v in clusterstate.decorations.items() if k in clusterstate.G.nodes()}
    nx.draw_networkx_labels(clusterstate.G, pos=nx.nx_agraph.graphviz_layout(clusterstate.G), labels=updated_decorations)
    plt.show()

