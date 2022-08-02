from mygraph import MyGraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# copying a NetworkX Graph object into a MyGraph object
def copy_nx_graph(nx_graph):
    my_graph = MyGraph()
    for v in nx_graph.nodes():
        my_graph.add_vertex(v)
    my_graph.from_edge_list(list(nx_graph.edges()))
    return my_graph


# copying a MyGraph object into a NetworkX Graph object
def copy_my_graph(my_graph):
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(my_graph.get_vertex_list())
    nx_graph.add_edges_from(my_graph.get_edge_list())
    return nx_graph


# drawing the graph
# noinspection SpellCheckingInspection
def dg(graph, pos="default", node_size=200, fig_size=(8, 6), dpi=80):
    if isinstance(graph, MyGraph):
        g = copy_my_graph(graph)
    else:
        g = graph
    if pos == 'kamada':
        pos0 = nx.kamada_kawai_layout(g)
    elif pos == "circular":
        pos0 = nx.circular_layout(g)
    else:
        pos0 = nx.spring_layout(g)

    figure(figsize=fig_size, dpi=dpi)
    nx.draw_networkx(g, node_size=node_size, node_color="lightblue", pos=pos0)
    plt.show()


# ---  matrix and vector representations of graph nodes functions:   ----------


def ndf_dict(myg, starting_points):
    """
    :param myg: the graph name (MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :return: a dictionary whose keys are nodes of the graph and values are the NDF vectors (as numpy array)
    """
    temp_dict = {}
    for node in myg.adj_list:
        temp_dict[node] = np.array(myg.ndf(node, starting_points=starting_points))
    return temp_dict


def degree_vector_dict(myg, starting_points):
    """
    :param myg: the graph name (MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :return: a dictionary whose keys are nodes of the graph and values are the NDF vectors (as numpy array)
    """
    temp_dict = {}
    for node in myg.adj_list:
        temp_dict[node] = np.array(myg.degree_vector(node, starting_points=starting_points))
    return temp_dict


def ndf_np_vectors(myg, starting_points):
    """
    :param myg: the graph name (MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :return: a 2-dimensional numpy array consisting of NDF vector representation of all nodes of the graph
    """
    num_rows = len(myg.adj_list)
    num_cols = len(starting_points)
    dict_rep = ndf_dict(myg, starting_points=starting_points)
    matrix = np.zeros((num_rows, num_cols))
    for i, node in enumerate(dict_rep):
        matrix[i] += dict_rep[node]
    return matrix


# generating the NDFC (resp. RNDFC) matrix representation of nodes as 3-dimensional numpy array
def NDFC_matrix_rep(myg, starting_points, radius=0, raw=False, decimals=6):
    """
    This function generates the NDFC (resp. RNDFC) matrix rep's of nodes of the graph
    :param myg: the name of the graph (a MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :param radius: the order of the NDFC (resp. RNDFC) matrix
    :param raw: False means NDFC and True means RNDFC (raw NDFC)
    :param decimals: the number of decimals
    :return: a 3-dimensional numpy array consisting of the NDFC (resp. RNDFC) matrix rep's of all nodes of the graph
    """
    num_matrices = len(myg.adj_list)
    num_rows = radius + 1
    num_cols = len(starting_points)
    ndf_dict_rep = ndf_dict(myg, starting_points=starting_points)
    tensor = np.zeros((num_matrices, num_rows, num_cols))
    for k, node in enumerate(ndf_dict_rep):
        matrix = np.zeros((num_rows, num_cols))
        circles = myg.circles(node, radius=radius)
        for i, circle in enumerate(circles):
            if len(circle) > 0:
                for vertex in circle:
                    matrix[i] += ndf_dict_rep[vertex]
                if not raw:
                    matrix[i] = matrix[i] / len(circle)
        if raw:
            tensor[k] += matrix
        else:
            tensor[k] += np.round(matrix, decimals=decimals)
    return tensor


# generating the NDFC (resp. RNDFC) matrix representation of nodes as 3-dimensional numpy array
def discounted_NDFC_matrix_rep(myg, starting_points, radius=0, decimals=6):
    """
    This function generates the NDFC (resp. RNDFC) matrix rep's of nodes of the graph
    :param myg: the name of the graph (a MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :param radius: the order of the NDFC (resp. RNDFC) matrix
    :param decimals: the number of decimals
    :return: a 3-dimensional numpy array consisting of the NDFC (resp. RNDFC) matrix rep's of all nodes of the graph
    """
    num_matrices = len(myg.adj_list)
    num_rows = radius + 1
    num_cols = len(starting_points)
    ndf_dict_rep = ndf_dict(myg, starting_points=starting_points)
    tensor = np.zeros((num_matrices, num_rows, num_cols))
    for k, node in enumerate(ndf_dict_rep):
        matrix = np.zeros((num_rows, num_cols))
        circles = myg.circles(node, radius=radius)
        for i, circle in enumerate(circles):
            if i == 0:
                matrix[i] += ndf_dict_rep[node]
            elif len(circle) > 0:
                for vertex in circle:
                    matrix[i] += ndf_dict_rep[vertex] / myg.deg(vertex)
                matrix[i] = matrix[i] / len(circle)
        tensor[k] += np.round(matrix, decimals=decimals)
    return tensor


# generating the CDF (resp. RCDF) matrix representation of nodes as 3-dimensional numpy array
def CDF_matrix_rep(myg,  starting_points, raw=False, radius=1, decimals=6):
    """
    This function generates the CDF (resp. RCDF) matrix representations of nodes of the graph
    :param myg: the name of the graph (a MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :param radius: the order of the CDF (resp. RCDF) matrix
    :param raw: False means CDF and True means RCDF (raw CDF)
    :param decimals: the number of decimals
    :return: a 3-dimensional numpy array consisting of CDF (resp. RCDF) matrix rep's of all nodes of the graph
    """
    num_matrices = len(myg.adj_list)
    num_rows = radius
    num_cols = len(starting_points)
    tensor = np.zeros((num_matrices, num_rows, num_cols))
    degree_vectors = degree_vector_dict(myg, starting_points)
    for k, node in enumerate(myg.adj_list.keys()):
        matrix = np.zeros((num_rows, num_cols))
        circles = myg.circles(node, radius=radius)
        del circles[0]
        for i, circle in enumerate(circles):
            if len(circle) > 0:
                for vertex in circle:
                    matrix[i] += degree_vectors[vertex]
                if not raw:
                    matrix[i] = matrix[i] / len(circle)
        if raw:
            tensor[k] += matrix
        else:
            tensor[k] += np.round(matrix, decimals=decimals)
    return tensor


# generating the VNDFC matrix representation of nodes as 3-dimensional numpy array
def VNDFC_matrix_rep(myg, starting_points, radius=0, decimals=6):
    num_matrices = len(myg.adj_list)
    num_rows = radius + 2
    num_cols = len(starting_points)
    NDFC_tensor = NDFC_matrix_rep(myg, starting_points=starting_points, radius=radius, decimals=decimals)
    tensor = np.zeros((num_matrices, num_rows, num_cols))
    for k, node in enumerate(myg.adj_list):
        matrix = np.zeros((num_rows, num_cols))
        matrix[0] += np.array(myg.degree_vector(node, starting_points=starting_points))
        matrix[1:] += NDFC_tensor[k]
        tensor[k] += matrix
    return tensor


def vec_normalization(v, method=None):
    """
    v is a numpy array.
    """
    if method == "max":
        return v / v.max()
    if method == "sum":
        return v / v.sum()
    if method == "norm":
        return v / np.linalg.norm(v)
    if method == "normal":
        return (v - v.mean()) / v.std()
    if method == "sigmoid":
        return 1 / (1 + np.exp(-v))
    return v


# --- End of matrix-vector reps   -------------------------
# -----     Adding and Removing Edges and Nodes    ------------------


def remove_add_random_edges(myg, num_remove, num_add, seed=None):
    """
    We assume the original graph is connected. After removing certain edges, to make sure the graph produced by 
    this method is still connected, we consider all possible connected components and connect them with some edges. 
    Therefore, we may have to add more than num_add edges.
    :param seed: Sets the seed for random choices
    :param myg: The name of the graph
    :param num_remove: number of edges to be removed
    :param num_add: number of edges to be added
    :return: The modified graph
    """
    import random

    temp_graph = myg.copy()
    edges = temp_graph.get_edge_list()
    removed_edges = []
    added_edges = []
    for i in range(num_remove):  # removing some edges randomly
        tmp_seed = None
        if seed or seed == 0:
            tmp_seed = seed + i
        random.seed(tmp_seed)
        edge = random.choice(edges)
        temp_graph.remove_edge(edge[0], edge[1])
        removed_edges.append(edge)
        edges.remove(edge)
    components = temp_graph.connected_components()
    nodes_partition = [comp.get_vertex_list() for comp in components]
    for j in range(len(components) - 1):
        tmp_seed = None
        if seed or seed == 0:
            tmp_seed = seed + j
        random.seed(tmp_seed)
        node_j = random.choice(nodes_partition[j])
        node_j1 = random.choice(nodes_partition[j + 1])
        temp_graph.add_edge(node_j, node_j1)
        added_edges.append((node_j, node_j1))
    nodes_list = temp_graph.get_vertex_list()
    edges_list = temp_graph.get_edge_list()
    k = 0
    while num_add > len(added_edges):
        tmp_seed = None
        if seed or seed == 0:
            tmp_seed = seed + k
            k += 1
        random.seed(tmp_seed)
        node1 = random.choice(nodes_list)
        node2 = random.choice(nodes_list)
        if node1 != node2 and (node1, node2) not in edges_list and (node2, node1) not in edges_list:
            temp_graph.add_edge(node1, node2)
            edges_list.append((node1, node2))
            added_edges.append((node1, node2))
    return temp_graph

# -----   End of Adding and Removing Edges    ------------------

# printing the statistics of networkx graphs


def print_nx_stats(nxg):
    myg = copy_nx_graph(nxg)
    print_dict(myg.graph_stats())


def print_dict(dic, keys_len=20):
    for x in dic.keys():
        print(f"{str(x):{keys_len}} {str(dic[x])} ")
