from mydigraph import MyDiGraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# copying a MyDiGraph object into a NetworkX DiGraph object
def copy_my_digraph(mydg):
    nxdg = nx.DiGraph()
    nxdg.add_nodes_from(mydg.get_vertex_list())
    nxdg.add_edges_from(mydg.get_edge_list())
    return nxdg


# copying a NetworkX DiGraph object into a MyDiGraph object
def copy_nx_graph(nx_graph):
    my_graph = MyDiGraph()
    for v in nx_graph.nodes():
        my_graph.add_vertex(v)
    my_graph.from_edge_list(list(nx_graph.edges()))
    return my_graph


def dg(graph, pos="default", node_size=200, fig_size=(8, 6), dpi=80):
    if isinstance(graph, MyDiGraph):
        g = copy_my_digraph(graph)
    else:
        g = graph
    if pos == 'kamada':
        pos0 = nx.kamada_kawai_layout(g)
    elif pos == "circular":
        pos0 = nx.circular_layout(g)
    else:
        pos0 = nx.spring_layout(g)

    figure(figsize=fig_size, dpi=dpi)
    nx.draw_networkx_nodes(g, pos=pos0)
    nx.draw_networkx(g, node_size=node_size, node_color="lightblue", pos=pos0)
    plt.show()


# ---  matrix and vector representations of graph nodes functions:   ----------


def ndf_dict(myg, starting_points, inward=True):
    """
    :param myg: the graph name (MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :param inward: whether to consider inward or outward procedure
    :return: a dictionary whose keys are nodes of the graph and values are the NDF vectors (as numpy array)
    """
    temp_dict = {}
    for node in myg.adj_list:
        temp_dict[node] = np.array(myg.ndf(node, starting_points=starting_points, inward=inward))
    return temp_dict


def degree_vector_dict(myg, starting_points, inward=True):
    """
    :param myg: the graph name (MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :param inward: whether to consider inward or outward procedure
    :return: a dictionary whose keys are nodes of the graph and values are the NDF vectors (as numpy array)
    """
    temp_dict = {}
    for node in myg.adj_list:
        temp_dict[node] = np.array(myg.degree_vector(node, starting_points=starting_points, inward=inward))
    return temp_dict


def ndf_np_vectors(myg, starting_points, inward=True):
    """
    :param myg: the graph name (MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :param inward: whether to consider inward or outward procedure
    :return: a 2-dimensional numpy array consisting of NDF vector representation of all nodes of the graph
    """
    num_rows = len(myg.adj_list)
    num_cols = len(starting_points)
    dict_rep = ndf_dict(myg, starting_points=starting_points, inward=inward)
    matrix = np.zeros((num_rows, num_cols))
    for i, node in enumerate(dict_rep):
        matrix[i] += dict_rep[node]
    return matrix


# generating the NDFC (resp. RNDFC) matrix representation of nodes as 3-dimensional numpy array
def NDFC_matrix_rep(myg, starting_points, inward=True, radius=0, raw=False, decimals=6):
    """
    This function generates the NDFC (resp. RNDFC) matrix rep's of nodes of the graph
    :param myg: the name of the graph (a MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :param inward: whether to consider inward or outward procedure
    :param radius: the order of the NDFC (resp. RNDFC) matrix
    :param raw: False means NDFC and True means RNDFC (raw NDFC)
    :param decimals: the number of decimals
    :return: a 3-dimensional numpy array consisting of the NDFC (resp. RNDFC) matrix rep's of all nodes of the graph
    """
    num_matrices = len(myg.adj_list)
    num_rows = radius + 1
    num_cols = len(starting_points)
    ndf_dict_rep = ndf_dict(myg, starting_points=starting_points, inward=inward)
    tensor = np.zeros((num_matrices, num_rows, num_cols))
    for k, node in enumerate(ndf_dict_rep):
        matrix = np.zeros((num_rows, num_cols))
        if inward:
            circles = myg.in_circles(node, radius=radius)
        else:
            circles = myg.out_circles(node, radius=radius)
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


# generating the discounted inward NDFC matrix representation of nodes as 3-dimensional numpy array
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
        circles = myg.in_circles(node, radius=radius)
        for i, circle in enumerate(circles):
            if i == 0:
                matrix[i] += ndf_dict_rep[node]
            elif len(circle) > 0:
                for vertex in circle:
                    matrix[i] += ndf_dict_rep[vertex] / myg.out_deg(vertex)
                matrix[i] = matrix[i] / len(circle)
        tensor[k] += np.round(matrix, decimals=decimals)
    return tensor


# generating the CDF (resp. RCDF) matrix representation of nodes as 3-dimensional numpy array
def CDF_matrix_rep(myg,  starting_points, inward=True, radius=1, raw=False, decimals=6):
    """
    This function generates the CDF (resp. RCDF) matrix representations of nodes of the graph
    :param myg: the name of the graph (a MyGraph object)
    :param starting_points: the list of starting points that defines the intervals list of NDF
    :param inward: whether to consider inward or outward procedure
    :param radius: the order of the CDF (resp. RCDF) matrix
    :param raw: False means CDF and True means RCDF (raw CDF)
    :param decimals: the number of decimals
    :return: a 3-dimensional numpy array consisting of CDF (resp. RCDF) matrix rep's of all nodes of the graph
    """
    num_matrices = len(myg.adj_list)
    num_rows = radius
    num_cols = len(starting_points)
    tensor = np.zeros((num_matrices, num_rows, num_cols))
    degree_vectors = degree_vector_dict(myg, starting_points, inward=inward)
    for k, node in enumerate(myg.adj_list.keys()):
        matrix = np.zeros((num_rows, num_cols))
        if inward:
            circles = myg.in_circles(node, radius=radius)
        else:
            circles = myg.out_circles(node, radius=radius)
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
def VNDFC_matrix_rep(myg, starting_points, inward=True, radius=0, decimals=6):
    num_matrices = len(myg.adj_list)
    num_rows = radius + 2
    num_cols = len(starting_points)
    NDFC_tensor = NDFC_matrix_rep(myg, starting_points=starting_points, inward=inward, radius=radius, decimals=decimals)
    tensor = np.zeros((num_matrices, num_rows, num_cols))
    for k, node in enumerate(myg.adj_list):
        matrix = np.zeros((num_rows, num_cols))
        matrix[0] += np.array(myg.degree_vector(node, starting_points=starting_points, inward=inward))
        matrix[1:] += NDFC_tensor[k]
        tensor[k] += matrix
    return tensor

# --- End of matrix-vector reps   -------------------------


def print_dict(dic, keys_len=20):
    for x in dic.keys():
        print(f"{str(x):{keys_len}} {str(dic[x])} ")
