class MyGraph:
    """
    In this class, we implement basic functionalities of undirected graphs as well as necessary
    functions and methods for our algorithms related to NDF embeddings.
    We only care about the topology of the graph and so we do not consider properties
    and attributes for vertices and edges. We develop the relevant mathematical functions and methods to deal with
    the metric structure of graphs.
    We also add several other methods useful for working with graphs.
    Here, we do not allow multi edges between two vertices or loops.

    The underlying data structure:
    An undirected graph is implemented as an object possessing only one property named adj_list which is a dictionary
    containing the adjacency list of the graph. The keys in this dictionary are nodes and the value associated to
    a key (node) is a set consisting of the neighbors of the node.

    """

    def __init__(self):
        """
        It initializes an empty graph.
        """
        self.adj_list = {}

    def __str__(self):
        """
        It returns the dictionary of the adjacency list of the graph as a string ready to be printed.
        """
        return str(self.adj_list)

    def add_vertex(self, vertex):
        """
        If the vertex is not already in the graph, this method adds it to the graph.
        """
        if vertex not in self.adj_list:
            self.adj_list[vertex] = set()
            return True
        return False

    def add_edge(self, x, y):
        """
        If the edge already exist, it return False. Otherwise it adds an edge between x and y.
        If one or both vertices do not exist, this method first adds the missing vertices and then adds the edge.
        """
        if x == y:  # discarding loops
            return False
        if x not in self.adj_list:
            self.adj_list[x] = set()
        if y not in self.adj_list:
            self.adj_list[y] = set()
        if y not in self.adj_list[x]:
            self.adj_list[x].add(y)
        if x not in self.adj_list[y]:
            self.adj_list[y].add(x)
        return True

    def from_edge_list(self, edge_list):
        """
        It builds a graph (or populates a non-empty graph) using the list of edges of a graph
        by adding appropriate vertices and edges to the empty graph (or non-empty graph).
        """
        for (x, y) in edge_list:
            self.add_edge(x, y)

    def get_vertex_list(self) -> list:
        """
        returns the list of all vertices of the graph.
        """
        return list(self.adj_list.keys())

    def get_edge_list(self) -> list:
        """
        returns the list of all edges of the graph.
        """
        edge_list = []
        visited_vertices = []
        for vertex in self.adj_list:
            connected_edges = [(vertex, other) for other in self.adj_list[vertex] if other not in visited_vertices]
            visited_vertices.append(vertex)
            edge_list = edge_list + connected_edges
        return edge_list

    def deg(self, vertex):
        """
        returns the degree of the vertex in the graph.
        """
        return len(self.adj_list[vertex])

    def neighs(self, vertex) -> list:
        """
        returns the list of all neighbors of the vertex in the graph.
        """
        return list(self.adj_list[vertex])

    def __eq__(self, other):
        """
        overloads the == operator. In other words it checks the equality of two graphs.
        """
        if set(self.adj_list) != set(other.adj_list):
            return False
        else:
            for v in self.adj_list:
                if self.adj_list[v] != other.adj_list[v]:
                    return False
        return True

    def copy(self):
        """
        returns a copy of the graph.
        """
        the_copy = MyGraph()
        for v in self.adj_list:
            the_copy.adj_list[v] = self.adj_list[v].copy()
        return the_copy

    def __add__(self, other):
        """
        overloading the + operator. In fact, this function merges two graphs.
        That is it does not repeat the common vertices and edges.
        """
        temp = self.copy()
        for v in other.adj_list:
            if v in temp.adj_list:
                temp.adj_list[v] = temp.adj_list[v].union(other.adj_list[v].copy())
            else:
                temp.adj_list[v] = other.adj_list[v].copy()
        return temp

    def remove_edge(self, x, y):
        """
        removes the edge (x, y) if it exists and returns True, otherwise it returns False
        """
        if x == y:
            return False
        if x in self.adj_list:
            if y in self.adj_list[x]:
                self.adj_list[x].remove(y)
                self.adj_list[y].remove(x)
                return True
        return False

    def remove_vertex(self, vertex):
        """
        If x is a vertex, this method removes all edges incident to x and then removes x and returns True,
        otherwise it returns False.
        """
        if vertex in self.adj_list:
            for v in self.adj_list[vertex]:
                x = self.adj_list[v].copy()
                x.remove(vertex)
                self.adj_list[v] = x
            self.adj_list.pop(vertex)  # This deletes the vertex
            return True
        return False

    def bfs_circles(self, center, radius=-1):
        """
        It returns an iterable of all circles (layers of Breadth-First Search) with radius less than or
        equal to "radius" around the center in the connected component of the center, here radius=-1 amounts to
        infinity. It employs a lazy strategy and does not continue Breadth-First Search beyond the given radius.
        """
        r = 0
        current_circle = {center}
        visited = {center}
        while current_circle:
            next_circle = set()
            for v in current_circle:
                for w in self.adj_list[v]:
                    if w not in visited:
                        next_circle.add(w)
                        visited.add(w)
            yield current_circle
            current_circle = next_circle
            r += 1
            if r == radius + 1:
                break

    def circles(self, center, radius=-1):
        """
        It returns a list of all circles around the "center" with radius less than or equal to "radius" in the connected
        component of the center, here radius=-1 amounts to infinity.
        """
        return [x for x in self.bfs_circles(center, radius)]

    def size_circles(self, center, radius=-1):
        """
        It returns a list of sizes of circles around the center with radius less than or equal to "radius" in the
        connected component of the center, here radius=-1 amounts to infinity.
        """
        return [len(x) for x in self.bfs_circles(center, radius)]

    def discs(self, center, radius=-1):
        """
        It returns a list of discs (as sets) with radius less than or equal to "radius" around the "center" in the
        connected component of the center, here radius=-1 amounts to infinity.
        """
        discs = [{center}]
        circles = self.circles(center, radius)
        for i in range(1, len(circles)):
            discs.append(discs[i - 1].union(circles[i]))
        return discs

    def connected_components(self):
        """
        This method returns a list of connected components of the graph as independent graphs.
        """
        components = []
        explored = set()
        for vertex in self.adj_list:
            if vertex not in explored:
                explored.add(vertex)
                current_queue = Queue(vertex)
                new_component = MyGraph()
                new_component.add_vertex(vertex)
                new_component.adj_list[vertex] = self.adj_list[vertex].copy()
                while current_queue.length > 0:
                    vertex = current_queue.dequeue()
                    for other_vertex in self.adj_list[vertex]:
                        if other_vertex not in explored:
                            explored.add(other_vertex)
                            current_queue.enqueue(other_vertex)
                            new_component.add_vertex(other_vertex)
                            new_component.adj_list[other_vertex] = self.adj_list[other_vertex].copy()
                components.append(new_component)
        return components

    def depth_first_search(self, vertex):
        """
        An implementation of Depth-First Search.
        """

        def get_next_vertex(v):
            temp = None
            for x in self.adj_list[v]:
                if x not in visited_list:
                    temp = x
                    break
            return temp

        visited_list = [vertex]
        visited_stack = Stack(vertex)
        current_vertex = vertex
        backward = False
        while visited_stack.height > 0:
            if backward:
                current_vertex = visited_stack.pop()
            next_vertex = get_next_vertex(current_vertex)
            backward = False
            if next_vertex:
                if (visited_stack.top and visited_stack.top.value != current_vertex) or (
                        visited_stack.top is None and get_next_vertex(current_vertex)):
                    visited_stack.push(current_vertex)
                current_vertex = next_vertex
                visited_stack.push(next_vertex)
                visited_list.append(next_vertex)
            else:
                backward = True
        return visited_list

    def degrees_list(self):
        """
        returns a sorted list of all degrees of nodes (without multiplicity) in the graph
        """
        degrees_list = list({len(self.adj_list[vertex]) for vertex in self.adj_list})
        degrees_list.sort()
        return degrees_list

    def degrees_dict(self):
        """
        :return: the (sorted) dictionary of all occurring degrees as keys and the number their occurrence as values
        """
        deg_dict = {}
        for d in self.degrees_list():
            deg_dict[d] = 0
        for vertex in self.adj_list:
            deg_dict[len(self.adj_list[vertex])] += 1
        return deg_dict

    def max_degree(self):
        """
        returns the maximum degree of nodes in the graph
        """
        maximum_degree = 0
        for vertex in self.adj_list:
            degree = len(self.adj_list[vertex])
            if degree > maximum_degree:
                maximum_degree = degree
        return maximum_degree

    def starting_points(self, ratio=1, max_length=10, starting_length=1, last_point=0) -> list:
        """
        :param ratio:
                ratio = 0 corresponds to minimal intervals list,
                ratio < 1 and ratio != 0 corresponds to vanilla intervals list
                ratio = 1 corresponds to uniform length intervals list
                ratio > 1 corresponds to increasing length intervals list
        :param max_length: the maximum length of intervals
        :param starting_length: the starting length of intervals. It is used only when ratio is greater than 1
        :param last_point: when it is positive, sets the last point in the list of starting points
        :return: a list of starting points of the intervals list
        """
        if last_point <= 0:
            last_point = self.max_degree()
        if ratio == 0:
            degrees_list = self.degrees_list()
            degrees_list[0] = 1  # To make sure the dynamic minimal list of starting points begins with 1
            return [x for x in degrees_list if x <= last_point]
        if ratio < 1:
            return list(range(1, last_point + 1))
        if ratio == 1:
            return self.uniform_points(max_length, last_point)
        else:
            return self.increasing_points(ratio, max_length, starting_length, last_point)

    @staticmethod
    def uniform_points(max_length, last_point) -> list:
        next_point = last_point
        starting_points = []
        while next_point > 0:
            starting_points.append(next_point)
            next_point -= max_length
            if next_point <= 1:
                starting_points.append(1)
                break
        starting_points.reverse()
        return starting_points

    @staticmethod
    def increasing_points(ratio, max_length, starting_length, last_point) -> list:
        next_length = starting_length
        starting_points = [1, starting_length + 1]
        while starting_points[-1] < last_point:
            next_length = min(next_length * ratio, max_length)
            next_point = int(starting_points[-1] + next_length)
            starting_points.append(next_point)
        if starting_points[-1] > last_point:
            starting_points.pop(-1)
        return starting_points

    def ndf(self, node, starting_points) -> list:
        """
        It returns the dynamic NDF vector representation of a node in the graph.
        """
        vector_length = len(starting_points)
        vector = [0] * vector_length
        for neighbor in self.neighs(node):
            neighbor_degree = self.deg(neighbor)
            for i in range(vector_length - 1):
                if starting_points[i + 1] > neighbor_degree >= starting_points[i]:
                    vector[i] += 1
                    break
            if neighbor_degree >= starting_points[vector_length - 1]:
                vector[vector_length - 1] += 1
        return vector

    def degree_vector(self, node, starting_points) -> list:
        """
        It returns the degree vector representation of nodes in a possibly dynamic graph.
        """
        vector_length = len(starting_points)
        vec = [0] * vector_length
        degree = self.deg(node)
        for i in range(vector_length - 1):
            if starting_points[i + 1] > degree >= starting_points[i]:
                vec[i] += 1
                break
        if degree >= starting_points[vector_length - 1]:
            vec[vector_length - 1] += 1
        return vec

    # -------------- Some functions to get to know the graph -------------

    def graph_stats(self):
        """"
        returns a dictionary of the graph statistics.
        """
        stat_dict = {"num_nodes": len(self.adj_list), "num_edges": len(self.get_edge_list()),
                     "nodes_degrees": self.degrees_list()}
        stat_dict["average_degree"] = 2 * stat_dict["num_edges"] / stat_dict["num_nodes"]
        return stat_dict

    def distance(self, vertex1, vertex2):
        """"
        returns the distance of two nodes. If they are not in the same component, it returns -1
        """
        dist = 0
        for circle in self.bfs_circles(vertex1):
            if vertex2 in circle:
                return dist
            dist += 1
        return -1

    def distances(self):
        """"
        returns a dictionary of all distances between distinct pairs of nodes in the graph
        which lie in the same component
        """
        dist_dict = {}
        counted = set()
        for x in self.adj_list:
            for y in self.adj_list:
                if (x, y) in counted or (y, x) in counted or x == y:
                    continue
                dist_dict[(x, y)] = self.distance(x, y)
                counted.add((x, y))
        return dist_dict

    def mean_distance(self):
        """
        returns the average distances of pairs of vertices in the graph, provided that they are in the same component
        """
        distances = list(self.distances().values())
        return sum(distances) / len(distances)

    def distances_from(self, vertex):
        """"
        returns a dictionary of distances between the given vertex and the rest of vertices
        in the same component of the graph
        """
        dist_dict = {}
        circles = self.circles(vertex)
        for d, circle in enumerate(circles):
            for node in circle:
                dist_dict[node] = d
        return dist_dict

    def max_distance_from(self, vertex):
        """
        returns the maximum distance from the given vertex
        """
        return len(self.circles(vertex)) - 1

    def mean_distance_from(self, vertex):
        """
        returns the average distance from the given vertex
        """
        distances = list(self.distances_from(vertex).values())
        return sum(distances) / len(distances)


"""
Here we build minimum versions of queue and stack classes to use in Breadth-First Search and Depth-First Search methods.
"""


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class Queue:
    def __init__(self, value=None):

        if value is None:
            self.first = None
            self.last = None
            self.length = 0
        else:
            new_node = Node(value)
            self.first = new_node
            self.last = new_node
            self.length = 1

    def enqueue(self, value):
        new_node = Node(value)
        if self.length == 0:
            self.first = new_node
            self.last = new_node
        else:
            self.last.next = new_node
            self.last = new_node
        self.length += 1
        return True

    def dequeue(self):
        if self.length == 0:
            return None
        temp = self.first
        self.first = temp.next
        self.length -= 1
        if self.length == 0:
            self.last = None
        return temp.value


class Stack:
    def __init__(self, value=None):
        if value is None:
            self.top = None
            self.height = 0
        else:
            new_node = Node(value)
            self.top = new_node
            self.height = 1

    def push(self, value):
        new_node = Node(value)
        if self.height == 0:
            self.top = new_node
            self.height += 1
            return True
        new_node.next = self.top
        self.top = new_node
        self.height += 1
        return True

    def pop(self):
        if self.height == 0:
            return None
        temp = self.top
        self.top = self.top.next
        temp.next = None
        self.height -= 1
        return temp.value
