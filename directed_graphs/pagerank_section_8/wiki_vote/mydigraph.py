class MyDiGraph:
    """
    In this class, we implement basic functionalities of directed graphs as well as necessary
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
            self.adj_list[vertex] = (set(), set())
            # The first set records all vertices that there is a directed edge from vertex into them.
            # So the first set accounts all outward edges from vertex.
            # The second set records all vertices that there is a directed edge from them into vertex.
            # The second set account all inward edges to vertex.
            return True
        return False

    def add_edge(self, x, y):
        """
        If the edge does not already exist, it adds an edge between x and y.
        If one or both vertices do not exist, this method first adds the missing vertices and then adds the edge.
        """
        if x == y:  # discarding loops
            return False
        if x not in self.adj_list:
            self.adj_list[x] = (set(), set())
        if y not in self.adj_list:
            self.adj_list[y] = (set(), set())
        if y not in self.adj_list[x][0]:
            self.adj_list[x][0].add(y)
        if x not in self.adj_list[y][1]:
            self.adj_list[y][1].add(x)
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
        edges = set()
        for vertex in self.adj_list:
            outward_edges = {(vertex, other) for other in self.adj_list[vertex][0]}
            edges = edges.union(outward_edges)
        return list(edges)

    def in_deg(self, vertex):
        """
        returns the indegree of the vertex in the graph.
        """
        return len(self.adj_list[vertex][1])

    def out_deg(self, vertex):
        """
        returns the outdegree of the vertex in the graph.
        """
        return len(self.adj_list[vertex][0])

    def in_neighs(self, vertex) -> list:
        """
        returns the list of all nodes in the graph which there is a directed edge from them towards vertex.
        """
        return list(self.adj_list[vertex][1])

    def out_neighs(self, vertex) -> list:
        """
        returns the list of all nodes in the graph which there is a directed edge from vertex towards them.
        """
        return list(self.adj_list[vertex][0])

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
        the_copy = MyDiGraph()
        for v in self.adj_list:
            the_copy.adj_list[v] = (self.adj_list[v][0].copy(), self.adj_list[v][1].copy())
        return the_copy

    def prefix_vertices(self, prefix="x"):
        """
        returns a copy of the graph with the vertices are prefixed.
        """
        temp = MyDiGraph()
        for v in self.adj_list:
            temp.adj_list[prefix+str(v)] = ({prefix+str(u) for u in self.adj_list[v][0]},
                                            {prefix+str(u) for u in self.adj_list[v][1]})
        return temp

    def __add__(self, other):
        """
        overloading the + operator. In fact, this function merges two graphs.
        That is it does not repeat the common vertices and edges.
        """
        temp = self.copy()
        for v in other.adj_list:
            if v in temp.adj_list:
                temp.adj_list[v] = (temp.adj_list[v][0].union(other.adj_list[v][0].copy()),
                                    temp.adj_list[v][1].union(other.adj_list[v][1].copy()))
            else:
                temp.adj_list[v] = (other.adj_list[v][0].copy(), other.adj_list[v][1].copy())
        return temp

    def remove_edge(self, x, y):
        """
        removes the edge (x, y) if it exists and returns True, otherwise it returns False
        """
        if x == y:
            return False
        if x in self.adj_list:
            if y in self.adj_list[x][0]:
                self.adj_list[x][0].remove(y)
                return True
        return False

    def remove_vertex(self, vertex):
        """
        If x is a vertex, this method removes all edges incident to x and then removes x and returns True,
        otherwise it returns False.
        """
        if vertex in self.adj_list:
            for v in self.adj_list[vertex][0]:
                x1 = self.adj_list[v][1].copy()
                x1.remove(vertex)
                self.adj_list[v] = (self.adj_list[v][0], x1)
            for v in self.adj_list[vertex][1]:
                x0 = self.adj_list[v][0].copy()
                x0.remove(vertex)
                self.adj_list[v] = (x0, self.adj_list[v][1])
            self.adj_list.pop(vertex)  # This deletes the vertex
            return True
        return False

    def bfs_in_circles(self, center, radius=-1):
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
                for w in self.adj_list[v][1]:
                    if w not in visited:
                        next_circle.add(w)
                        visited.add(w)
            yield current_circle
            current_circle = next_circle
            r += 1
            if r == radius + 1:
                break

    def bfs_out_circles(self, center, radius=-1):
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
                for w in self.adj_list[v][0]:
                    if w not in visited:
                        next_circle.add(w)
                        visited.add(w)
            yield current_circle
            current_circle = next_circle
            r += 1
            if r == radius + 1:
                break

    def in_circles(self, center, radius=-1):
        """
        It returns a list of inward circles around the "center" with radius <= "radius",
        here radius=-1 amounts to infinity.
        """
        return [x for x in self.bfs_in_circles(center, radius)]

    def out_circles(self, center, radius=-1):
        """
        It returns a list of outward circles around the "center" with radius <= "radius",
        here radius=-1 amounts to infinity.
        """
        return [x for x in self.bfs_out_circles(center, radius)]

    def out_connected(self, center):
        """
        It returns the union of all outward circles around the "center".
        """
        temp_set = set()
        for circ in self.bfs_out_circles(center):
            temp_set = temp_set.union(circ)
        return temp_set

    def in_connected(self, center):
        """
        It returns the union of all inward circles around the "center".
        """
        temp_set = set()
        for circ in self.bfs_in_circles(center):
            temp_set = temp_set.union(circ)
        return temp_set

    def strongly_connected(self, center):
        """
        It returns the set o all vertices that are both inward and outward connected to the "center".
        """
        return self.in_connected(center).intersection(self.out_connected(center))

    def size_in_circles(self, center, radius=-1):
        """
        It returns a list of sizes of circles around the center with radius less than or equal to "radius" in the
        connected component of the center, here radius=-1 amounts to infinity.
        """
        return [len(x) for x in self.bfs_in_circles(center, radius)]

    def size_out_circles(self, center, radius=-1):
        """
        It returns a list of sizes of circles around the center with radius less than or equal to "radius" in the
        connected component of the center, here radius=-1 amounts to infinity.
        """
        return [len(x) for x in self.bfs_out_circles(center, radius)]

    def in_discs(self, center, radius=-1):
        """
        It returns a list of inward discs (as sets) with radius <=> "radius" around the "center",
        here radius=-1 amounts to infinity.
        """
        discs = [{center}]
        circles = self.in_circles(center, radius)
        for i in range(1, len(circles)):
            discs.append(discs[i - 1].union(circles[i]))
        return discs

    def out_discs(self, center, radius=-1):
        """
        It returns a list of outward discs (as sets) with radius <=> "radius" around the "center",
        here radius=-1 amounts to infinity.
        """
        discs = [{center}]
        circles = self.out_circles(center, radius)
        for i in range(1, len(circles)):
            discs.append(discs[i - 1].union(circles[i]))
        return discs

    def strongly_connected_components(self):
        """
        This method returns a list of strongly connected components of the directed graph as independent graphs.
        """
        components = []
        explored = set()
        for vertex in self.adj_list:
            if vertex not in explored:
                new_component = MyDiGraph()
                s_connected = self.strongly_connected(vertex)
                explored = explored.union(s_connected)
                for v in s_connected:
                    new_component.adj_list[v] = (self.adj_list[v][0].intersection(s_connected),
                                                 self.adj_list[v][1].intersection(s_connected))
                components.append(new_component)
        return components

    def in_degrees_list(self):
        """
        returns a sorted list of all indegrees of nodes (without multiplicity) in the graph
        """
        degrees_list = list({len(self.adj_list[vertex][1]) for vertex in self.adj_list})
        degrees_list.sort()
        return degrees_list

    def out_degrees_list(self):
        """
        returns a sorted list of all outdegrees of nodes (without multiplicity) in the graph
        """
        degrees_list = list({len(self.adj_list[vertex][0]) for vertex in self.adj_list})
        degrees_list.sort()
        return degrees_list

    def max_in_degree(self):
        """
        returns the maximum indegree of nodes in the graph
        """
        maximum_degree = 0
        for vertex in self.adj_list:
            degree = len(self.adj_list[vertex][1])
            if degree > maximum_degree:
                maximum_degree = degree
        return maximum_degree

    def max_out_degree(self):
        """
        returns the maximum outdegree of nodes in the graph
        """
        maximum_degree = 0
        for vertex in self.adj_list:
            degree = len(self.adj_list[vertex][0])
            if degree > maximum_degree:
                maximum_degree = degree
        return maximum_degree

    def starting_points(self, ratio=1, max_length=10, starting_length=1, inward=True, last_point=0) -> list:
        """
        :param ratio:
                ratio = 0 corresponds to minimal intervals list,
                ratio < 1 and ratio != 0 corresponds to vanilla intervals list
                ratio = 1 corresponds to uniform length intervals list
                ratio > 1 corresponds to increasing length intervals list
        :param max_length: the maximum length of intervals
        :param starting_length: the starting length of intervals. It is used only when ratio is greater than 1
        :param inward: True or indegrees and False for outdegrees
        :param last_point: when it is positive, sets the last point in the list of starting points
        :return: a list of starting points of the intervals list
        """
        if last_point <= 0:
            if inward:
                last_point = self.max_in_degree()
            else:
                last_point = self.max_out_degree()
        if ratio == 0:
            if inward:
                degree_list = self.in_degrees_list()
            else:
                degree_list = self.out_degrees_list()
            degree_list[0] = 0  # to make sure the dynamic minimal list of starting points begins with 0
            return [x for x in degree_list if x <= last_point]
        if ratio < 1:
            return list(range(0, last_point + 1))
        if ratio == 1:
            return self.uniform_points(max_length, last_point)
        else:
            return self.increasing_points(ratio, max_length, starting_length, last_point)

    @staticmethod
    def uniform_points(max_length, last_point) -> list:
        next_point = last_point
        starting_points = []
        while next_point > -1:
            starting_points.append(next_point)
            next_point -= max_length
            if next_point <= 0:
                starting_points.append(0)
                break
        starting_points.reverse()
        return starting_points

    @staticmethod
    def increasing_points(ratio, max_length, starting_length, last_point) -> list:
        next_length = starting_length
        starting_points = [0, starting_length]
        while starting_points[-1] < last_point:
            next_length = min(next_length * ratio, max_length)
            next_point = int(starting_points[-1] + next_length)
            starting_points.append(next_point)
        if starting_points[-1] > last_point:
            starting_points.pop(-1)
        return starting_points

    def ndf(self, node, starting_points, inward=True) -> list:
        """
        It returns the dynamic NDF vector representation of a node in the graph.
        """
        if inward:
            neighbors = self.in_neighs(node)
        else:
            neighbors = self.out_neighs(node)
        vector_length = len(starting_points)
        vector = [0] * vector_length
        for neighbor in neighbors:
            if inward:
                neighbor_degree = self.in_deg(neighbor)
            else:
                neighbor_degree = self.out_deg(neighbor)
            for i in range(vector_length - 1):
                if starting_points[i + 1] > neighbor_degree >= starting_points[i]:
                    vector[i] += 1
                    break
            if neighbor_degree >= starting_points[vector_length - 1]:
                vector[vector_length - 1] += 1
        return vector

    def degree_vector(self, node, starting_points, inward=True) -> list:
        """
        It returns the degree vector representation of nodes in a possibly dynamic graph.
        """
        vector_length = len(starting_points)
        vec = [0] * vector_length
        if inward:
            degree = self.in_deg(node)
        else:
            degree = self.out_deg(node)
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
                     "nodes_indegrees": self.in_degrees_list(), "nodes_outdegrees": self.out_degrees_list()}
        stat_dict["average_degree"] = 2 * stat_dict["num_edges"] / stat_dict["num_nodes"]
        return stat_dict
