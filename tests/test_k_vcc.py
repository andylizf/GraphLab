import unittest
import networkx as nx
from graphmaster.graph import Graph
from itertools import combinations

class TestKVCC(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 20  # Number of nodes in each test graph
        self.edge_prob = 0.05  # Probability of edge creation for Erdos-Renyi graph
        self.k = 3  # The k value for k-VCC
        self.graphs = {
            "erdos_renyi": nx.erdos_renyi_graph(self.num_nodes, self.edge_prob),
            "scale_free": nx.barabasi_albert_graph(self.num_nodes, 3),
            "small_world": nx.watts_strogatz_graph(self.num_nodes, 4, 0.1),
            "sparse": nx.gnm_random_graph(self.num_nodes, 150)
        }
        self.graph_instances = {name: self.create_graph_instance(g) for name, g in self.graphs.items()}

    def create_graph_instance(self, nx_graph):
        g = Graph()
        for u, v in nx_graph.edges():
            g.add_edge(u, v)
        return g

    def verify_kvcc(self, k_vcc, k):
        for subset in combinations(k_vcc.nodes, k-1):
            subgraph = k_vcc.copy()
            subgraph.remove_nodes_from(subset)
            if not nx.is_connected(subgraph):
                return False
        return True

    def verify_maximality(self, graph, k_vcc, k):
        nodes = set(k_vcc.nodes)
        for node in graph.nodes:
            if node not in nodes:
                new_nodes = nodes | {node}
                subgraph = graph.subgraph(new_nodes).copy()
                if self.verify_kvcc(subgraph, k):
                    return False
        return True

    def verify_kvcc_results(self, graph, k_vccs, k):
        for k_vcc in k_vccs:
            if not self.verify_kvcc(k_vcc, k):
                return False
            if not self.verify_maximality(graph, k_vcc, k):
                return False
        return True

    def test_kvcc(self):
        for graph_name, graph in self.graph_instances.items():
            with self.subTest(graph=graph_name):
                k_vccs = graph.find_k_vcc(self.k)
                self.assertTrue(self.verify_kvcc_results(self.graphs[graph_name], k_vccs, self.k))

if __name__ == "__main__":
    unittest.main()
