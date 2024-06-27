import unittest
import networkx as nx
from graphmaster.graph import Graph


class TestKCoreDecomposition(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 100
        self.edge_prob = 0.1
        self.graphs = {
            "erdos_renyi": nx.erdos_renyi_graph(self.num_nodes, self.edge_prob),
            "scale_free": nx.barabasi_albert_graph(self.num_nodes, 3),
            "small_world": nx.watts_strogatz_graph(self.num_nodes, 4, 0.1),
            "sparse": nx.gnm_random_graph(self.num_nodes, 150)
        }

    def test_k_core_decomposition(self):
        for graph_name, graph in self.graphs.items():
            graph_wrapper = Graph()
            for u, v in graph.edges:
                graph_wrapper.add_edge(u, v)

            with self.subTest(graph=graph_name, k=3):
                k_core_nodes = graph_wrapper.k_core(3)
                expected_nodes = nx.k_core(graph, 3)
                self.assertCountEqual(k_core_nodes, expected_nodes)

            with self.subTest(graph=graph_name, k=5):
                k_core_nodes = graph_wrapper.k_core(5)
                expected_nodes = nx.k_core(graph, 5)
                self.assertCountEqual(k_core_nodes, expected_nodes)

            with self.subTest(graph=graph_name, k=10):
                k_core_nodes = graph_wrapper.k_core(10)
                expected_nodes = nx.k_core(graph, 10)
                self.assertCountEqual(k_core_nodes, expected_nodes)

if __name__ == "__main__":
    unittest.main()
