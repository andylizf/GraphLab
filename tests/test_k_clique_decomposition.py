import unittest
import networkx as nx
from graphmaster.graph import Graph


class TestKCliqueDecomposition(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 100
        self.edge_prob = 0.1
        self.graphs = {
            "erdos_renyi": nx.erdos_renyi_graph(self.num_nodes, self.edge_prob),
            "scale_free": nx.barabasi_albert_graph(self.num_nodes, 3),
            "small_world": nx.watts_strogatz_graph(self.num_nodes, 4, 0.1),
            "sparse": nx.gnm_random_graph(self.num_nodes, 150)
        }

    def get_k_cliques(self, graph, k):
        """Get all cliques of size k from the graph."""
        cliques = list(nx.find_cliques(graph))
        return [clique for clique in cliques if len(clique) == k]

    def test_k_clique_decomposition(self):
        for graph_name, graph in self.graphs.items():
            graph_wrapper = Graph()
            for u, v in graph.edges:
                graph_wrapper.add_edge(u, v)

            for k in [3, 4, 5]:
                with self.subTest(graph=graph_name, k=k):
                    expected_cliques = self.get_k_cliques(graph, k)
                    my_cliques = graph_wrapper.k_clique_decomposition(k)

                    # Convert sets to lists of sorted nodes for comparison
                    expected_cliques = [sorted(clique) for clique in expected_cliques]
                    my_cliques = [sorted(clique) for clique in my_cliques]

                    self.assertEqual(len(expected_cliques), len(my_cliques))
                    for clique in expected_cliques:
                        self.assertIn(clique, my_cliques)

if __name__ == "__main__":
    unittest.main()
