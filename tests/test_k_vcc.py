import os
import unittest
import subprocess
from graphmaster.graph import Graph
import networkx as nx
from itertools import combinations

def run_cpp_program(graph_file, k, executable, cwd, output_file):
    result = subprocess.run(
        [os.path.join(cwd, executable), "-g", graph_file, "-k", str(k), "-o", output_file],
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd
    )
    return result.stdout

def parse_cpp_output(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    kvccs = []
    current_kvcc = []
    for line in lines:
        if line.startswith("Node num"):
            if current_kvcc:
                kvccs.append(current_kvcc)
                current_kvcc = []
        else:
            try:
                nodes = list(map(int, line.split()))
                if nodes:
                    current_kvcc.extend(nodes)
            except ValueError:
                continue  # Ignore lines that can't be parsed
    if current_kvcc:
        kvccs.append(current_kvcc)
    return kvccs

class TestKVCC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.path.join(os.path.dirname(__file__), "third-party", "k_vcc")
        cls.executable = os.path.join(cls.cwd, "kvcc_baseline")
        cls.output_file = os.path.join(cls.cwd, "kvcc_output.txt")

        # Compile the C++ file using make
        try:
            result = subprocess.run(
                ["make"],
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=cls.cwd
            )
        except subprocess.CalledProcessError as e:
            print("Compilation failed:")
            print(e.stderr.decode())
            raise

    @classmethod
    def tearDownClass(cls):
        # Clean up the generated files after tests
        try:
            print("Compiling")
            subprocess.run(
                ["make", "clean"],
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=cls.cwd
            )
        except subprocess.CalledProcessError as e:
            print("Clean up failed:")
            print(e.stderr.decode())
            raise
        # Remove the output file if it exists
        if os.path.exists(cls.output_file):
            os.remove(cls.output_file)


    def setUp(self):
        self.num_nodes = 20  # Number of nodes in each test graph
        self.edge_prob = 0.05  # Probability of edge creation for Erdos-Renyi graph
        self.k = 3  # The k value for k-VCC
        self.graphs = {
            "erdos_renyi": nx.erdos_renyi_graph(self.num_nodes, self.edge_prob),
            "scale_free": nx.barabasi_albert_graph(self.num_nodes, 3),
            "small_world": nx.watts_strogatz_graph(self.num_nodes, 4, 0.1),
            "sparse": nx.gnm_random_graph(self.num_nodes, 15)
        }
        self.graph_instances = {name: self.create_graph_instance(g) for name, g in self.graphs.items()}

    def create_graph_instance(self, nx_graph):
        g = Graph()
        for u, v in nx_graph.edges():
            g.add_edge(u, v)
        return g

    def save_graph_to_file(self, graph, filename):
        with open(filename, 'w') as f:
            f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
            for u, v in graph.edges():
                f.write(f"{u + 1} {v + 1}\n")  # Convert 0-based to 1-based indexing

    def verify_kvcc(self, k_vcc, k):
        for subset in combinations(k_vcc.nodes, k-1):
            subgraph = k_vcc.copy()
            subgraph.remove_nodes_from(subset)
            if not nx.is_connected(subgraph):
                return False
        return True

    def verify_maximality(self, graph, k_vcc, k):
        nodes = set(k_vcc)
        for node in graph.nodes:
            if node not in nodes:
                new_nodes = nodes | {node}
                subgraph = graph.subgraph(new_nodes).copy()
                if self.verify_kvcc(subgraph, k):
                    return False
        return True

    def verify_kvcc_results(self, graph, k_vccs, k):
        for k_vcc in k_vccs:
            if not self.verify_kvcc(graph.subgraph(k_vcc), k):
                return False
            if not self.verify_maximality(graph, k_vcc, k):
                return False
        return True

    def test_kvcc(self):
        for graph_name, graph in self.graphs.items():
            with self.subTest(graph=graph_name):
                # Create GraphWrapper instance
                graph_wrapper = self.create_graph_instance(graph)

                # Calculate the k-VCC using Python implementation
                python_kvccs = graph_wrapper.find_k_vcc(self.k)

                # Verify Python k-VCC results
                self.assertTrue(self.verify_kvcc_results(graph, python_kvccs, self.k))

                # Save the graph to a file for C++ program
                graph_file = os.path.join(self.cwd, f"{graph_name}.txt")
                self.save_graph_to_file(graph, graph_file)

                # Run the C++ program and get the result
                run_cpp_program(graph_file, self.k, self.executable, self.cwd, self.output_file)
                cpp_kvccs = parse_cpp_output(self.output_file)
                cpp_kvccs = [[node - 1 for node in kvcc] for kvcc in cpp_kvccs]  # Convert 1-based to 0-based indexing

                # Verify the Python results against the C++ results
                for python_kvcc in python_kvccs:
                    self.assertIn(sorted(python_kvcc), sorted(cpp_kvccs))

                # Clean up the graph file
                os.remove(graph_file)

if __name__ == "__main__":
    unittest.main()
