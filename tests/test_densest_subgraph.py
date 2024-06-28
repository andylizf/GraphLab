import os
import platform
import unittest
import subprocess
from graphmaster.graph import Graph
import networkx as nx

def run_cpp_program(input_data, executable, cwd):
    result = subprocess.run(
        [os.path.join(cwd, executable)],
        input=input_data,
        text=True,
        capture_output=True,
        check=True,
        cwd=cwd
    )
    return result.stdout

def parse_cpp_output(output):
    lines = output.strip().split('\n')
    return list(map(int, lines))

def calculate_density(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    if num_nodes > 1:
        return num_edges / num_nodes
    return 0

class TestDensestSubgraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cwd = os.path.join(os.path.dirname(__file__), "third-party", "densest_subgraph")
        cls.executable = os.path.join(cls.cwd, "densest_subgraph")

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

    def setUp(self):
        self.graph_types = {
            "erdos_renyi": nx.erdos_renyi_graph(10, 0.5),
            "scale_free": nx.barabasi_albert_graph(10, 3),
            "small_world": nx.watts_strogatz_graph(10, 4, 0.1),
            "sparse": nx.gnm_random_graph(10, 15)
        }

    def test_densest_subgraph(self):
        for graph_name, graph in self.graph_types.items():
            with self.subTest(graph=graph_name):
                graph_wrapper = Graph()
                for u, v in graph.edges():
                    graph_wrapper.add_edge(u, v)

                # Prepare input data for C++ program
                input_data = f"{graph.number_of_nodes()} {graph.number_of_edges()}\n"
                for u, v in graph.edges():
                    input_data += f"{u + 1} {v + 1}\n" # Convert 0-based to 1-based indexing

                # Calculate the densest subgraph using Python
                python_nodes = graph_wrapper.densest_subgraph()

                # Run the C++ program and get the result
                cpp_output = run_cpp_program(input_data, self.executable, self.cwd)
                cpp_nodes = parse_cpp_output(cpp_output)
                cpp_nodes = [node - 1 for node in cpp_nodes] # Convert 1-based to 0-based indexing

                # Compare the results
                self.assertEqual(set(python_nodes), set(cpp_nodes))

    def test_approximate_densest_subgraph(self):
        for graph_name, graph in self.graph_types.items():
            with self.subTest(graph=graph_name):
                graph_wrapper = Graph()
                for u, v in graph.edges():
                    graph_wrapper.add_edge(u, v)

                # Prepare input data for C++ program
                input_data = f"{graph.number_of_nodes()} {graph.number_of_edges()}\n"
                for u, v in graph.edges():
                    input_data += f"{u + 1} {v + 1}\n" # Convert 0-based to 1-based indexing

                # Run the C++ program and get the result
                cpp_output = run_cpp_program(input_data, self.executable, self.cwd)
                cpp_nodes = parse_cpp_output(cpp_output)
                cpp_nodes = [node - 1 for node in cpp_nodes] # Convert 1-based to 0-based indexing

                # Convert C++ result to subgraph and calculate density
                cpp_subgraph = graph.subgraph(cpp_nodes)
                cpp_density = calculate_density(cpp_subgraph)

                # Calculate the approximate densest subgraph using Python
                _, approx_density = graph_wrapper.approximate_densest_subgraph()

                # Ensure the density of the approximate densest subgraph is at least half of the densest subgraph
                self.assertGreaterEqual(approx_density, cpp_density / 2)

if __name__ == "__main__":
    unittest.main()
