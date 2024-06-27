import os
import platform
import unittest
import subprocess
from graphmaster.graph import Graph
import networkx as nx
import random


def generate_random_graph(num_nodes, edge_prob) -> nx.Graph:
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    for u, v in G.edges():
        G.edges[u, v]["weight"] = random.uniform(0.1, 10.0)
    return G


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
    return map(int, lines)

class TestDensestSubgraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile the C++ file
        cls.cpp_file = "densest_subgraph.cpp"
        cls.executable = "densest_subgraph"

        if platform.system() == "Windows":
            cls.executable += ".exe"

        cls.cwd = os.path.dirname(__file__)
        compile_command = f"g++ -o {cls.executable} {cls.cpp_file}"
        try:
            result = subprocess.run(
                compile_command,
                shell=True,
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=cls.cwd
            )
            print(result.stdout.decode())
            print(result.stderr.decode())
        except subprocess.CalledProcessError as e:
            print("Compilation failed:")
            print(e.stderr.decode())
            raise

    @classmethod
    def tearDownClass(cls):
        # Remove the executable file after tests
        executable_path = os.path.join(cls.cwd, cls.executable)
        if os.path.exists(executable_path):
            os.remove(executable_path)

    def setUp(self):
        # Generate a random graph for each test
        self.num_nodes = 10
        self.edge_prob = 0.5
        self.graph = generate_random_graph(self.num_nodes, self.edge_prob)
        self.graph_wrapper = Graph()
        for u, v in self.graph.edges():
            self.graph_wrapper.add_edge(u, v)

        # Prepare input data for C++ program
        self.input_data = f"{self.num_nodes} {self.graph.number_of_edges()}\n"
        for u, v in self.graph.edges():
            self.input_data += f"{u + 1} {v + 1}\n" # Convert 0-based to 1-based indexing

    def test_densest_subgraph(self):
        # Calculate the densest subgraph using Python
        python_nodes = self.graph_wrapper.densest_subgraph()

        # Run the C++ program and get the result
        cpp_output = run_cpp_program(self.input_data, self.executable, self.cwd)
        cpp_nodes = parse_cpp_output(cpp_output)
        cpp_nodes = [node - 1 for node in cpp_nodes] # Convert 1-based to 0-based indexing

        # Compare the results
        self.assertEqual(set(python_nodes), set(cpp_nodes))

if __name__ == "__main__":
    unittest.main()
