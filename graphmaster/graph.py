import networkx as nx
import plotly.graph_objects as go
import re
import numpy as np


class Graph:
    def __init__(self):
        self.graph = nx.Graph()
        self.mapping = {}
        self.reverse_mapping = {}

    def load(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if not line.startswith("#"):
                    lines = lines[i:]
                    break

            for line in lines:
                numbers = re.findall(r"\d+", line)
                if len(numbers) >= 2:
                    u, v = map(int, numbers[:2])
                    self.add_edge(u, v)

    def save(self, output_path):
        with open(output_path, "w") as f:
            for edge in self.graph.edges():
                u, v = edge
                original_u, original_v = (
                    self.reverse_mapping[u],
                    self.reverse_mapping[v],
                )
                f.write(f"{original_u} {original_v}\n")

    def add_node(self, node) -> None:
        if node not in self.mapping:
            index = len(self.mapping)
            self.mapping[node] = index
            self.reverse_mapping[index] = node
            self.graph.add_node(index)

    def remove_node(self, node):
        if node in self.mapping:
            index = self.mapping.pop(node)
            self.reverse_mapping.pop(index)
            self.graph.remove_node(index)

    def add_edge(self, u, v):
        for node in (u, v):
            if node not in self.mapping:
                self.add_node(node)
        u_mapped, v_mapped = self.mapping[u], self.mapping[v]
        if u_mapped != v_mapped:
            self.graph.add_edge(u_mapped, v_mapped)

    def remove_edge(self, u, v):
        if u in self.mapping and v in self.mapping:
            u_mapped, v_mapped = self.mapping[u], self.mapping[v]
            self.graph.remove_edge(u_mapped, v_mapped)

    # region: algo

    # Algorithm 1: K-core decomposition
    def k_core(self, k):
        subgraph = self.graph.copy()
        while True:
            nodes_to_remove = [node for node, degree in subgraph.degree if degree < k]
            if not nodes_to_remove:
                break
            subgraph.remove_nodes_from(nodes_to_remove)
        k_core_nodes = subgraph.nodes()
        return [self.reverse_mapping[node] for node in k_core_nodes]

    # Algorithm 2: Densest subgraph
    def densest_subgraph(self):
        S = "S"
        T = "T"

        m = self.graph.number_of_edges()
        n = self.graph.number_of_nodes()

        def construct_flow_network(g):
            flow_graph = nx.DiGraph()
            flow_graph.add_node(S)
            flow_graph.add_node(T)

            for u, v in self.graph.edges():
                flow_graph.add_edge(u, v, capacity=1)
                flow_graph.add_edge(v, u, capacity=1)

            for node, d_i in self.graph.degree:
                flow_graph.add_edge(S, node, capacity=m)
                flow_graph.add_edge(node, T, capacity=m + 2 * g - d_i)

            return flow_graph

        def find_min_cut(flow_graph):
            cut_value, partition = nx.minimum_cut(flow_graph, S, T)
            reachable, non_reachable = partition
            return reachable - {S}, non_reachable - {T}, cut_value

        l, u = 0, m
        xtol = 1 / (n * (n - 1))
        while u - l > xtol:
            g = (u + l) / 2
            flow_graph = construct_flow_network(g)
            V1, _, _ = find_min_cut(flow_graph)

            if len(V1) != 0:
                l = g
            else:
                u = g

        return [self.reverse_mapping[node] for node in V1]

    def approximate_densest_subgraph(self):
        def density(subgraph):
            if subgraph.number_of_nodes() == 0:
                return 0
            return subgraph.number_of_edges() / subgraph.number_of_nodes()

        best_subgraph = (None, 0)
        current_subgraph = self.graph.copy()

        while current_subgraph.number_of_nodes() > 0:
            best_subgraph = max(
                best_subgraph,
                (current_subgraph.copy(), density(current_subgraph)),
                key=lambda x: x[1],
            )

            min_degree_node = min(
                current_subgraph.nodes(), key=lambda node: self.graph.degree[node]
            )
            current_subgraph.remove_node(min_degree_node)

        assert best_subgraph[0] is not None
        densest_nodes = list(best_subgraph[0].nodes())
        return [self.reverse_mapping[node] for node in densest_nodes], best_subgraph[1]

    # Algorithm 3: K-clique decomposition
    def bron_kerbosch(self, r, p, x, cliques):
        if not p and not x:
            cliques.append(r)
            return
        for v in list(p):
            self.bron_kerbosch(
                r | {v},
                p & set(self.graph.neighbors(v)),
                x & set(self.graph.neighbors(v)),
                cliques,
            )
            p.remove(v)
            x.add(v)

    def find_maximal_cliques(self):
        cliques = []
        self.bron_kerbosch(set(), set(self.graph.nodes()), set(), cliques)
        return cliques

    def k_clique_decomposition(self, k):

        cliques = self.find_maximal_cliques()
        k_cliques = [clique for clique in cliques if len(clique) == k]
        return k_cliques

    # Algorithm 4: K-clique densest subgraph
    def k_clique_densest_subgraph(self, k, iterations=1000):
        # Step 1: Initialize node values
        r = {node: 0 for node in self.graph.nodes}

        for _ in range(iterations):
            # Step 2: Iterate over all nodes to set s
            s = r.copy()

            # Step 3: For each k-clique, update r
            for clique in self._find_k_cliques(k):
                min_node = min(clique, key=lambda node: s[node])
                r[min_node] += 1

        # Step 4: Normalize r by the number of iterations
        for node in r:
            node /= iterations

        # Step 5: Extract the densest subgraph
        densest_subgraph = self._extract_densest_subgraph(r, k)
        return densest_subgraph

    def _find_k_cliques(self, k):
        # Use networkx to find all k-cliques
        cliques = [clique for clique in nx.find_cliques(self.graph) if len(clique) == k]
        return cliques

    def _extract_densest_subgraph(self, r, k):
        sorted_nodes = sorted(r, key=r.get, reverse=True)
        subgraph_nodes = sorted_nodes[:k]
        subgraph = self.graph.subgraph(subgraph_nodes)
        return subgraph, nx.density(subgraph)

    # endregion

    def visualize(
        self,
        highlight_nodes=None,
        secondary_highlight_nodes=None,
        node_color="lightblue",
        node_size=20,
        edge_color="gray",
        with_labels=True,
    ):
        pos = nx.spring_layout(self.graph, k=0.3, iterations=100)

        if highlight_nodes:
            # Calculate the geometric center of the highlight nodes
            center_x = np.mean([pos[node][0] for node in highlight_nodes])
            center_y = np.mean([pos[node][1] for node in highlight_nodes])
            center = np.array([center_x, center_y])

            # Adjust highlight nodes to be closer to the center
            contraction_factor = 0.5
            for node in highlight_nodes:
                pos[node] = center + contraction_factor * (pos[node] - center)

            # Move non-highlight nodes away from the center
            expansion_factor = 1.5
            for node in set(self.graph.nodes()) - set(highlight_nodes):
                pos[node] = center + expansion_factor * (pos[node] - center)

        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color=edge_color),
            hoverinfo="none",
            mode="lines",
        )

        node_x, node_y = zip(*pos.values())
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if with_labels else "markers",
            marker=dict(size=node_size, color=node_color, line_width=2),
            text=list(self.graph.nodes()) if with_labels else None,
            hoverinfo="text" if with_labels else "none",
            textposition="top center" if with_labels else None,
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            ),
        )

        if highlight_nodes:
            highlight_x, highlight_y = zip(*[pos[node] for node in highlight_nodes])
            highlight_trace = go.Scatter(
                x=highlight_x,
                y=highlight_y,
                mode="markers+text" if with_labels else "markers",
                marker=dict(size=node_size, color="red", line_width=2),
                text=highlight_nodes if with_labels else None,
                hoverinfo="text" if with_labels else "none",
                textposition="top center" if with_labels else None,
            )
            fig.add_trace(highlight_trace)

        if secondary_highlight_nodes:
            secondary_highlight_x, secondary_highlight_y = zip(
                *[pos[node] for node in secondary_highlight_nodes]
            )
            secondary_highlight_trace = go.Scatter(
                x=secondary_highlight_x,
                y=secondary_highlight_y,
                mode="markers+text" if with_labels else "markers",
                marker=dict(size=node_size, color="green", line_width=2),
                text=secondary_highlight_nodes if with_labels else None,
                hoverinfo="text" if with_labels else "none",
                textposition="top center" if with_labels else None,
            )
            fig.add_trace(secondary_highlight_trace)

        # Show the plot in the browser
        fig.show()
