import argparse
import os
from graphmaster.graph import Graph


def generate_output_filename(input_file, suffix):
    base, ext = os.path.splitext(input_file)
    return f"{base}_{suffix}{ext}"


def main():
    parser = argparse.ArgumentParser(description="Graph processing and analysis.")
    parser.add_argument("input_file", type=str, help="Path to the input graph file.")
    parser.add_argument(
        "algorithm",
        type=str,
        choices=[
            "k_core",
            "densest_subgraph",
            "k_clique_decomposition",
            "k_clique_densest_subgraph",
        ],
        help="Algorithm to run.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Parameter k for k-core or k-clique algorithms.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for k-clique densest subgraph algorithm.",
    )
    parser.add_argument("--output_file", type=str, help="Path to the output file.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the graph and algorithm results.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "circular", "kamada_kawai", "random", "shell"],
        help="Layout for graph visualization.",
    )
    parser.add_argument(
        "--with_labels", action="store_true", help="Display labels on nodes."
    )

    args = parser.parse_args()

    graph = Graph()
    graph.load(args.input_file)

    if not args.output_file:
        args.output_file = generate_output_filename(args.input_file, args.algorithm)

    if args.algorithm == "k_core":
        k_core_nodes = graph.k_core(args.k)
        with open(args.output_file, "w") as f:
            for node in k_core_nodes:
                f.write(f"{node}\n")

    elif args.algorithm == "densest_subgraph":
        exact_densest_nodes = graph.densest_subgraph()
        approx_densest_nodes, approx_density = graph.approximate_densest_subgraph()
        with open(args.output_file, "w") as f:
            f.write("Exact Densest Subgraph Nodes:\n")
            for node in exact_densest_nodes:
                f.write(f"{node}\n")
            f.write("\nApproximate Densest Subgraph Nodes:\n")
            for node in approx_densest_nodes:
                f.write(f"{node}\n")
            f.write(f"\nApproximate Density: {approx_density}\n")

    elif args.algorithm == "k_clique_decomposition":
        k_cliques = graph.k_clique_decomposition(args.k)
        with open(args.output_file, "w") as f:
            for clique in k_cliques:
                f.write(" ".join(map(str, clique)) + "\n")

    elif args.algorithm == "k_clique_densest_subgraph":
        densest_subgraph, density = graph.k_clique_densest_subgraph(
            args.k, args.iterations
        )
        with open(args.output_file, "w") as f:
            f.write(f"Density: {density}\n")
            for node in densest_subgraph:
                f.write(f"{node}\n")

    if args.visualize:
        if args.algorithm in ["k_core"]:
            highlight_nodes = k_core_nodes
            graph.visualize(
                highlight_nodes=highlight_nodes,
                layout=args.layout,
                with_labels=args.with_labels,
            )
        elif args.algorithm == "densest_subgraph":
            graph.visualize(
                highlight_nodes=exact_densest_nodes,
                secondary_highlight_nodes=approx_densest_nodes,
                layout=args.layout,
                with_labels=args.with_labels,
            )
        else:
            graph.visualize(layout=args.layout, with_labels=args.with_labels)


if __name__ == "__main__":
    main()
