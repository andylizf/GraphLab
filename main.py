import argparse
import os
import time
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
            "k_vcc",
        ],
        help="Algorithm to run.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Parameter k for k-core or k-clique algorithms.",
    )
    parser.add_argument("--output_file", type=str, help="Path to the output file.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the graph and algorithm results.",
    )
    parser.add_argument(
        "--with-labels", action="store_true", help="Display labels on nodes."
    )

    args = parser.parse_args()

    graph = Graph()
    graph.load(args.input_file)

    if not args.output_file:
        args.output_file = generate_output_filename(args.input_file, args.algorithm)

    if args.algorithm == "k_core":
        start_time = time.time()
        k_core_nodes = graph.k_core(args.k)
        elapsed_time = time.time() - start_time

        with open(args.output_file, "w") as f:
            f.write(f"{elapsed_time:.4f}s\n")
            f.write(f"{args.k}-core Subgraph:\n")
            f.write(" ".join(map(str, k_core_nodes)) + "\n")

    elif args.algorithm == "densest_subgraph":
        start_time = time.time()
        exact_densest_nodes, exact_density = graph.densest_subgraph()
        elapsed_time = time.time() - start_time

        start_time_approx = time.time()
        approx_densest_nodes, approx_density = graph.approximate_densest_subgraph()
        elapsed_time_approx = time.time() - start_time_approx

        with open(args.output_file, "w") as f:
            f.write(f"Exact Densest Subgraph:\n")
            f.write(f"{elapsed_time:.4f}s\n")
            f.write(f"density: {exact_density}\n")
            f.write(" ".join(map(str, exact_densest_nodes)) + "\n")

            f.write(f"Approximate Densest Subgraph:\n")
            f.write(f"{elapsed_time_approx:.4f}s\n")
            f.write(f"density: {approx_density}\n")
            f.write(" ".join(map(str, approx_densest_nodes)) + "\n")

    elif args.algorithm == "k_clique_decomposition":
        start_time = time.time()
        k_cliques = graph.k_clique_decomposition(args.k)
        elapsed_time = time.time() - start_time

        with open(args.output_file, "w") as f:
            f.write(f"{elapsed_time:.4f}s\n")
            f.write(f"{args.k}-clique Decomposition:\n")
            for clique in k_cliques:
                f.write(" ".join(map(str, clique)) + "\n")

    elif args.algorithm == "k_vcc":
        start_time = time.time()
        k_vccs = graph.k_vcc(args.k)
        elapsed_time = time.time() - start_time

        with open(args.output_file, "w") as f:
            f.write(f"{elapsed_time:.4f}s\n")
            f.write(f"{args.k}-VCCs:\n")
            for k_vcc in k_vccs:
                f.write(" ".join(map(str, k_vcc)) + "\n")

    if args.visualize:
        if args.algorithm == "k_core":
            highlight_nodes = k_core_nodes
            graph.visualize(
                highlight_nodes=highlight_nodes,
                with_labels=args.with_labels,
            )
        elif args.algorithm == "densest_subgraph":
            graph.visualize(
                highlight_nodes=exact_densest_nodes,
                secondary_highlight_nodes=approx_densest_nodes,
                with_labels=args.with_labels,
            )
        elif args.algorithm == "k_clique_decomposition":
            graph.visualize(
                highlight_cliques=k_cliques,
                with_labels=args.with_labels,
            )
        elif args.algorithm == "k_vcc":
            graph.visualize(
                highlight_cliques=k_vccs,
                with_labels=args.with_labels,
            )
        else:
            graph.visualize(with_labels=args.with_labels)


if __name__ == "__main__":
    main()
