import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)

FN_NODE_METRICS = os.path.join(OUT_DIR, "node_metrics.csv")
FN_NODE_METRICS_COMM = os.path.join(OUT_DIR, "node_metrics_with_communities.csv")
FN_TOP_NODES = os.path.join(OUT_DIR, "top_nodes.txt")
FN_SPRING = os.path.join(OUT_DIR, "spring_layout.png")
FN_CIRCULAR = os.path.join(OUT_DIR, "circular_layout.png")
FN_COMM = os.path.join(OUT_DIR, "communities.png")
FN_TOP_NODES_IMG = os.path.join(OUT_DIR, "top_nodes.png")
FN_DEGREE_HIST = os.path.join(OUT_DIR, "degree_hist.png")
FN_GEXF = os.path.join(OUT_DIR, "graph.gexf")
FN_README = os.path.join(OUT_DIR, "README_generated.txt")


def load_graph_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if {'source', 'target'}.issubset(df.columns):
            G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph())
            print(f"Loaded graph from {csv_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            return G
        else:
            print(
                "CSV exists but required columns 'source' and 'target' not found. Expected columns exactly named 'source' and 'target'.")
            return None
    except Exception as e:
        print("Failed to load CSV:", e)
        return None


def generate_synthetic_graph(n=800, m=2):
    print(f"Generating Barabasi-Albert graph with n={n}, m={m}")
    return nx.barabasi_albert_graph(n, m, seed=SEED)


def girvan_newman_communities(G, k):
    comp_gen = nx.community.girvan_newman(G)
    for communities in itertools.islice(comp_gen, 0, G.number_of_nodes()):
        if len(communities) >= k:
            return [list(c) for c in communities]
    return [list(c) for c in communities]


def main():
    csv_path = os.path.join(BASE_DIR, "links.csv")
    if os.path.exists(csv_path):
        G = load_graph_from_csv(csv_path)
        if G is None:
            G = generate_synthetic_graph()
    else:
        G = generate_synthetic_graph()

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    degree_dict = dict(G.degree())
    print("Computing betweenness centrality (this may take time on large graphs)...")
    betweenness = nx.betweenness_centrality(G, normalized=True)
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, alpha=0.85)

    df = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [degree_dict[n] for n in G.nodes()],
        "betweenness": [betweenness[n] for n in G.nodes()],
        "clustering": [clustering[n] for n in G.nodes()],
        "pagerank": [pagerank[n] for n in G.nodes()]
    })
    df = df.sort_values(by="degree", ascending=False)
    df.to_csv(FN_NODE_METRICS, index=False)
    print("Saved node metrics to", FN_NODE_METRICS)

    try:
        communities = girvan_newman_communities(G, k=4)
        node_community = {}
        for cid, comm in enumerate(communities):
            for n in comm:
                node_community[n] = cid
    except Exception as e:
        print("Girvan-Newman failed or too slow; falling back to connected components as communities.", e)
        node_community = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for n in comp:
                node_community[n] = i

    df["community"] = df["node"].map(node_community)
    df.to_csv(FN_NODE_METRICS_COMM, index=False)
    print("Saved node metrics with communities to", FN_NODE_METRICS_COMM)

    top_degree = df.nlargest(10, "degree")[["node", "degree"]]
    top_betw = df.nlargest(10, "betweenness")[["node", "betweenness"]]
    top_pr = df.nlargest(10, "pagerank")[["node", "pagerank"]]
    with open(FN_TOP_NODES, "w") as f:
        f.write("Top 10 by degree:\n")
        f.write(top_degree.to_string(index=False))
        f.write("\n\nTop 10 by betweenness:\n")
        f.write(top_betw.to_string(index=False))
        f.write("\n\nTop 10 by pagerank:\n")
        f.write(top_pr.to_string(index=False))
    print("Saved top nodes summary to", FN_TOP_NODES)

    nx.write_gexf(G, FN_GEXF)
    print("Saved GEXF to", FN_GEXF)

    pos_spring = nx.spring_layout(G, seed=SEED)
    pos_circular = nx.circular_layout(G)

    sizes = [(degree_dict[n] + 1) * 20 for n in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos_spring, node_size=sizes, node_color='lightgray')
    nx.draw_networkx_edges(G, pos_spring, alpha=0.3, width=0.5)
    plt.title("Figure 5 – Force-directed (spring) layout (placeholder)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(FN_SPRING, dpi=200)
    plt.close()
    print("Saved spring layout to", FN_SPRING)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos_circular, node_size=sizes, node_color='lightgray')
    nx.draw_networkx_edges(G, pos_circular, alpha=0.3, width=0.5)
    plt.title("Figure 9 – Circular layout (placeholder)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(FN_CIRCULAR, dpi=200)
    plt.close()
    print("Saved circular layout to", FN_CIRCULAR)

    plt.figure(figsize=(10, 8))
    communities_list = [node_community.get(n, 0) for n in G.nodes()]
    nx.draw_networkx(G, pos_spring, node_size=sizes, node_color=communities_list, cmap=plt.cm.get_cmap('tab10'),
                     with_labels=False)
    plt.title("Figure 4 – Community structure (placeholder)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(FN_COMM, dpi=200)
    plt.close()
    print("Saved communities image to", FN_COMM)

    top5_betw = df.nlargest(5, "betweenness")["node"].tolist()
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos_spring, node_size=sizes, node_color='lightgray')
    nx.draw_networkx_edges(G, pos_spring, alpha=0.2)
    nx.draw_networkx_nodes(G, pos_spring, nodelist=top5_betw, node_color='red',
                           node_size=[(degree_dict[n] + 1) * 80 for n in top5_betw])
    plt.title("Figure 6 – Highlighted key nodes (top-5 betweenness) (placeholder)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(FN_TOP_NODES_IMG, dpi=200)
    plt.close()
    print("Saved top nodes highlight to", FN_TOP_NODES_IMG)

    plt.figure(figsize=(8, 5))
    plt.hist(list(degree_dict.values()), bins=30)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Figure 7 – Degree distribution histogram (placeholder)")
    plt.tight_layout()
    plt.savefig(FN_DEGREE_HIST, dpi=150)
    plt.close()
    print("Saved degree histogram to", FN_DEGREE_HIST)

    with open(FN_README, "w") as f:
        f.write("Generated on: " + datetime.now().isoformat() + "\n")
        f.write(f"Nodes: {n_nodes}, Edges: {n_edges}\n")
        f.write("Files produced:\n")
        for fn in [FN_NODE_METRICS, FN_NODE_METRICS_COMM, FN_TOP_NODES, FN_GEXF, FN_SPRING, FN_CIRCULAR, FN_COMM,
                   FN_TOP_NODES_IMG, FN_DEGREE_HIST]:
            f.write(" - " + os.path.basename(fn) + "\n")
    print("Saved README summary to", FN_README)
    print("All done. Outputs saved in", OUT_DIR)


if __name__ == "__main__":
    main()
