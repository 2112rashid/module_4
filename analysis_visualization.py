import os
import math
import random
from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_NODES = 800
M_ATTACH = 3

def save_fig(fig, name, dpi=200, bbox_inches='tight'):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved {path}")

def set_title_and_axis(ax, title):
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def figure_1_workflow():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    boxes = [
        ("Data Collection\n(crawls, SNAP, logs)", (0.05, 0.65)),
        ("Graph Construction\n(nodes, directed edges)", (0.35, 0.65)),
        ("Network Analysis\n(degree, centrality,\nclustering)", (0.65, 0.65)),
        ("Community Detection\n(Greedy modularity)", (0.35, 0.25)),
        ("Visualization\n(spring, circular)\n+ Reports", (0.65, 0.25)),
    ]
    for text, (x, y) in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.25, 0.18, fill=True, color='whitesmoke', ec='black', lw=0.8))
        ax.text(x+0.125, y+0.09, text, ha='center', va='center', fontsize=9)

    arrow_props = dict(arrowstyle="->", color='black', lw=1)
    ax.annotate("", xy=(0.3, 0.74), xytext=(0.3, 0.84), arrowprops=arrow_props)
    ax.annotate("", xy=(0.6, 0.74), xytext=(0.6, 0.84), arrowprops=arrow_props)
    ax.annotate("", xy=(0.47, 0.53), xytext=(0.47, 0.65), arrowprops=arrow_props)
    ax.annotate("", xy=(0.8, 0.36), xytext=(0.6, 0.36), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1 – Workflow of web graph analysis and visualization", fontsize=11)
    save_fig(fig, "Figure_1_workflow.png")
    plt.close(fig)

def build_graph(n=N_NODES, m=M_ATTACH):
    G = nx.barabasi_albert_graph(n, m, seed=SEED)
    DG = nx.DiGraph()
    for u, v in G.edges():
        if random.random() < 0.5:
            DG.add_edge(u, v)
        else:
            DG.add_edge(v, u)
    return DG

def figure_2_example_construction():
    toy = nx.DiGraph()
    toy.add_edges_from([
        ("page_A", "page_B"),
        ("page_B", "page_C"),
        ("page_A", "page_C"),
        ("page_C", "page_D"),
        ("page_D", "page_A"),
        ("page_E", "page_C"),
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    pos = nx.spring_layout(toy, seed=SEED)
    nx.draw_networkx_nodes(toy, pos, node_size=700, ax=ax)
    nx.draw_networkx_labels(toy, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(toy, pos, arrows=True, arrowstyle='->', ax=ax)
    ax.set_title("Figure 2 – Example graph construction from link data")
    ax.axis('off')
    save_fig(fig, "Figure_2_example_construction.png")
    plt.close(fig)

def compute_metrics(G):
    print("Computing metrics...")
    metrics = {}
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg = dict(G.degree())
    metrics['in_degree'] = in_deg
    metrics['out_degree'] = out_deg
    metrics['degree'] = deg

    metrics['betweenness'] = nx.betweenness_centrality(G)
    metrics['closeness'] = nx.closeness_centrality(G.to_undirected())
    metrics['clustering'] = nx.clustering(G.to_undirected())

    try:
        metrics['pagerank'] = nx.pagerank(G, alpha=0.85)
    except Exception as e:
        print("PageRank failed, using fallback:", e)
        metrics['pagerank'] = {}

    return metrics

def figure_3_degree_distribution_loglog(G):
    deg_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degrees = np.array(deg_sequence)
    counts = np.bincount(degrees)
    nonzero_idx = np.nonzero(counts)[0]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.loglog(nonzero_idx, counts[nonzero_idx], marker='o', linestyle='none')
    ax.set_xlabel("Degree k (log)")
    ax.set_ylabel("Number of nodes with degree k (log)")
    ax.set_title("Figure 3 – Degree distribution (log-log plot)")
    save_fig(fig, "Figure_3_degree_distribution_loglog.png")
    plt.close(fig)

def figure_7_degree_histogram(G):
    degrees = [d for n, d in G.degree()]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(degrees, bins=30)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title("Figure 7 – Degree distribution histogram")
    save_fig(fig, "Figure_7_degree_histogram.png")
    plt.close(fig)

def detect_communities(G):
    print("Detecting communities (greedy modularity)...")
    und = G.to_undirected()
    communities = list(nx.algorithms.community.greedy_modularity_communities(und))
    node2comm = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node2comm[n] = i
    return communities, node2comm

def figure_4_community_structure(G, communities, node2comm, max_display=200):
    if G.number_of_nodes() > max_display:
        nodes_sample = list(islice(G.nodes(), max_display))
        sub = G.subgraph(nodes_sample).copy()
    else:
        sub = G

    comm_indices = [node2comm.get(n, -1) for n in sub.nodes()]
    unique_comms = sorted(set(comm_indices))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) if i >=0 else (0.5,0.5,0.5) for i in comm_indices]

    fig, ax = plt.subplots(figsize=(7,6))
    pos = nx.spring_layout(sub, seed=SEED)
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=60, ax=ax)
    nx.draw_networkx_edges(sub, pos, alpha=0.4, ax=ax)
    ax.set_title("Figure 4 – Community structure of the web graph (sample/subgraph)")
    ax.axis('off')
    save_fig(fig, "Figure_4_community_structure.png")
    plt.close(fig)

def figure_5_force_directed(G):
    fig, ax = plt.subplots(figsize=(8,8))
    pos = nx.spring_layout(G, seed=SEED, k=None, iterations=100)
    nx.draw_networkx_nodes(G, pos, node_size=20, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=False, ax=ax)
    ax.set_title("Figure 5 – Visualization of the web graph with force-directed layout")
    ax.axis('off')
    save_fig(fig, "Figure_5_force_directed.png")
    plt.close(fig)

def figure_6_highlight_key_nodes(G, metrics, top_n=5):
    if metrics.get('pagerank'):
        ranking = sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)
        metric_name = "PageRank"
    else:
        ranking = sorted(metrics['betweenness'].items(), key=lambda x: x[1], reverse=True)
        metric_name = "Betweenness"
    top_nodes = [n for n, v in ranking[:top_n]]
    print(f"Top {top_n} nodes by {metric_name}:", top_nodes)

    fig, ax = plt.subplots(figsize=(8,8))
    pos = nx.spring_layout(G, seed=SEED, iterations=80)

    nx.draw_networkx_nodes(G, pos, node_size=20, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax, arrows=False)

    nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_size=200, node_color='red', ax=ax)
    labels = {n: str(n) for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color='black', ax=ax)
    ax.set_title(f"Figure 6 – Highlighted key nodes in the web graph (top {top_n} by {metric_name})")
    ax.axis('off')
    save_fig(fig, "Figure_6_highlight_key_nodes.png")
    plt.close(fig)

def figure_8_table_top_nodes(metrics, top_k=10):
    nodes = list(metrics['degree'].keys())
    df = pd.DataFrame({
        'node': nodes,
        'degree': [metrics['degree'][n] for n in nodes],
        'in_degree': [metrics['in_degree'][n] for n in nodes],
        'out_degree': [metrics['out_degree'][n] for n in nodes],
        'betweenness': [metrics['betweenness'][n] for n in nodes],
        'clustering': [metrics['clustering'][n] for n in nodes],
        'closeness': [metrics['closeness'][n] for n in nodes],
        'pagerank': [metrics.get('pagerank', {}).get(n, 0.0) for n in nodes]
    })
    if 'pagerank' in df.columns and df['pagerank'].sum() > 0:
        df_sorted = df.sort_values('pagerank', ascending=False)
        rank_by = 'pagerank'
    else:
        df_sorted = df.sort_values('betweenness', ascending=False)
        rank_by = 'betweenness'

    top_df = df_sorted.head(top_k).reset_index(drop=True)
    csv_path = os.path.join(OUT_DIR, "top_nodes_by_centrality.csv")
    top_df.to_csv(csv_path, index=False)
    print(f"Saved top nodes CSV to {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * top_k))
    ax.axis('off')
    ax.set_title("Figure 8 – Table of top nodes by centrality score (CSV saved)", fontsize=11)
    table = ax.table(cellText=top_df.values.round(6).astype(str),
                     colLabels=top_df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    save_fig(fig, "Figure_8_top_nodes_table.png")
    plt.close(fig)

def figure_9_layout_comparison(G, sample_n=400):
    # For speed/sample if graph large
    if G.number_of_nodes() > sample_n:
        nodes_sample = list(islice(G.nodes(), sample_n))
        sub = G.subgraph(nodes_sample).copy()
    else:
        sub = G

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    pos1 = nx.spring_layout(sub, seed=SEED, iterations=80)
    nx.draw(sub, pos=pos1, ax=axes[0], node_size=30, with_labels=False, arrows=False)
    axes[0].set_title("Spring (force-directed)")

    pos2 = nx.circular_layout(sub)
    nx.draw(sub, pos=pos2, ax=axes[1], node_size=30, with_labels=False, arrows=False)
    axes[1].set_title("Circular layout")

    fig.suptitle("Figure 9 – Comparison of visualization layouts (spring vs circular)")
    save_fig(fig, "Figure_9_layout_comparison.png")
    plt.close(fig)

def main():
    figure_1_workflow()
    figure_2_example_construction()

    print(f"Building Barabasi-Albert graph with n={N_NODES}, m={M_ATTACH} ...")
    G = build_graph(n=N_NODES, m=M_ATTACH)
    print(f"Graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    metrics = compute_metrics(G)

    figure_3_degree_distribution_loglog(G)

    communities, node2comm = detect_communities(G)
    print(f"Detected {len(communities)} communities (greedy modularity)")
    figure_4_community_structure(G, communities, node2comm)
    figure_5_force_directed(G)
    figure_6_highlight_key_nodes(G, metrics, top_n=5)
    figure_7_degree_histogram(G)
    figure_8_table_top_nodes(metrics, top_k=10)
    figure_9_layout_comparison(G, sample_n=500)

    nodes = list(G.nodes())
    df_all = pd.DataFrame({
        'node': nodes,
        'degree': [metrics['degree'][n] for n in nodes],
        'in_degree': [metrics['in_degree'][n] for n in nodes],
        'out_degree': [metrics['out_degree'][n] for n in nodes],
        'betweenness': [metrics['betweenness'][n] for n in nodes],
        'clustering': [metrics['clustering'][n] for n in nodes],
        'closeness': [metrics['closeness'][n] for n in nodes],
        'pagerank': [metrics.get('pagerank', {}).get(n, 0.0) for n in nodes]
    })
    df_all.to_csv(os.path.join(OUT_DIR, "node_metrics_full.csv"), index=False)
    print("Saved full node metrics to figures/node_metrics_full.csv")

    print("All figures (Figure_1 ... Figure_9) and CSV outputs were created inside the 'figures' folder.")

if __name__ == "__main__":
    main()
