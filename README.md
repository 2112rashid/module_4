# Module Assignment 4 — Network Analysis & Visualization of Web Graph Data

## Overview
This repository contains a reproducible Python pipeline for **analyzing and visualizing a web-like network** (nodes = pages/users, edges = hyperlinks/interactions). It was prepared to match the uploaded assignment *"Module Assignment №4"* and includes the main analysis script, generated outputs (figures & CSVs), and guidance for using your own edge-list data.

## Project structure
```
WebGraph_ModuleAssignment4/
├── analysis_visualization.py 
├── requirements.txt 
├── README.md 
└── figures/
├── Figure_1_workflow.png
├── Figure_2_example_construction.png
├── Figure_3_degree_distribution_loglog.png
├── Figure_4_community_structure.png
├── Figure_5_force_directed.png
├── Figure_6_highlight_key_nodes.png
├── Figure_7_degree_histogram.png
├── Figure_8_top_nodes_table.png
├── Figure_9_layout_comparison.png
├── top_nodes_by_centrality.csv
└── node_metrics_full.csv
```
> **Note:** The `figures/` folder is created and populated after running `analysis_visualization.py`.

## Requirements
- Python 3.8+  
- Install dependencies:
```bash
pip install -r requirements.txt
# or
pip install networkx matplotlib numpy pandas
