# Social Media Marketing ROI Analysis

This repository contains the final report and supporting files for a social network analysis project based on a Twitter dataset. The project evaluates marketing return on investment (ROI) by analyzing community structures, influencer identification, and influence propagation models.

## Project Overview
- **Objective**: Analyze Twitter network data to optimize marketing strategies using community detection, influence scoring, and diffusion models (IC and LT).
- **Dataset**: A sampled Twitter network (`G_sample`) with 294 nodes and 6,354 edges, derived from `twitter_combined.txt`.
- **Tools**: Python (NetworkX, Matplotlib), with Neo4j integration planned for future scalability.

## Files
- **Social_Media_FinalProject_Report.pdf**: The complete project report (6.8-6.9 pages), detailing methodology, results, and evaluation.
- **ic_model_spread_20250522_210951.png**: Visualization of the IC model's influence spread (191 infected nodes).
- **lt_model_spread_20250522_210951.png**: Visualization of the LT model's influence spread (67 activated nodes).
- **community_visualization_20250522_210951.png**: Community structure of the sampled network.
- **degree_distribution_hist.png**: Histogram of node degree distribution.
- **[Other code/visualization files]**: Include additional files as needed (e.g., `roi_analysis.png`).

## Key Findings
- Louvain algorithm outperformed LPA in community detection (modularity 0.256 vs. 0.109).
- IC model achieved broader influence (65% coverage) than LT (23% activation).
- TopInfluence strategy, leveraging influencers like node 40981798, was most effective.

## Installation and Usage
1. Clone the repository:
