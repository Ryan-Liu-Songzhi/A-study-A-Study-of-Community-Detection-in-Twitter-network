# Social Network Analysis: Community Detection, Influence Diffusion, and Marketing ROI Strategy on Twitter Data

**Team Members:** Yuting Zhu, Songzhi Liu  
**Course:** Social Media Analytics, Spring 2025

## 1. Project Overview

This project presents a comprehensive end-to-end pipeline for social network analysis, focusing on community detection, influence analysis, information diffusion modeling, and culminating in a Return on Investment (ROI) analysis for marketing strategies. The study utilizes a standard Twitter dataset from the Stanford Network Analysis Project (SNAP) to uncover latent network properties, identify key influencers, compare diffusion models, and ultimately recommend cost-effective information propagation strategies.

The project implements and compares various algorithms and models, including Louvain and Label Propagation Algorithm (LPA) for community detection, and Independent Cascade (IC) and Linear Threshold (LT) models for information diffusion. A core contribution is the systematic ROI analysis framework, which evaluates different seed selection strategies (Top Influence, High Degree, High Betweenness, Random) varying seed set sizes (K) to maximize infection rates under resource constraints. Optional integration with Neo4j graph database for data persistence and querying is also included.
！![image](https://github.com/user-attachments/assets/a6fb4bba-b1e5-4ffe-b77e-3568c843afb4)

## 2. Key Features & Pipeline Components

The project follows a structured pipeline:

1.  **Data Loading & Initial Processing**:
    * Loads edge list data from the `twitter_combined.txt` file.
    * Initial graph construction (directed, then converted to undirected for most analyses).
2.  **Data Quality Checking & Preprocessing**:
    * `check_data_quality()`: Identifies and reports self-loops, duplicate edges, isolated nodes, and "supernodes" (nodes with excessively high degrees).
    * `preprocess_graph()`: Cleans data by removing self-loops, filtering nodes by degree, extracting the Largest Connected Component (LCC), and applying K-core decomposition to focus on the core network structure.
    * Sampling (`G_sample`): A subgraph is sampled from the processed graph for computationally intensive tasks like diffusion simulations and detailed centrality analysis.
3.  **Community Detection**:
    * **Louvain Algorithm**: Implemented using the `community_louvain` library, optimized for modularity.
    * **Label Propagation Algorithm (LPA)**: Custom implementation (`label_propagation()`) for fast community detection.
    * Comparison based on modularity score, number of communities, and execution time.
4.  **Centrality & Influence Analysis**:
    * Calculation of standard centrality measures: Degree, Betweenness (approximated), Closeness, and PageRank.
    * `influence_score()`: A composite influence metric combining weighted centrality scores (0.4*Degree + 0.3*Betweenness + 0.3*PageRank).
    * `get_top_influencers_from_scores()`: Identifies top K influencers based on the precomputed composite influence scores.
5.  **Information Diffusion Models**:
    * **Independent Cascade (IC) Model**: Custom implementation (`independent_cascade()`) simulating probabilistic information spread.
    * **Linear Threshold (LT) Model**: Custom implementation (`linear_threshold()`) simulating influence accumulation until a threshold is met.
6.  **Marketing ROI Analysis (Core Application)**:
    * Systematic comparison of different seed selection strategies:
        * `TopInfluence`: Seeds with the highest composite influence score.
        * `HighDegree`: Seeds with the highest degree.
        * `HighBetweenness`: Seeds with the highest betweenness centrality.
        * `Random`: Randomly selected seeds as a baseline.
    * Varying seed set sizes (K).
    * Simulation of IC and LT models for each strategy and K value, repeated (multiple) times for robustness.
    * Collection and aggregation of results (average infection rate, standard deviation).
    * Calculation of supplementary ROI metrics (e.g., `(avg_infected_nodes - k_seeds) / k_seeds`) and marginal benefits.
7.  **Data Persistence **:
    * `Neo4jConnection` class for interacting with a Neo4j graph database (controlled by `USE_NEO4J` flag).
    * Stores graph structure, node attributes (community, centrality, influence, infection status).
8.  **Visualization**:
    * Utilizes Matplotlib for various plots:
        * Community structures.
        * Degree distribution histograms.
        * Information spread visualization (`plot_spread()`).
        * **Cost-Benefit Curves for ROI Analysis**: Plotting average infection rate vs. number of seed nodes (K) for different strategies and models.
        * Plots for supplementary ROI metrics and marginal benefits.
    * Neo4jConnection class, designed for scalable graph data storage and querying.
9.  **Output**:
    * Generated CSV files for aggregated ROI results, centrality measures, influencer lists, and infected node sets.
    * Generated PNG image files for all visualizations.

## 3. Dataset

* **Source**: SNAP (Stanford Network Analysis Project) - [Twitter social graph](https://snap.stanford.edu/data/ego-Twitter.html) (`twitter_combined.txt`).
* **Description**: This dataset represents a network of Twitter users, where nodes are users and edges represent "follows" relationships. The original dataset contains 81,306 nodes and 1,768,149 edges.
* **Note**: The raw `twitter_combined.txt` dataset is **NOT** included in this repository due to its size. You need to download it from the SNAP website and place it in the path specified in the script.
* **Processed Data**: The script performs significant preprocessing and sampling, with the core ROI analysis typically run on a smaller subgraph (`G_sample`, e.g., 294 nodes in one of the test runs).

## 4. Prerequisites & Dependencies

* Python 3.9+ (recommended)
* Required Python libraries (see `requirements.txt` for specific versions):
    * `networkx`
    * `python-louvain` (or `community`)
    * `pandas`
    * `matplotlib`
    * `numpy` (usually a dependency of pandas/matplotlib)
    * `neo4j`
 
## 5. Setup and Running the Project

### 5.1 Clone the Repository

```bash
git clone https://github.com/Ryan-Liu-Songzhi/A-study-A-Study-of-Community-Detection-in-Twitter-network.git
cd A-study-A-Study-of-Community-Detection-in-Twitter-network
```
### 5.2 Download the Dataset
- **Download the `twitter_combined.txt` file** from the SNAP Twitter dataset page.
- **Place the downloaded file** in the location expected by the script, or update the `file_path` variable within the `[CommunityDetection_Final.py]` script (around line 240 in the provided version).
- **Example path in script**: `file_path = r"C:\Path\To\Your\Social Media FinalProject\twitter_combined.txt"`

### 5.3 Configure Neo4j (Optional)
If you want to use the Neo4j integration (`USE_NEO4J = True` in the script, default is `False` in the provided code):
- Ensure you have a Neo4j instance running.
- Update the Neo4j connection details (URI, USERNAME, PASSWORD) at the beginning of the `CommunityDetection_Final.py` script (around lines 13-15):
```bash
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "your_neo4j_password" # IMPORTANT: Replace with your actual password
```
- **Security Warning**: It is strongly recommended **NOT to commit actual passwords to a public GitHub repository**. Use environment variables or a local configuration file (added to `.gitignore`) for managing sensitive credentials in a real-world scenario. For this academic project, if you keep it, clearly state it's a placeholder or for local testing only.

## 5.4 Review Script Parameters
At the beginning of `CommunityDetection_Final.py`, you can find parameters like:
- `top_n` (for community selection)
- `sample_max` (for `G_sample` size)
- `betweenness_k` (for betweenness approximation)
- `run_diffusion_models`
- The ROI experiment `config` dictionary
Adjust these as needed.

## 5.5 Execute the Script
- cmd- python CommunityDetection_05221.py

# 6. Expected Output
Running the script will:
- Print analysis progress and key findings to the console.
- Generate and save several CSV files in the script's directory, including:
  - `roi_aggregated_results_optimized_YYYYMMDD_HHMMSS.csv`: The main results of the ROI analysis.
  - `centrality_results_YYYYMMDD_HHMMSS.csv`: Centrality measures for `G_sample`.
  - `influencer_communities_YYYYMMDD_HHMMSS.csv`: Information about top influencers.
  - `infected_nodes_ic_YYYYMMDD_HHMMSS.csv` and `infected_nodes_lt_YYYYMMDD_HHMMSS.csv`: Nodes infected by IC/LT models in a specific run (if `run_diffusion_models` is `True` and that part of the code is active).
  - `model_comparison_YYYYMMDD_HHMMSS.csv`: Comparison of IC and LT spread from a fixed set of top influencers.
- Generate and save several PNG image files for visualizations in the script's directory or a sub-directory (e.g., `roi_plots_final/`), including:
  - Community structure plots.
  - Degree distribution histograms.
  - Information spread visualizations.
  - Cost-Benefit curves for ROI analysis.
  - Plots for supplementary ROI metrics and marginal benefits.
