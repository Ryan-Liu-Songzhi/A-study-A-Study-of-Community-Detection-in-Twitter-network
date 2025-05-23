import networkx as nx
from community import community_louvain
from pandas import Timestamp
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
import random
import pandas as pd
from datetime import datetime
import warnings
import time
from neo4j import GraphDatabase
import networkx.algorithms.community as nx_comm
from collections import defaultdict
import traceback

warnings.filterwarnings("ignore", category=UserWarning)

# global variable
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "password123"  # Replace with your password

USE_NEO4J = False  # Controls whether or not to enable Neo4j
# Neo4jConnection class
class Neo4jConnection:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=30)
            print(f"✅ Connected to Neo4j at {uri}")
            with self.driver.session() as session:
                session.run("RETURN 1 AS test")
            print("✅ Connection test successful")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j at {uri}: {str(e)}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            print("✅ Neo4j connection closed")

    def add_graph(self, G, centrality_df=None, partition=None):
        with self.driver.session() as session:
            # Store nodes and attributes (in batches)
            batch_size = 1000
            total_nodes = G.number_of_nodes()
            print(f"Starting to store {total_nodes} nodes")
            nodes_list = list(G.nodes())
            for i in range(0, total_nodes, batch_size):
                batch_nodes = nodes_list[i:i + batch_size]
                for node in batch_nodes:
                    node_data = {'id': int(node)}
                    if centrality_df is not None:
                        node_row = centrality_df[centrality_df['node'] == node]
                        if not node_row.empty:
                            node_data['degree'] = float(node_row['degree'].iloc[0])
                            node_data['is_influencer'] = node_data['degree'] > centrality_df['degree'].quantile(0.95)
                    if partition is not None:
                        node_data['community'] = int(partition.get(node, -1))
                    session.run(
                        "CREATE (n:Node {id: $id, degree: $degree, community: $community, is_influencer: $is_influencer})",
                        id=node_data['id'],
                        degree=node_data.get('degree', 0.0),
                        community=node_data.get('community', -1),
                        is_influencer=node_data.get('is_influencer', False)
                    )
                print(f"Stored batch {i//batch_size + 1} of {total_nodes//batch_size + 1} node batches")

                # Verify that the node is created
                created_nodes = session.run("MATCH (n:Node) RETURN count(n) AS count")
                count = created_nodes.single()["count"]
                print(f"Total nodes in Neo4j after batch {i//batch_size + 1}: {count}")

        # Store edges (in batches)
        batch_size = 1000
        total_edges = G.number_of_edges()
        print(f"Starting to store {total_edges} edges")
        edges_list = list(G.edges())
        for i in range(0, total_edges, batch_size):
            batch_edges = edges_list[i:i + batch_size]
            with self.driver.session() as session:
                tx = session.begin_transaction()
                try:
                    # Batch Creation of Edges with UNWIND
                    edge_data = [{"source": int(edge[0]), "target": int(edge[1])} for edge in batch_edges]
                    tx.run(
                        """
                        UNWIND $edges AS edge
                        MATCH (a:Node {id: edge.source}), (b:Node {id: edge.target})
                        CREATE (a)-[:FOLLOWS]->(b)
                        """,
                        edges=edge_data
                    )
                    tx.commit()
                    print(f"Stored batch {i//batch_size + 1} of {total_edges//batch_size + 1} edge batches")
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    tx.close()                    
                    # Check and record missing nodes
                    for edge in batch_edges:
                        source_exists = session.run(
                            "MATCH (n:Node {id: $id}) RETURN count(n) AS count",
                            id=int(edge[0])
                        ).single()["count"]
                        target_exists = session.run(
                            "MATCH (n:Node {id: $id}) RETURN count(n) AS count",
                            id=int(edge[1])
                        ).single()["count"]
                        if source_exists == 0 or target_exists == 0:
                            print(f"Skipping edge {edge}: Source or target node not found")
                finally:
                    tx.close()

        print("✅ Graph and attributes stored in Neo4j")

    def query_high_degree_nodes(self, community_id=None, limit=5):
        with self.driver.session() as session:
            query = (
                "MATCH (n:Node) "
                "WHERE ($community_id IS NULL OR n.community = $community_id) "
                "RETURN n.id AS id, n.degree AS degree, n.community AS community "
                "ORDER BY n.degree DESC LIMIT $limit"
            )
            result = session.run(
                query,
                community_id=community_id,
                limit=limit
            )
            return [(record["id"], record["degree"], record["community"]) for record in result]

    def update_infected_nodes(self, infected_nodes):
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $infected_nodes AS node_id
                MATCH (n:Node {id: node_id})
                SET n.is_infected = true
                """,
                infected_nodes=list(infected_nodes)
            )

    def add_infection_relationships(self, infection_pairs):
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $infection_pairs AS pair
                MATCH (source:Node {id: pair[0]}), (target:Node {id: pair[1]})
                CREATE (source)-[:INFECTED_BY]->(target)
                """,
                infection_pairs=infection_pairs
            )

# Initialize neo4j_conn
if USE_NEO4J:
    neo4j_conn = Neo4jConnection(URI, USERNAME, PASSWORD)
else:
    neo4j_conn = None
    print("Neo4j connection disabled")
    
#--------Neo4j connection class-------

#-----------Data quality checks (check_data_quality function)--------------
def check_data_quality(G):
    """Basic data quality checks for the graph"""
    print("\n🔍 Data quality check:")
    selfloops = list(nx.selfloop_edges(G))
    duplicates = len(G.edges()) - len(set(G.edges()))
    isolated = list(nx.isolates(G))
    degrees = dict(G.degree())
    outliers = [n for n, d in degrees.items() if d > 1000]
    print(f"- Number of self-loops: {len(selfloops)}")
    print(f"- Number of duplicate edges: {duplicates}")
    print(f"- Number of isolated nodes: {len(isolated)}")
    print(f"- Super nodes (degree > 1000): {len(outliers)}")
    return selfloops, isolated, outliers

def preprocess_graph(G, min_degree=1, max_degree=500):
    """Integrated preprocessing: denoising, filtering anomalies, K-core"""
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.Graph(G)  # Remove weights
    degrees = dict(G.degree())
    nodes_to_keep = [n for n, d in degrees.items() if min_degree <= d <= max_degree]
    G = G.subgraph(nodes_to_keep).copy()
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    return nx.k_core(G, k=min_degree)

def influence_score(node_id, precomputed_degree_cent_dict, precomputed_betweenness_cent_dict, precomputed_pagerank_dict):
    
    deg = precomputed_degree_cent_dict.get(node_id, 0.0) # Use .get to provide a default value of 0.0 in case the node is not in the dictionary
    bet = precomputed_betweenness_cent_dict.get(node_id, 0.0)
    pr = precomputed_pagerank_dict.get(node_id, 0.0)
    return 0.4 * deg + 0.3 * bet + 0.3 * pr

def get_top_influencers(degree_dict, betweenness_dict, pagerank_dict, n=10):
    """Get the n nodes with the highest influence"""
    scores = {node: influence_score(node, degree_dict, betweenness_dict, pagerank_dict) for node in degree_dict}
    return sorted(scores, key=scores.get, reverse=True)[:n]

def independent_cascade(graph, seeds, prob=0.1, steps=10, verbose=False): # Verbose parameter retained, but internal printing removed
    """Independent Cascade Model"""
    if not graph or not seeds:
        return set()
    
    valid_seeds = {s for s in seeds if s in graph}
    if not valid_seeds:
        return set()

    current_wave = set(valid_seeds)
    all_activated_nodes = set(valid_seeds)

    for _ in range(steps): # The iteration variable step is no longer used
        newly_activated_this_step = set()
        if not current_wave:
            break

        for node in current_wave:
            if node not in graph:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in all_activated_nodes and random.random() < prob: # Use prob
                    newly_activated_this_step.add(neighbor)
        
        if not newly_activated_this_step:
            break
        
        all_activated_nodes.update(newly_activated_this_step)
        current_wave = newly_activated_this_step
            
    return all_activated_nodes

def linear_threshold(graph, seeds, steps=10, verbose=False): # Verbose parameter retained, but internal printing removed
    """Linear Threshold Model"""
    if not graph or not seeds:
        return set()

    valid_seeds = {s for s in seeds if s in graph}
    if not valid_seeds:
        return set()
        
    all_activated_nodes = set(valid_seeds)
    # The thresholds are still randomly generated inside the function for each node
    thresholds = {n: random.random() for n in graph.nodes()}

    for _ in range(steps): # The iteration variable step is no longer used
        newly_activated_this_step = set()
        # Check all currently inactive nodes
        nodes_to_check = graph.nodes() - all_activated_nodes
        
        for node in nodes_to_check:
            if node not in graph:
                continue
            
            current_node_neighbors = list(graph.neighbors(node))
            if not current_node_neighbors:
                continue

            influence = 0.0
            active_influencing_neighbors_count = 0
            for neighbor in current_node_neighbors:
                if neighbor in all_activated_nodes:
                    if graph.degree(neighbor) > 0:
                        influence += 1.0 / graph.degree(neighbor)
                    active_influencing_neighbors_count +=1
            
            if active_influencing_neighbors_count == 0:
                continue
            
            if influence >= thresholds.get(node, 1.0):
                newly_activated_this_step.add(node)
        
        if not newly_activated_this_step:
            break
            
        all_activated_nodes.update(newly_activated_this_step)

    return all_activated_nodes

def influence_score(node_id, precomputed_degree_cent_dict, precomputed_betweenness_cent_dict, precomputed_pagerank_dict):
    deg = precomputed_degree_cent_dict.get(node_id, 0.0) 
    bet = precomputed_betweenness_cent_dict.get(node_id, 0.0)
    pr = precomputed_pagerank_dict.get(node_id, 0.0)
    return 0.4 * deg + 0.3 * bet + 0.3 * pr

def get_top_influencers_from_scores(k_influencers, all_influence_scores_dict):
    if not all_influence_scores_dict:
        print("Warning: get_top_influencers_from_scores The received influence score dictionary is empty. Returns an empty list.")
        return []
    sorted_nodes_by_influence = sorted(all_influence_scores_dict.items(), key=lambda item: item[1], reverse=True)
    top_k_nodes = [node_id for node_id, score in sorted_nodes_by_influence[:k_influencers]]
    return top_k_nodes

def run_comparison(G, seeds, prob=0.1, steps=10):
    """Compare the Independent Cascade (IC) and Linear Threshold (LT) models"""
    print("\n🔬 Comparative Experiments on Propagation Models")
    print("\n📌 Independent Cascade (IC) Model:")
    ic_result = independent_cascade(G, seeds, prob, steps)
    print("\n📌 Linear Threshold (LT) Model:")
    lt_result = linear_threshold(G, seeds, steps)
    print(f"\n📊 Comparison of results: IC infected {len(ic_result)} nodes | LT activated {len(lt_result)} nodes")
    return ic_result, lt_result

def plot_spread(G, active_nodes, partition, title, timestamp):
    """Visualize propagation results"""
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    inactive = set(G.nodes()) - set(active_nodes)
    nx.draw_networkx_nodes(
        G, pos, nodelist=inactive,
        node_color=[partition[n] for n in inactive],
        cmap=plt.cm.tab20, node_size=20, alpha=0.3
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=active_nodes,
        node_color='red', node_size=50, alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{title.lower().replace(' ', '_')}_{timestamp}.png")
    plt.show()

# Label Propagation Algorithm (LPA)
def label_propagation(G, max_iter=100, min_change=0.01):
    labels = {node: i for i, node in enumerate(G.nodes())}
    nodes = list(G.nodes())
    total_nodes = len(nodes)
    for iteration in range(max_iter):
        random.shuffle(nodes)
        changed = 0
        for node in nodes:
            neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
            if neighbor_labels:
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                if labels[node] != most_common:
                    labels[node] = most_common
                    changed += 1
        change_ratio = changed / total_nodes
        print(f"LPA Iteration {iteration + 1}: Changed {changed} nodes ({change_ratio:.4f})")
        if change_ratio < min_change:
            break
    return labels

def community_density(G, partition):
    density = {}
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    for comm, nodes in communities.items():
        subgraph = G.subgraph(nodes)
        if subgraph.number_of_nodes() > 0:
            density[comm] = nx.density(subgraph)
    return density
    
# ---------- Parameters ----------
top_n = 10  # Extract the top N large communities
sample_max = 5000  # Upper limit for centrality sampling
betweenness_k = 300  # Number of samples for betweenness centrality
run_diffusion_models = True  # Controls whether to run IC and LT models

# ---------- 1. Loading Graph ----------
file_path = r"C:\Users\z1562\OneDrive - Debreceni Egyetem\Asztal\Social Media FinalProject\twitter_combined.txt"
with open(file_path, 'r') as f:
    lines = [line for line in f if not line.startswith('#')][:10000]
G = nx.DiGraph()
for line in lines:
    source, target = map(int, line.strip().split())
    G.add_node(source)
    G.add_node(target)
    G.add_edge(source, target)
print(f"✅ Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges in G")
G_undirected = G.to_undirected()
print(f"✅ G_undirected has {G_undirected.number_of_nodes()} nodes and {G_undirected.number_of_edges()} edges")

#-----------Data quality check----------
edges_list = [(u, v) for u, v in G.edges()]
edges = pd.DataFrame(edges_list, columns=["source", "target"])
print("Duplicates:", edges.duplicated().sum())
edges = edges.drop_duplicates()
print("Missing values:", edges.isnull().sum())
edges = edges.dropna()
print(f"Edges after dropping missing values: {len(edges)}")
G = nx.from_pandas_edgelist(edges, source="source", target="target", create_using=nx.DiGraph())
print(f"✅ After data quality check: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
G_undirected = G.to_undirected()
print(f"✅ G_undirected has {G_undirected.number_of_nodes()} nodes and {G_undirected.number_of_edges()} edges")

# ---------- 2. Convert to undirected graph + extract maximal connected components ----------
G_undirected = G.to_undirected()
if nx.is_empty(G_undirected):
    raise ValueError("Empty graph!!")

selfloops, isolated, outliers = check_data_quality(G_undirected)

# Store to Neo4j
if USE_NEO4J:
    neo4j_conn.add_graph(G_undirected, centrality_df=None, partition=None)
else:
    print("Skipped storing graph in Neo4j")
if USE_NEO4J:
    high_degree_nodes = neo4j_conn.query_high_degree_nodes()
    print("Top 5 High-Degree Nodes from Neo4j:")
    for node, degree, community in high_degree_nodes:
        print(f"Node {node}: Degree {degree}, Community {community}")
else:
    print("Skipped querying high-degree nodes from Neo4j")
components = list(nx.connected_components(G_undirected))
if not components:
    raise ValueError("Can't find a connected subgraph!")

largest_cc = max(components, key=len)
G_sub = G_undirected.subgraph(largest_cc).copy()
print(f"Largest connected component: {G_sub.number_of_nodes()} nodes")

# Add labels
if USE_NEO4J:
    with neo4j_conn.driver.session() as session:
        session.run(
            """
            UNWIND $nodes AS node_id
            MATCH (n:Node {id: node_id})
            SET n.is_largest_cc = true
            """,
            nodes=list(largest_cc)
        )
    print("✅ Marked largest connected component nodes in Neo4j")
else:
    print("Skipped marking largest connected component nodes in Neo4j")

# ---------- 3. Enhanced preprocessing ----------
avg_degree = sum(dict(G_undirected.degree()).values()) / G_undirected.number_of_nodes()
k_value = 2 if avg_degree > 5 else 1
G_kcore = preprocess_graph(G_undirected, min_degree=k_value, max_degree=500)
if G_kcore.number_of_nodes() == 0:
    raise ValueError(f"k-core Empty graph after processing (k={k_value})")
print(f"✅ After preprocessing: {G_kcore.number_of_nodes()} nodes, {G_kcore.number_of_edges()} edges (k={k_value})")

# ---------- 4. Louvain Community Detection ----------
print("\n⏳ Running Louvain Community Detection (k-core)...")
start_time = time.time()
partition_all = community_louvain.best_partition(G_kcore)
louvain_time = time.time() - start_time
num_communities0 = len(set(partition_all.values()))
print(f"✔️ Louvain detected {num_communities0} communities (Time: {louvain_time:.2f} seconds)")

# ---------- 5. Extract the top N large community subgraphs ----------
community_sizes = Counter(partition_all.values())
top_communities = [c for c, _ in community_sizes.most_common(top_n)]
selected_nodes = [n for n in G_kcore.nodes() if partition_all[n] in top_communities]
G_selected = G_kcore.subgraph(selected_nodes).copy()
print(f"📌 Top {top_n} large community subgraphs: {G_selected.number_of_nodes()} nodes, {G_selected.number_of_edges()} edges")
print(f"G_selected has {G_selected.number_of_nodes()} nodes and {G_selected.number_of_edges()} edges")

# ---------- 6. Run Louvain again (for analysis) ----------
print("\n🔁 Re-running Louvain detection on subgraph...")
partition = community_louvain.best_partition(G_selected)
num_communities = len(set(partition.values()))
print(f"✔️ Subgraph Louvain detected {num_communities} communities")

# ---------- 7. Centrality Analysis ----------
print("\n🎯 Sampling nodes for centrality analysis...")
sample_size = min(sample_max, G_selected.number_of_nodes())
sampled_nodes = random.sample(list(G_selected.nodes()), sample_size)
G_sample = G_selected.subgraph(sampled_nodes).copy()
print(f"Sampled graph size: {G_sample.number_of_nodes()} nodes, {G_sample.number_of_edges()} edges")
print(f"G_sample has {G_sample.number_of_nodes()} nodes and {G_sample.number_of_edges()} edges after sampling")

print("⚙️ Computing centrality metrics (using approximation)...")
k_bt = min(betweenness_k, G_sample.number_of_nodes())
degree_centrality = nx.degree_centrality(G_sample)
betweenness_centrality = nx.betweenness_centrality(G_sample, k=k_bt, seed=42)
closeness_centrality = nx.closeness_centrality(G_sample)
partition_sample = community_louvain.best_partition(G_sample)

centrality_df = pd.DataFrame({
    'node': list(G_sample.nodes()),
    'degree': [degree_centrality[n] for n in G_sample.nodes()],
    'betweenness': [betweenness_centrality[n] for n in G_sample.nodes()],
    'closeness': [closeness_centrality[n] for n in G_sample.nodes()],
    'community': [partition_sample[n] for n in G_sample.nodes()]
})

# Update node attributes in Neo4j
if USE_NEO4J:
    neo4j_conn.add_graph(G_sample, centrality_df=centrality_df, partition=partition_sample)
    print("Updated node attributes in Neo4j")
else:
    print("Skipped updating node attributes in Neo4j")

# ------ ROI Experiment Parameters ------
# Precomputation
"""
influence_scores_gsample = {n: influence_score(G_sample, n) for n in G_sample.nodes()}
degree_sorted = [n for n, _ in sorted(G_sample.degree(), key=lambda x: x[1], reverse=True)]
betweenness_scores = dict(zip(G_sample.nodes(), [betweenness_centrality[n] for n in G_sample.nodes()]))
"""
# Experiment configuration
config = {
    'seed_sizes': [1, 3, 5, 10, 15],
    'num_runs': 10,
    'prob_ic': 0.1,
    'steps_ic': 10,  # Number of simulation steps for IC model
    'steps_lt': 10   # Number of simulation steps for LT model
}
# --- Step 1: Precompute all required node metrics (for G_sample) ---
print(f"\nPrecomputing metrics for G_sample ({G_sample.number_of_nodes()} nodes)...")
precomputation_start_time = time.time()
print("  Computing node degrees...")
degrees_gsample = dict(G_sample.degree())
print("  Computing PageRank...")
start_time_pagerank = time.time()
try:
    pagerank_gsample_dict = nx.pagerank(G_sample, alpha=0.85)
except Exception: # Simplified exception handling
    pagerank_gsample_dict = {n: 0.0 for n in G_sample.nodes()}
print(f"  PageRank computation completed, time taken: {time.time() - start_time_pagerank:.2f}s")
print("  Computing degree centrality...")
start_time_deg_cent = time.time()
try:
    degree_centrality_gsample_dict = nx.degree_centrality(G_sample)
except Exception:
    degree_centrality_gsample_dict = {n: 0.0 for n in G_sample.nodes()}
print(f"  Degree centrality computation completed, time taken: {time.time() - start_time_deg_cent:.2f}s")
print("  Computing betweenness centrality...")
start_time_bet_cent = time.time()
betweenness_k_approx = min(100, G_sample.number_of_nodes())
if G_sample.number_of_nodes() <= 2:
    betweenness_centrality_gsample_dict = {n: 0.0 for n in G_sample.nodes()}
else:
    try:
        betweenness_centrality_gsample_dict = nx.betweenness_centrality(G_sample, k=betweenness_k_approx, normalized=True, seed=42)
    except Exception:
        betweenness_centrality_gsample_dict = {n: 0.0 for n in G_sample.nodes()}
print(f"  Betweenness centrality computation completed, time taken: {time.time() - start_time_bet_cent:.2f}s")
print("  Computing composite influence scores...")
start_time_influence = time.time()
influence_scores_gsample = {
    n: influence_score(n, degree_centrality_gsample_dict, betweenness_centrality_gsample_dict, pagerank_gsample_dict)
    for n in G_sample.nodes()
}
print(f"  Composite influence score computation completed, time taken: {time.time() - start_time_influence:.2f}s")
print(f"All metrics precomputation completed, total time taken: {time.time() - precomputation_start_time:.2f}s")

# --- Step 2: Define seed selection strategies (using precomputed data) ---
# Lambda functions no longer take graph object g as a parameter, as they are based on precomputed results from G_sample
# They only take the number of seeds k
strategies = {
    'TopInfluence': lambda k_val: get_top_influencers_from_scores(k_val, influence_scores_gsample),
    'HighDegree': lambda k_val: sorted(degrees_gsample, key=degrees_gsample.get, reverse=True)[:k_val],
    'HighBetweenness': lambda k_val: sorted(betweenness_centrality_gsample_dict, 
                                            key=betweenness_centrality_gsample_dict.get, reverse=True)[:k_val],
    'Random': lambda k_val: random.sample(list(G_sample.nodes()), min(k_val, G_sample.number_of_nodes()))
}

# Experiment loop
results = defaultdict(list) # Stores infection rate lists for each (k, strategy, model) combination
total_nodes_in_sample = G_sample.number_of_nodes()

print(f"\n🚀 Starting ROI experiment loop (total {len(config['seed_sizes']) * len(strategies) * 2 * config['num_runs']} propagation simulations)...")

for k in config['seed_sizes']:
    # print(f"\n  Testing seed size K = {k}...") # Removed for simplification
    if k == 0: continue
    if total_nodes_in_sample == 0 : break
    if k > total_nodes_in_sample: continue # Simplified handling

    for strategy_name, strategy_func in strategies.items():
        # print(f"    Strategy: {strategy_name}") # Removed for simplification
        try:
            current_seeds = strategy_func(k)
            if not current_seeds and k > 0: continue
        except Exception as e:
            print(f"      Error: Strategy {strategy_name} K={k} failed during seed selection: {e}. Skipping.")
            continue

        ic_rates_for_this_setup = []
        lt_rates_for_this_setup = []

        for i_run in range(config['num_runs']):
            # IC model
            try:
                current_ic_prob = config['prob_ic']
                current_ic_steps = config['steps_ic']
                ic_nodes = independent_cascade(G_sample, current_seeds,
                                               prob=current_ic_prob,
                                               steps=current_ic_steps,
                                               verbose=False)
                ic_rate = len(ic_nodes) / total_nodes_in_sample if total_nodes_in_sample > 0 else 0
                ic_rates_for_this_setup.append(ic_rate)
            except Exception as e:
                # Simplified error printing, uncomment traceback.print_exc() if further debugging needed
                print(f"        IC model K={k}, Strat={strategy_name}, Run={i_run+1} error: {repr(e)}")
                # traceback.print_exc()
                ic_rates_for_this_setup.append(0.0)

            # LT model
            try:
                current_lt_steps = config['steps_lt']
                lt_nodes = linear_threshold(G_sample, current_seeds,
                                            steps=current_lt_steps,
                                            verbose=False)
                lt_rate = len(lt_nodes) / total_nodes_in_sample if total_nodes_in_sample > 0 else 0
                lt_rates_for_this_setup.append(lt_rate)
            except Exception as e:
                print(f"        LT model K={k}, Strat={strategy_name}, Run={i_run+1} error: {repr(e)}")
                # traceback.print_exc()
                lt_rates_for_this_setup.append(0.0)

        results[(k, strategy_name, 'IC')].extend(ic_rates_for_this_setup)
        results[(k, strategy_name, 'LT')].extend(lt_rates_for_this_setup)
    print(f"  K = {k} completed.") # Retain minimal progress printing

print("\n--- All ROI simulations completed ---")

# --- Step 4: Aggregate results and prepare output ---
print("\nAggregating experiment results...")
aggregated_data = []
for (k, strategy, model_type), rates_list in results.items():
    if rates_list: # Ensure the list is not empty
        avg_rate = sum(rates_list) / len(rates_list)
        std_dev_rate = pd.Series(rates_list).std() if len(rates_list) > 1 else 0.0 # Calculate standard deviation
        num_actual_runs = len(rates_list)
    else: # If no rates were collected for some reason
        avg_rate = 0.0
        std_dev_rate = 0.0
        num_actual_runs = 0
        
    aggregated_data.append({
        'k': k,
        'strategy': strategy,
        'model_type': model_type,
        'avg_infection_rate': avg_rate,
        'std_dev_infection_rate': std_dev_rate,
        'num_runs_collected': num_actual_runs # Record the number of actual runs collected
    })

aggregated_results_df = pd.DataFrame(aggregated_data)

# Save aggregated results
if not aggregated_results_df.empty:
    from datetime import datetime # Ensure import
    timestamp_agg = datetime.now().strftime('%Y%m%d_%H%M%S')
    agg_csv_path = f"roi_aggregated_results_optimized_{timestamp_agg}.csv"
    aggregated_results_df.to_csv(agg_csv_path, index=False)
    print(f"\n✅ ROI aggregated analysis results saved to: {agg_csv_path}")

    # Get the absolute path of the file
    full_path = os.path.abspath(agg_csv_path)
    print(f"\n✅ ROI aggregated analysis results saved to: {full_path}")  # Print full path

# Step 3: Visualization (Pure Matplotlib implementation)
import pandas as pd
import matplotlib.pyplot as plt

try:
    # Load CSV data
    results_df = pd.read_csv(agg_csv_path)
    print(f"Data loaded successfully from '{agg_csv_path}'! Here are the first few rows:")
    print(results_df.head())
    print("\nData column names:")
    print(results_df.columns)

    # Ensure required columns exist
    required_columns = ['k', 'strategy', 'model_type', 'avg_infection_rate', 'std_dev_infection_rate']
    if not all(col in results_df.columns for col in required_columns):
        print(f"Error: CSV file is missing required columns. Required columns: {required_columns}")
        exit()

    # ------------------------- Visualization Section -------------------------
    print("\nGenerating visualization charts...")

    # Create figure
    plt.figure(figsize=(14, 6))

    # ========== Chart 1: Average Infection Rate Comparison for Different Strategies and Model Types (Bar Plot) ==========
    plt.subplot(1, 2, 1)

    # Get unique strategies and model types
    strategies = results_df['strategy'].unique()
    model_types = results_df['model_type'].unique()

    # Set bar positions and width
    bar_width = 0.35
    x_pos = range(len(strategies))

    # Plot bars for each model type
    for i, model in enumerate(model_types):
        # Calculate average for each strategy
        values = [results_df[(results_df['strategy'] == s) & 
                             (results_df['model_type'] == model)]['avg_infection_rate'].mean()
                  for s in strategies]
        # Plot bars (offset to display bars for different model types side by side)
        plt.bar([x + i * bar_width for x in x_pos], values, 
                width=bar_width, label=model)

    # Add chart elements
    plt.title("Average Infection Rate by Strategy")
    plt.xticks([x + bar_width / 2 for x in x_pos], strategies, rotation=45)
    plt.xlabel("Strategy")
    plt.ylabel("Average Infection Rate")
    plt.legend(title="Model Type")
    plt.grid(True, linestyle='--', alpha=0.6)

    # ========== Chart 2: Effect of k on Infection Rate (Line Plot) ==========
    plt.subplot(1, 2, 2)

    # Define different line styles and markers
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    # Plot lines for each strategy and model type combination
    for (strategy, model), group in results_df.groupby(['strategy', 'model_type']):
        # Sort by k
        group = group.sort_values('k')
        # Select style and marker
        linestyle = line_styles[list(results_df['strategy'].unique()).index(strategy) % len(line_styles)]
        marker = markers[list(results_df['model_type'].unique()).index(model) % len(markers)]
        # Plot line
        plt.plot(group['k'], group['avg_infection_rate'], 
                 linestyle=linestyle, marker=marker,
                 label=f"{strategy} ({model})")

    # Add chart elements
    plt.title("Effect of Seed Size (k) on Infection Rate")
    plt.xlabel("Seed Size (k)")
    plt.ylabel("Average Infection Rate")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("infection_rate_analysis.png")  # Save chart
    print("✅ Visualization chart saved as infection_rate_analysis.png")
    plt.show()

except FileNotFoundError:
    print(f"Error: CSV file '{agg_csv_path}' not found. Please ensure the filename and path are correct.")
    exit()
except Exception as e:
    print(f"Error loading or processing CSV: {e}")
    exit()

    print("\nAggregated results preview:")
    print(aggregated_results_df.sort_values(by=['k', 'strategy', 'model_type']).head(10))
else:
    print("\nROI analysis did not generate aggregatable results.")
    
# Assume you have this value (you can set the actual number)
TOTAL_NODES_IN_SAMPLE = 1000

try: 
    # ===== Mapping Logic (Automatically Identify Actual Column Names) =====
    expected_to_actual = {
        'k_seeds': ['k', 'k_seeds', 'num_seeds'],
        'strategy': ['strategy', 'seed_strategy', 'method'],
        'model_type': ['model_type', 'diffusion_model'],
        'avg_infection_rate': ['avg_infection_rate', 'infection_mean'],
        'std_dev_infection_rate': ['std_dev_infection_rate', 'infection_std']
    }

    col_map = {}
    for standard_col, possible_names in expected_to_actual.items():
        for actual_name in possible_names:
            if actual_name in results_df.columns:
                col_map[standard_col] = actual_name
                break
        if standard_col not in col_map:
            raise ValueError(f"❌ Missing required column: {standard_col}, please confirm CSV contains one of: {possible_names}")

    print("\n✅ Column name mapping:")
    print(col_map)

    # ===== Calculate ROI =====
    if TOTAL_NODES_IN_SAMPLE > 0:
        results_df['avg_infection_count'] = results_df[col_map['avg_infection_rate']] * TOTAL_NODES_IN_SAMPLE
        results_df['simple_roi'] = results_df.apply(
            lambda row: (row['avg_infection_count'] - row[col_map['k_seeds']]) / row[col_map['k_seeds']]
            if row[col_map['k_seeds']] > 0 else 0, axis=1
        )
        print("\n✅ Calculated 'avg_infection_count' and 'simple_roi' columns:")
        print(results_df[[col_map['k_seeds'], col_map['strategy'], col_map['model_type'],
                          col_map['avg_infection_rate'], 'avg_infection_count', 'simple_roi']].head())
    else:
        print("\n⚠️ Warning: TOTAL_NODES_IN_SAMPLE is 0, cannot calculate ROI.")

    # ===== Simple ROI vs k Visualization (Example) =====
    plt.figure(figsize=(8, 6))
    for (strategy, model), group in results_df.groupby([col_map['strategy'], col_map['model_type']]):
        group = group.sort_values(col_map['k_seeds'])
        plt.plot(group[col_map['k_seeds']], group['simple_roi'], marker='o', label=f"{strategy} ({model})")

    plt.title("ROI vs Seed Size (k)")
    plt.xlabel("Seed Size (k)")
    plt.ylabel("ROI")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("roi_analysis.png")
    print("\n✅ ROI chart saved as roi_analysis.png")
    plt.show()

except FileNotFoundError:
    print(f"❌ Error: CSV file '{agg_csv_path}' not found. Please ensure the path is correct.")
except Exception as e:
    print(f"❌ Error loading or processing CSV: {e}")

# ===== Calculate Marginal Benefit (Marginal Benefit of avg_infection_rate w.r.t k_seeds) =====
try:
    grouped_results = results_df.groupby([col_map['strategy'], col_map['model_type']])

    print("\n📊 Marginal Benefit of avg_infection_rate w.r.t k_seeds for each strategy + model:")
    all_marginal_benefits = []

    for (strategy_name, model_name), group in grouped_results:
        sorted_group = group.sort_values(by=col_map['k_seeds']).copy()
        sorted_group['marginal_benefit_rate'] = sorted_group[col_map['avg_infection_rate']].diff() / sorted_group[col_map['k_seeds']].diff()
        print(f"\n▶️ Strategy: {strategy_name}, Model: {model_name}")
        print(sorted_group[[col_map['k_seeds'], col_map['avg_infection_rate'], 'marginal_benefit_rate']])
        all_marginal_benefits.append(sorted_group)

    # Combine all groups
    marginal_df = pd.concat(all_marginal_benefits, ignore_index=True)

    # Optional: Save marginal benefit results
    marginal_df.to_csv("marginal_benefit_results.csv", index=False)
    print("\n✅ All marginal benefit results saved as marginal_benefit_results.csv")

    # ===== Visualize Marginal Benefit =====
    plt.figure(figsize=(8, 6))
    for (strategy_name, model_name), group in marginal_df.groupby([col_map['strategy'], col_map['model_type']]):
        group = group.sort_values(by=col_map['k_seeds'])
        plt.plot(group[col_map['k_seeds']], group['marginal_benefit_rate'],
                 marker='o', label=f"{strategy_name} ({model_name})")

    plt.title("Marginal Infection Rate Benefit vs Seed Size (k)")
    plt.xlabel("Seed Size (k)")
    plt.ylabel("Marginal Infection Rate Benefit")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("marginal_benefit_plot.png")
    print("✅ Marginal benefit chart saved as marginal_benefit_plot.png")
    plt.show()

except Exception as e:
    print(f"\n❌ Error calculating marginal benefit: {e}")

# (Function 3) IC vs LT Model Comparison Plot for the Same Strategy
output_plot_dir = "roi_plots"
os.makedirs(output_plot_dir, exist_ok=True)

colors = plt.cm.get_cmap('tab10', len(model_types))
markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']

for strategy in results_df['strategy'].unique():
    plt.figure(figsize=(10, 7))
    for model_idx, model in enumerate(results_df['model_type'].unique()):
        subset = results_df[(results_df['strategy'] == strategy) & (results_df['model_type'] == model)].sort_values(by='k')
        if not subset.empty:
            plt.plot(subset['k'], subset['avg_infection_rate'],
                     marker=markers[model_idx % len(markers)],
                     linestyle='-',
                     color=colors(model_idx),
                     label=model)

            if 'std_dev_infection_rate' in subset.columns:
                plt.fill_between(subset['k'],
                                 subset['avg_infection_rate'] - subset['std_dev_infection_rate'],
                                 subset['avg_infection_rate'] + subset['std_dev_infection_rate'],
                                 color=colors(model_idx), alpha=0.15)

    plt.title(f'Model Comparison for Strategy "{strategy}" (IC vs LT)', fontsize=16)
    plt.xlabel("Seed Size (k)", fontsize=14)
    plt.ylabel("Average Infection Rate", fontsize=14)
    plt.legend(title="Model Type", fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = os.path.join(output_plot_dir, f"compare_ic_lt_{strategy.lower().replace(' ', '_')}.png")
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"✅ Model comparison chart saved as: {filename}")

# Feature Selection
from scipy.stats import pearsonr
correlations = {}
centrality_values = centrality_df[['degree', 'betweenness', 'closeness']].values
for i, metric1 in enumerate(['degree', 'betweenness', 'closeness']):
    for j, metric2 in enumerate(['degree', 'betweenness', 'closeness']):
        if i < j:
            corr, _ = pearsonr(centrality_values[:, i], centrality_values[:, j])
            correlations[(metric1, metric2)] = corr
print("\n")
print("Correlation between centrality measures:")
for (m1, m2), corr in correlations.items():
    print(f"{m1} vs {m2}: {corr:.4f}")

selected_features = ['degree', 'closeness'] if abs(correlations.get(('degree', 'betweenness'), 0)) > 0.75 else ['degree', 'betweenness', 'closeness']
centrality_df = centrality_df[['node', 'community'] + selected_features]
print(f"Selected features: {selected_features}")

# Discretize centrality
degree_bins = pd.cut(centrality_df['degree'], bins=3, labels=['Low', 'Medium', 'High'])
centrality_df['degree_category'] = degree_bins
print("Degree Centrality Distribution (Discretized):")
print(degree_bins.value_counts())

# Define timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Visualize degree centrality distribution
plt.figure(figsize=(8, 6))
degree_bins.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Degree Centrality Distribution (Discretized)")
plt.xlabel("Category")
plt.ylabel("Count")
plt.savefig(f"degree_centrality_distribution_{timestamp}.png", dpi=300)
plt.close()

# Save results
csv_path = os.path.join(os.getcwd(), f"centrality_results_{timestamp}.csv")
centrality_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ Centrality analysis results saved as: {csv_path}")

# Community feature extraction
densities = community_density(G_sample, partition_sample)
print("Community Densities:")
for comm, density in densities.items():
    print(f"Community {comm}: Density = {density:.4f}")

# Visualize community densities
plt.figure(figsize=(8, 6))
density_values = list(densities.values())
plt.bar(range(len(densities)), density_values, color='skyblue', edgecolor='black')
ax = degree_bins.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title("Community Densities")
plt.xlabel("Community ID")
plt.ylabel("Density")
plt.savefig(f"community_densities_{timestamp}.png", dpi=300)
plt.close()

# Community aggregation
community_avg_degree = defaultdict(list)
for node, comm in partition_sample.items():
    community_avg_degree[comm].append(G_sample.degree(node))
print("Community Average Degrees:")
for comm, degrees in community_avg_degree.items():
    print(f"Community {comm}: Average Degree = {sum(degrees)/len(degrees):.2f}")

# ---------- 8. Community Detection Comparison ----------
# LPA
print("\n⏳ Running Label Propagation Algorithm (LPA)...")
start_time = time.time()
partition_lpa = label_propagation(G_sample)
lpa_time = time.time() - start_time
num_communities_lpa = len(set(partition_lpa.values()))
print(f"✔️ LPA detected {num_communities_lpa} communities (Time: {lpa_time:.2f} seconds)")

# Compare community detection
print("\n📊 Community Detection Comparison:")
print(f"Louvain: {num_communities} communities (Time: {louvain_time:.2f} seconds)")
print(f"LPA: {num_communities_lpa} communities (Time: {lpa_time:.2f} seconds)")
#print(f"Girvan-Newman: {num_communities_gn} communities (on 300-node subgraph, Time: {gn_time:.2f} seconds)")

# Modularity comparison
louvain_mod = nx_comm.modularity(G_sample, [{n for n, c in partition_sample.items() if c == i} for i in set(partition_sample.values())])
lpa_mod = nx_comm.modularity(G_sample, [{n for n, c in partition_lpa.items() if c == i} for i in set(partition_lpa.values())])
#gn_mod = nx_comm.modularity(G_small, best_communities)
print("\n📊 Modularity Comparison:")
print(f"Louvain Modularity: {louvain_mod}")
print(f"LPA Modularity: {lpa_mod}")
print("Girvan-Newman Modularity: Skipped (Girvan-Newman algorithm commented out)")
#print(f"Girvan-Newman Modularity: {gn_mod} (on 300-node subgraph)")

# ---------- 9. Influence Analysis and IC/LT Models ----------
if run_diffusion_models:
    print("\n🚀 Running influence propagation model comparison...")
    top_influencers = get_top_influencers(degree_centrality_gsample_dict, betweenness_centrality_gsample_dict, 
                                          pagerank_gsample_dict, 5)
    print(f"Top 5 Influencers: {top_influencers}")
    ic_nodes, lt_nodes = run_comparison(G_sample, top_influencers)
    
    # Update infected nodes in Neo4j
    if USE_NEO4J:
        with neo4j_conn.driver.session() as session:
            neo4j_conn.update_infected_nodes(ic_nodes)
        print("Updated infected nodes in Neo4j")
        infection_pairs = [(seed, node) for seed in top_influencers for node in ic_nodes if G_sample.has_edge(seed, node)]
        neo4j_conn.update_infected_nodes(ic_nodes)
        print("Added infection relationships in Neo4j")
    else:
        print("Skipped updating infected nodes in Neo4j")
    infection_pairs = [(seed, node) for seed in top_influencers for node in ic_nodes if G_sample.has_edge(seed, node)]

    results_df = pd.DataFrame({
        'Model': ['IC', 'LT'],
        'Influenced_Nodes': [len(ic_nodes), len(lt_nodes)],
        'Seeds': [str(top_influencers)] * 2
    })
    results_df.to_csv(f"model_comparison_{timestamp}.csv", index=False)
    print("\n🎨 Visualizing propagation results...")
    plot_spread(G_sample, ic_nodes, partition_sample, "IC Model Spread", timestamp)
    plot_spread(G_sample, lt_nodes, partition_sample, "LT Model Spread", timestamp)

    # Compatible with legacy output
    influencer_communities = {n: partition_sample.get(n, -1) for n in top_influencers}
    community_dist = Counter(influencer_communities.values())
    print("📈 Top Influencers Community Distribution:")
    for comm, count in sorted(community_dist.items()):
        print(f"Community {comm}: {count} influencers")

    # Save influencer CSV
    influencer_df = pd.DataFrame(
        [(n, influence_score(n, degree_centrality_gsample_dict, betweenness_centrality_gsample_dict,
                              pagerank_gsample_dict), partition_sample.get(n, -1)) for n in top_influencers],
        columns=['Node', 'Influence_Score', 'Community']
    )
    influencer_df.to_csv(f"influencer_communities_{timestamp}.csv", index=False)
    print(f"✅ Saved influencer_communities_{timestamp}.csv")

    # Save infected nodes CSV
    infected_df_ic = pd.DataFrame(
        [(n, partition_sample.get(n, -1)) for n in ic_nodes],
        columns=['Node', 'Community']
    )
    infected_df_ic.to_csv(f"infected_nodes_ic_{timestamp}.csv", index=False)
    print(f"✅ Saved infected_nodes_ic_{timestamp}.csv")

    infected_df_lt = pd.DataFrame(
        [(n, partition_sample.get(n, -1)) for n in lt_nodes],
        columns=['Node', 'Community']
    )
    infected_df_lt.to_csv(f"infected_nodes_lt_{timestamp}.csv", index=False)
    print(f"✅ Saved infected_nodes_lt_{timestamp}.csv")
else:
    print("\n⏭️ Skipped IC and LT model execution.")

# ---------- 10. Visualization ----------
print("\n🎨 Drawing community structure graph (sampled graph)...")
pos = nx.spring_layout(G_sample, seed=42)
node_colors = [partition_sample[n] for n in G_sample.nodes()]

plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(
    G_sample, pos,
    node_color=node_colors,
    cmap=plt.cm.tab20,
    node_size=20,
    alpha=0.8
)
nx.draw_networkx_edges(G_sample, pos, alpha=0.2, width=0.3)
plt.title("Louvain Community Structure Graph (Sampled Graph)")
plt.axis("off")
plt.tight_layout()
png_path = os.path.join(os.getcwd(), f"community_visualization_{timestamp}.png")
plt.savefig(png_path, dpi=300)
plt.show()
print(f"🖼️ Community structure graph saved as: {png_path}")

print("\n📊 Community sizes and top 10 nodes (by actual degree):")
community_nodes = defaultdict(list)
for node, comm in partition_sample.items():
    community_nodes[comm].append(node)

for comm_id, nodes in sorted(community_nodes.items(), key=lambda x: -len(x[1])):
    node_degrees = [(n, G_sample.degree[n]) for n in nodes]
    sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)[:10]
    print(f"\nCommunity {comm_id}: Total {len(nodes)} nodes")
    print("Top 10 nodes (by actual degree):")
    for node, degree in sorted_nodes:
        print(f"Node {node}, Degree {degree}")

# Visualize Degree Distribution (to reveal structural heterogeneity)
plt.figure(figsize=(8, 6))
degrees = [G_sample.degree(n) for n in G_sample.nodes()]
plt.hist(degrees, bins=30, color='skyblue', edgecolor='black')
plt.title("Degree Distribution Histogram (Sampled Graph)")
plt.xlabel("Degree")
plt.ylabel("Node Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
hist_path = os.path.join(os.getcwd(), "degree_distribution_hist.png")
plt.savefig(hist_path, dpi=300)
plt.show()
print(f"Degree distribution histogram saved as: {hist_path}")

if USE_NEO4J:
    neo4j_conn.close()
else:
    print("Skipped closing Neo4j connection")