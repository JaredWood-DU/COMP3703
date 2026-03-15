'''
FILE OVERVIEW:
- Underlying code intended for use in explore_complex_networks.ipynb to keep notebook cleaner
- Consists of various functions intended to derive complex network information to engineer into the dataset

=================================================

MISC COMMENTS:
- NA

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Batch Generation Functions
    - batch_generate_timing
- Visualization Functions
    - vis_original_overall_graph
    - vis_original_star_graph
    - vis_timing
    - vis_big_o
    - vis_bad_ips
- Feature Engineering Functions
    - generate_reduced_graph_df
    - remove_bad_ips
    - generate_graph_ids
    - generate_intensity_and_zscores
    - generate_complex_network_information
- Preprocessing Functions
    - preprocess_complex_data
- Helper Functions
    - get_nx_graph_generation_time
    - get_ig_graph_generation_time
    - get_nx_adj_matrix_generation_time
    - get_ig_adj_matrix_generation_time
    - get_nx_matrix_mult_generation_time
    - get_ig_matrix_mult_generation_time
    - calculate_big_o
'''
# ----- Imports -----------------------------------------------------------------------------------
# File Detection
import os

# Databasing
import numpy as np
import pandas as pd

# Networking
import networkx as nx
import igraph as ig

# Database splitting, encoding, scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Visualizations
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Timing
from time import time

# Matrix Manipulation
from scipy.sparse.linalg import eigsh

# ----- Global Variables --------------------------------------------------------------------------
# NA

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Batch Generation Functions
# =================================================================================================

def batch_generate_timing(pdDataFrame:pd.DataFrame) -> None:
    '''
    About
    -----
    - Convenience function to produce values and visualizations comparing NetworkX and iGraph timing
      and Big-O notation with graph generation, adjacency matrix generation, and matrix multiplication generation

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to be used during timing and visualization generation

    Returns
    -------
    - Multiple visualizations for ease of comparison between NetworkX and iGraph
    '''
    # ----- Visualize Graph Generation ------------------------------------------------------------
    # Print header statement
    print('\033[35m========== START GRAPH GENERATION COMPARISON ==========\033[0m')

    # Setup and visualize results
    nx_results = get_nx_graph_generation_time(pdDataFrame)
    ig_results = get_ig_graph_generation_time(pdDataFrame)
    vis_timing(pdDataFrame, nx_results, ig_results)
    vis_big_o(pdDataFrame, nx_results, ig_results)

    # ----- Visualize Adjacency Matrix Generation -------------------------------------------------
    # Print header statement
    print('\033[35m========== START ADJACENCY MATRIX GENERATION COMPARISON ==========\033[0m')

    # Setup and visualize results
    nx_results = get_nx_adj_matrix_generation_time(pdDataFrame)
    ig_results = get_ig_adj_matrix_generation_time(pdDataFrame)
    vis_timing(pdDataFrame, nx_results, ig_results)
    vis_big_o(pdDataFrame, nx_results, ig_results)

    # ----- Visualize Matrix Multiplication Generation --------------------------------------------
    # Print header statement
    print('\033[35m========== START MATRIX MULTIPLICATION GENERATION COMPARISON ==========\033[0m')

    # Setup and visualize results
    nx_results = get_nx_matrix_mult_generation_time(pdDataFrame)
    ig_results = get_ig_matrix_mult_generation_time(pdDataFrame)
    vis_timing(pdDataFrame, nx_results, ig_results)
    vis_big_o(pdDataFrame, nx_results, ig_results)    

# =================================================================================================
# END Batch Generation Functions
# START Visualization Functions
# =================================================================================================

def vis_original_overall_graph(pdDataFrame:pd.DataFrame,
                       sample_size:int=150,
                       target:str='attack'):
    '''
    About
    -----
    - Visualizes IP connectivity highlighting nodes with more than 5 connections.
        - Green: High-Degree Nodes (k > 5)
        - Red: Malicious nodes
        - Blue: Benign nodes

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to use for visualization
    - sample_size (int) :
        - Default: 150
        - The number of nodes to generate in the graph
    - target (str) :
        - Default: attack
        - The column name of the target classifier (Things like benign and ddos)

    Returns
    -------
    - Visualization of the original graph
    '''
    # ----- Generate The Graph --------------------------------------------------------------------
    # Create a smaller dataframe
    sample_df = pdDataFrame.sample(n=min(sample_size, len(pdDataFrame)), random_state=3703)

    # Create the undirected graph
    G = nx.from_pandas_edgelist(
        sample_df, 
        source='ipv4_src_addr', 
        target='ipv4_dst_addr', 
        edge_attr=target,
        create_using=nx.Graph() 
    )

    # ----- Print Is/Is Not Connected Graph -------------------------------------------------------
    is_connected = nx.is_connected(G)
    status = "Connected" if is_connected else "Not Connected"
    if is_connected:
        print(f'\033[32mGraph Status:\033[0m {status}')
    else:
        print(f'\033[31mGraph Status:\033[0m {status}')

    # ----- Determine Node Colors -----------------------------------------------------------------
    # Initialize variables
    degrees = dict(G.degree())
    node_colors = []
    node_labels = {}

    for node in G.nodes():
        # High-Degree Node Logic
        if degrees[node] > 5:
            node_colors.append('#2ecc71')
            node_labels[node] = f"{node}\n(k={degrees[node]})"

        # Otherwise, Blue if benign, Red if malicious
        else:
            incident_edges = G.edges(node, data=True)
            is_malicious = any(str(e[2][target]).lower() != 'benign' for e in incident_edges)
            node_colors.append('#e74c3c' if is_malicious else '#3498db')
            node_labels[node] = ""

    # ----- Visualize Data ------------------------------------------------------------------------
    # Visualization size and graph framing type
    plt.figure(figsize=(7, 7))
    pos = nx.kamada_kawai_layout(G)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.8, edge_color='#bdc3c7', alpha=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=120, node_color=node_colors, edgecolors='black', linewidths=0.5)
    
    # Draw labels for High-Degree nodes only
    label_pos = {k: [v[0], v[1] + 0.035] for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=8, font_weight='bold')

    # Draw legend
    green_dot = mlines.Line2D([], [], color='#2ecc71', marker='o', linestyle='None', markersize=10, label='High-Degree Node (k > 5)')
    red_dot = mlines.Line2D([], [], color='#e74c3c', marker='o', linestyle='None', markersize=10, label='Malicious')
    blue_dot = mlines.Line2D([], [], color='#3498db', marker='o', linestyle='None', markersize=10, label='Benign')
    plt.legend(handles=[green_dot, red_dot, blue_dot], loc='upper right', title="Security Profile", fontsize=10)

    # Show visualization
    plt.title(f"Global IP Connectivity (n={sample_size})", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def vis_original_star_graph(pdDataFrame:pd.DataFrame,
                            target:str='attack') -> None:
    '''
    About
    -----
    Visualizes the largest High-Degree Node (Star Hub) with a professional legend.
        - Green: High-Degree Node (Structural Hub)
        - Blue Neighbors: Benign traffic
        - Red Neighbors: Malicious traffic

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to use for visualization
    - target (str) :
        - Default: attack
        - The column name of the target classifier (Things like benign and ddos)

    Returns
    -------
    - Visualization of the original graph
    '''
    # ----- Generate Graph and Find Highest-Degree Node -------------------------------------------
    # Generate graph
    G_full = nx.from_pandas_edgelist(
        pdDataFrame.head(5000), 
        source='ipv4_src_addr', 
        target='ipv4_dst_addr', 
        edge_attr=target,
        create_using=nx.Graph()
    )

    # Find Highest-Degree Node
    degrees = dict(G_full.degree())
    max_hub = max(degrees, key=degrees.get)
    k_val = degrees[max_hub]
    
    # ----- Create Sub-Graph For Visualization ----------------------------------------------------
    neighbors = list(G_full.neighbors(max_hub))
    
    # Create a brand new Graph object to force a strict Star structure
    star_subgraph = nx.Graph()
    
    # Only add edges that connect the hub to a neighbor (no neighbor-to-neighbor edges)
    for neighbor in neighbors:
        edge_data = G_full.get_edge_data(max_hub, neighbor)
        star_subgraph.add_edge(max_hub, neighbor, **edge_data)

    # Verifiy that this is a star graph
    if len(star_subgraph.edges()) > 0:
        diam = nx.diameter(star_subgraph)
        print(f"Verified Topology: {'Star' if diam == 2 else 'Complex'} (Diam={diam})")

    # ----- Define Node Colors --------------------------------------------------------------------
    # Initialize variables
    node_colors = []
    node_labels = {}
    
    for node in star_subgraph.nodes():
        # High-Degree Node Logic
        if node == max_hub:
            node_colors.append('#2ecc71') # Green for High-Degree Node
            node_labels[node] = f"HIGH-DEGREE NODE\n{node}\n(k={k_val})"

        else:
            # Otherwise, Blue if benign, Red if malicious
            edge_data = G_full.get_edge_data(max_hub, node)
            attack_val = str(edge_data[target]).lower()
            node_colors.append('#e74c3c' if attack_val != 'benign' else '#3498db')
            node_labels[node] = ""

    # ----- Visualize Data ------------------------------------------------------------------------
    # Visualization size and graph framing type
    plt.figure(figsize=(14, 9))
    pos = nx.spring_layout(star_subgraph, k=0.4, seed=3703)

    # Draw edges
    nx.draw_networkx_edges(star_subgraph, pos, width=1.2, edge_color='#bdc3c7', alpha=0.5)
    
    # Draw Nodes
    nx.draw_networkx_nodes(star_subgraph, pos, node_size=1000, node_color=node_colors, edgecolors='black')

    # Draw labels for High-Degree nodes only
    label_pos = {k: [v[0], v[1] + 0.08] for k, v in pos.items()}
    nx.draw_networkx_labels(star_subgraph, label_pos, labels=node_labels, font_size=9, font_weight='bold')

    # Draw legend
    green_dot = mlines.Line2D([], [], color='#2ecc71', marker='o', linestyle='None', markersize=10, label='High-Degree Node')
    red_dot = mlines.Line2D([], [], color='#e74c3c', marker='o', linestyle='None', markersize=10, label='Malicious')
    blue_dot = mlines.Line2D([], [], color='#3498db', marker='o', linestyle='None', markersize=10, label='Benign')
    plt.legend(handles=[green_dot, red_dot, blue_dot], loc='upper right', title="Security Profile", fontsize=10)

    # Show visualization
    plt.title(f"Highest-Degree Star Graph (Diameter=2)", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def vis_timing(pdDataFrame:pd.DataFrame, 
               nx_size_time_dict:dict=None,
               ig_size_time_dict:dict=None) -> None:
    '''
    About
    -----
    - Visualization specifically for comparing timing of NetworkX and iGraph and their point-slope equations
    - The dictionary information is ideally from "get_nx/ig_..._generation_time"

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to use during
    - nx_size_time_dict (dict) :
        - Default: None
        - A dictionary representing the size and times taken to generate the NetworkX graphs
        - Ideally, this dictionary is derived from "get_nx_..._generation_time"
        - If None, this function ends prematurely
    - ig_size_time_dict (dict) :
        - Default: None
        - A dictionary representing the size and times taken to generate the iGraph graphs
        - Ideally, this dictionary is derived from "get_ig_..._generation_time"
        - If None, this function ends prematurely
    Returns
    -------
    - Visualization of NetworkX vs. iGraph generation time and their point-slope equations
    '''
    # ----- End Function Prematurely if Data Not Given --------------------------------------------
    if nx_size_time_dict is None or ig_size_time_dict is None:
        raise AttributeError('\033[31mENDING "vis_timing" PREMATURELY!\n'
                             'Please give BOTH nx_size_time_dict and ig_size_time_dict to function properly!\033[0m')

    # ----- Prepare Necessary Information For Visualization ---------------------------------------
    sizes = nx_size_time_dict['sizes']
    nx_times = nx_size_time_dict['times']
    ig_times = ig_size_time_dict['times']

    # Convert to arrays for regression
    sizes_arr = np.array(sizes)
    
    # Calculate Best Fit Slopes (y = mx + b)
    m_nx, b_nx = np.polyfit(sizes_arr, nx_times, 1)
    m_ig, b_ig = np.polyfit(sizes_arr, ig_times, 1)

    # ----- Visualize Data ------------------------------------------------------------------------
    # Visualization size
    plt.figure(figsize=(10, 6))

    # Plot NetworkX information
    plt.scatter(sizes, nx_times, color='red', label='NetworkX Data')
    plt.plot(sizes, m_nx*sizes_arr + b_nx, '--', color='red', label=f'NX Slope: {m_nx:.2e}')
    
    # Plot iGraph information
    plt.scatter(sizes, ig_times, color='green', label='iGraph Data')
    plt.plot(sizes, m_ig*sizes_arr + b_ig, '--', color='green', label=f'IG Slope: {m_ig:.2e}')

    # Plot annotations/labels then show the graph
    plt.title('Time Complexity: NX vs iGraph')
    plt.xlabel('Number of Edges')
    plt.ylabel('Time (Seconds)')
    plt.legend()
    plt.show()


def vis_big_o(pdDataFrame:pd.DataFrame, 
              nx_size_time_dict:dict=None,
              ig_size_time_dict:dict=None) -> None:
    '''
    About
    -----
    - Visualization specifically for comparing timing of NetworkX and iGraph in Big-O notation
    - The dictionary information is ideally from "get_nx/ig_..._generation_time"

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to use during
    - nx_size_time_dict (dict) :
        - Default: None
        - A dictionary representing the size and times taken to generate the NetworkX graphs
        - Ideally, this dictionary is derived from "get_nx_..._generation_time"
        - If None, this function ends prematurely
    - ig_size_time_dict (dict) :
        - Default: None
        - A dictionary representing the size and times taken to generate the iGraph graphs
        - Ideally, this dictionary is derived from "get_ig_..._generation_time"
        - If None, this function ends prematurely

    Raises
    ------
    - AttributeError
        - If nx_size_time_dict or ig_size_time_dict is None

    Returns
    -------
    - Visualization of NetworkX vs. iGraph timing in Big-O notation
    '''
    # ----- End Function Prematurely if Data Not Given --------------------------------------------
    if nx_size_time_dict is None or ig_size_time_dict is None:
        raise AttributeError('\033[31mENDING "vis_big_o" PREMATURELY!\n'
                             'Please give BOTH nx_size_time_dict and ig_size_time_dict to function properly!\033[0m')

    # ----- Prepare Necessary Information For Visualization ---------------------------------------
    sizes = nx_size_time_dict['sizes']
    nx_times = nx_size_time_dict['times']
    ig_times = ig_size_time_dict['times']

    # Derive Big-O exponent and intercept
    nx_exp, nx_intcpt = calculate_big_o(nx_size_time_dict)
    ig_exp, ig_intcpt = calculate_big_o(ig_size_time_dict)

    # Determine best-fit lines
    dense_sizes = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
    nx_fit = (10**nx_intcpt) * (dense_sizes**nx_exp)
    ig_fit = (10**ig_intcpt) * (dense_sizes**ig_exp)

    # ----- Visualize Data ------------------------------------------------------------------------
    # Visualization size
    plt.figure(figsize=(10, 6))

    # Plot data-points
    plt.scatter(sizes, nx_times, color='#E74C3C', label='NetworkX Actual', zorder=5)
    plt.scatter(sizes, ig_times, color='#2ECC71', label='igraph Actual', zorder=5)

    # Plot best-fit trend lines
    plt.plot(dense_sizes, nx_fit, '--', color='#E74C3C', label=f'NX: $O(n^{{{nx_exp:.4f}}})$')
    plt.plot(dense_sizes, ig_fit, '--', color='#2ECC71', label=f'IG: $O(n^{{{ig_exp:.4f}}})$')

    # Plot annotations/labels and show visualization
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Big-O Analysis: NetworkX vs. igraph')
    plt.xlabel('Number of Flow Records (n in log10)')
    plt.ylabel('Generation Time (Sec in log10)')
    plt.legend()
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.show()
    pass


def vis_bad_ips(pdDataFrame:pd.DataFrame,
                target:str = 'attack') -> None:
    '''
    About
    -----
    - Visualizes the bad IPs as a graph to determine if they have any value or not (0.0.0.0)

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to base the visualization off of
    - target (str) :    
        - The name of the categorical target identifier

    Returns
    -------
    - Visualization of the bad IPs as a graph
    '''
    # ----- Create Sub-Dataframe ------------------------------------------------------------------
    bad_ips_df = pdDataFrame[(pdDataFrame['source_ip'] == '0.0.0.0') | (pdDataFrame['destination_ip'] == '0.0.0.0')].head(100)
    
    if bad_ips_df.empty:
        return print("\033[32mNo 0.0.0.0 addresses found in this sample!\033[0m")

    # ----- Generate Graph ------------------------------------------------------------------------
    G = nx.from_pandas_edgelist(
        bad_ips_df, 
        source='source_ip', 
        target='destination_ip', 
        edge_attr=target,
        create_using=nx.Graph()
    )

    # ----- Define Node Colors --------------------------------------------------------------------
    # Initialize variables
    node_colors = []
    node_labels = {}
    degrees = dict(G.degree())

    for node in G.nodes():
        # Bad-IP Logic
        if node == '0.0.0.0':
            node_colors.append('#2ecc71')
            node_labels[node] = f"{node}\n(k={degrees[node]})"

        # Blue if benign, Red if malicious
        else:
            incident = G.edges(node, data=True)
            is_malicious = any(str(e[2][target]).lower() != 'benign' for e in incident)
            node_colors.append('#e74c3c' if is_malicious else '#3498db')
            node_labels[node] = ""

    # ----- Visualize Data ------------------------------------------------------------------------
    # Visualization size and graph type
    plt.figure(figsize=(14, 9))
    pos = nx.spring_layout(G, k=0.5, seed=3703)

    # Draw edges and nodes
    nx.draw_networkx_edges(G, pos, width=1.0, edge_color='#bdc3c7', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, edgecolors='black')
    
    # Draw labels
    label_pos = {k: [v[0], v[1] + 0.08] for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=9, font_weight='bold')

    # Draw legend
    green_dot = mlines.Line2D([], [], color='#2ecc71', marker='o', linestyle='None', markersize=10, label='Bad IP: (0.0.0.0)')
    red_dot = mlines.Line2D([], [], color='#e74c3c', marker='o', linestyle='None', markersize=10, label='Malicious')
    blue_dot = mlines.Line2D([], [], color='#3498db', marker='o', linestyle='None', markersize=10, label='Benign')
    plt.legend(handles=[green_dot, red_dot, blue_dot], loc='upper right', title="Security Profile")

    plt.title("0.0.0.0 IP Connectivity", fontsize=14)
    plt.axis('off')
    plt.show()

# =================================================================================================
# END Visualization Functions
# START Feature Engineering Functions
# =================================================================================================

def generate_reduced_graph_df(pdDataFrame: pd.DataFrame,
                              target: str = 'attack',
                              data_file: str = 'datasets/initial_complex.parquet') -> pd.DataFrame:
    '''
    About
    -----
    - Reduces network flow data into a graph-linked feature set
    - Collapses repeated connections into weighted edges
    - Engineers topological binary features (Star, Chain, Bridge)
    - Creates a parquet file based on "data_file"

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to reduce the information on
    - target (str) :
        - Default: attack
        - The name of the categorical attack identifier
    - data_file (str) :
        - Default: datasets/initial_complex.parquet
        - The name of the reduced data file to check for first before creation

    Returns
    -------
    - pd.DataFrame
        - The initial reduced dataframe for complex network feature engineering
    '''
    # ----- Initial Check If Dataset Already Exists -----------------------------------------------
    if os.path.exists(data_file):
        print(f'\033[32m{data_file} already exists! No need to recreate!\033[0m')
        return pd.read_parquet(data_file) 

    print(f'\033[33m{data_file} not detected. Attempting to create {data_file}...\033[0m')

    # Rename columns for clarity in graph representation
    df = pdDataFrame.rename(columns={
        'ipv4_src_addr': 'source_ip',
        'ipv4_dst_addr': 'destination_ip'
    })

    # ----- Grouping and Weight Aggregation -------------------------------------------------------
    # We group by src, dst, and target to preserve the structural dual-nature of compromised IPs
    reduced_df = df.groupby(['source_ip', 'destination_ip', target]).agg({
        'in_bytes': 'sum',
        'out_bytes': 'sum',
        'duration_in': 'mean'
    }).reset_index()

    # Create the continuous 'edge_weight' feature
    reduced_df['edge_weight'] = reduced_df['in_bytes'] + reduced_df['out_bytes'] + reduced_df['duration_in']

    # ----- Engineer Graph IDs --------------------------------------------------------------------
    # Generate the initial graph to find connected components (islands)
    G_temp = nx.from_pandas_edgelist(reduced_df, 'source_ip', 'destination_ip')
    components = list(nx.connected_components(G_temp))
    
    # Map nodes to their Graph IDs
    node_to_gid = {node: i for i, nodes in enumerate(components) for node in nodes}
    reduced_df['graph_id'] = reduced_df['source_ip'].map(node_to_gid)

    # ----- Engineer Topological IDs --------------------------------------------------------------
    # Initialize binary features for ML/NN training
    reduced_df['is_star_graph'] = 0
    reduced_df['is_chain_graph'] = 0
    reduced_df['is_bridge_link'] = 0

    # Iterate through each unique graph island to calculate diameter and bridges
    for gid in reduced_df['graph_id'].unique():
        subset = reduced_df[reduced_df['graph_id'] == gid]
        
        # Create iGraph instance for fast structural calculations
        edges = subset[['source_ip', 'destination_ip']].values
        g_ig = ig.Graph.TupleList(edges, directed=False)
        
        # A. Filter: Remove standalone nodes (Diameter = 1)
        diam = g_ig.diameter()
        if diam <= 1:
            reduced_df = reduced_df[reduced_df['graph_id'] != gid]
            continue

        # B. Geometry Classifiers
        is_star = 1 if diam == 2 else 0
        is_chain = 1 if diam > 2 else 0
        
        # C. Bridge Link Detection (Cut-edges)
        bridge_indices = g_ig.bridges()
        
        # Apply labels back to the dataframe subset
        idx = reduced_df[reduced_df['graph_id'] == gid].index
        reduced_df.loc[idx, 'is_star_graph'] = is_star
        reduced_df.loc[idx, 'is_chain_graph'] = is_chain
        
        # Mark specific edges as bridge links
        for b_idx in bridge_indices:
            edge = g_ig.es[b_idx]
            u, v = g_ig.vs[edge.source]['name'], g_ig.vs[edge.target]['name']
            
            # Locate the specific edge in the original dataframe (checking both directions)
            mask = (reduced_df['graph_id'] == gid) & (
                ((reduced_df['source_ip'] == u) & (reduced_df['destination_ip'] == v)) | 
                ((reduced_df['source_ip'] == v) & (reduced_df['destination_ip'] == u))
            )
            reduced_df.loc[mask, 'is_bridge_link'] = 1

    # ----- Final Slimming ------------------------------------------------------------------------
    final_cols = ['source_ip',
                  'destination_ip',
                  'edge_weight', 
                  'is_star_graph',
                  'is_chain_graph',
                  'is_bridge_link',
                  target]
    
    final_df = reduced_df[final_cols].reset_index(drop=True)
    
    # Save to Parquet
    final_df.to_parquet(data_file)
    print(f'\033[32mSuccessfully created {data_file}!\033[0m')
    
    return final_df


def remove_bad_ips(pdDataFrame:pd.DataFrame) -> pd.DataFrame:
    '''
    About
    -----
    - Removes source and destination IPs that are 0.0.0.0 to prevent structural noise.

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to remove 0.0.0.0 IPs from

    Returns
    -------
    - pd.DataFrame
        - A Pandas dataframe with all source and destination IPs of 0.0.0.0
    '''
    initial_count = len(pdDataFrame)
    
    # Filter out 0.0.0.0 from both source and destination
    df_clean = pdDataFrame[
        (pdDataFrame['source_ip'] != '0.0.0.0') & 
        (pdDataFrame['destination_ip'] != '0.0.0.0')
    ].copy()
    
    # Print off how many were removed
    removed = initial_count - len(df_clean)
    print(f"\033[32mRemoved {removed} rows containing 0.0.0.0.\033[0m")
    
    return df_clean


def generate_graph_ids(pdDataFrame:pd.DataFrame) -> pd.DataFrame:
    '''
    About
    -----
    - Identifies connected sub-graphs and assigns a graph ID to each entry denoting which connected graph the node belongs to

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to derive graph IDs from

    Returns
    -------
    - pd.DataFrame
        - The new Pandas dataframe with graph IDs attached to them
    '''
    # Build a temporary undirected graph for expediency
    G_temp = nx.from_pandas_edgelist(pdDataFrame, 'source_ip', 'destination_ip')
    
    # Find the connected sub-graphs
    components = list(nx.connected_components(G_temp))
    
    # Map nodes of the connected sub-graphs to a graph ID
    node_to_gid = {}
    for gid, nodes in enumerate(components):
        for node in nodes:
            node_to_gid[node] = gid
            
    # Append the graph ID to every entry
    pdDataFrame['graph_id'] = pdDataFrame['source_ip'].map(node_to_gid)
    
    return pdDataFrame


def generate_intensity_and_zscore(pdDataFrame:pd.DataFrame,
                                  target:str = 'attack') -> pd.DataFrame:
    '''
    About
    -----
    - Calculates a baseline edge_weight score and std for star and non-star graphs with only benign attack type entries
      then determines the respective edge_weight ratio and zscore for every entry compared to this baseline
    - Creates the following new features
        - baseline_edge_weight_ratio
        - baseline_edge_weight_zscore

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :  
        - The Pandas dataframe to generate the new features on
    - target (str) :
        - The name of the categorical attack type identifier

    Returns
    -------
    - pd.DataFrame
        - The new Pandas dataframe with the new information
    '''
    # Create temp benign only df
    benign_df = pdDataFrame[pdDataFrame[target].str.lower() == 'benign']
    
    # Generate baseline edge weight
    baseline_stats = benign_df.groupby('is_star_graph')['edge_weight'].agg(['mean', 'std']).rename(
        columns={'mean': 'baseline_mean', 'std': 'baseline_std'}
    )

    # Temporarily add baseline stats
    pdDataFrame = pdDataFrame.merge(baseline_stats, on='is_star_graph', how='left')

    # Establish intensity of the edge weight with the baseline edge weight
    pdDataFrame['baseline_edge_weight_ratio'] = pdDataFrame['edge_weight'] / pdDataFrame['baseline_mean']

    # Establish zscore of the edge weight
    pdDataFrame['baseline_edge_weight_zscore'] = (pdDataFrame['edge_weight'] - pdDataFrame['baseline_mean']) / pdDataFrame['baseline_std']

    # Remove temp baseline stats and return
    cols_to_drop = ['baseline_mean', 'baseline_std']
    return pdDataFrame.drop(columns=cols_to_drop)


def generate_complex_network_information(pdDataFrame:pd.DataFrame) -> pd.DataFrame:
    '''
    About
    -----
    - Generates the desired complex network information per unique graph_id and appends this as new features
    - Features Generated:
        - eigen_1 (The first eigenvalue)
        - eigen_2 (The second eigenvalue)
        - v1_src (The source node's component in the first eigenvector)
        - v2_src (The source node's component in the second eigenvector)
        - src_pagerank (The source node's PageRank)
        - dst_pagerank (The destination node's PageRank)
        - spectral_gap (The gap between the first and second eigenvalue)
        - global_convergence_steps (Total iterations for the entire graph to reach spectral stability)
        - node_convergence_steps (Iterations for the specific source node to reach spectral stability)

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame)
        - The Pandas dataframe to derive the complex network information on

    Returns
    -------
    - pd.DataFrame
        - The new Pandas dataframe with the complex network information
    '''
    # ----- Initial Setup -------------------------------------------------------------------------
    # Initialize the new features
    new_cols = [
        'eigen_1', 'eigen_2', 'v1_src', 'v2_src', 'src_pagerank', 'dst_pagerank', 
        'global_convergence_steps', 'node_convergence_steps'
    ]
    for col in new_cols:
        pdDataFrame[col] = 0.0

    # Establish the unique graph IDs
    unique_gids = pdDataFrame['graph_id'].unique()

    # ----- Generate Complex Network Information --------------------------------------------------
    for gid in unique_gids:
        # Create a subset of ONLY this graph ID
        subset = pdDataFrame[pdDataFrame['graph_id'] == gid]
        
        # Check if this graph is a star graph to determine convergence tracking
        is_star = subset['is_star_graph'].iloc[0] == 1
        
        # Setup the nodes, edges, and weights
        nodes = list(set(subset['source_ip']) | set(subset['destination_ip']))
        node_map = {name: i for i, name in enumerate(nodes)}
        edges = [(node_map[s], node_map[d]) for s, d in zip(subset['source_ip'], subset['destination_ip'])]
        weights = subset['edge_weight'].astype(float).values
        
        # Build the directed graph
        g_dir = ig.Graph(n=len(nodes), edges=edges, directed=True, edge_attrs={'weight': weights})
        
        # Derive PageRanks
        pr_scores = g_dir.pagerank(weights='weight')
        pr_lookup = dict(zip(nodes, pr_scores))
        
        # Transform directed to undirected (NECESSARY FOR EIGENVALUES)
        g_und = g_dir.as_undirected(mode="collapse", combine_edges="sum")
        
        # Derive Eigenvalues
        adj_sparse = g_und.get_adjacency_sparse(attribute='weight')
        
        # Initialize loop-specific variables
        eigval_1, eigval_2 = 0.0, 0.0
        v1_lookup, v2_lookup = {node: 0.0 for node in nodes}, {node: 0.0 for node in nodes}
        node_conv_lookup = {node: 0 for node in nodes}
        global_steps = 0
        
        try:
            # Scipy Matrix derivation method
            if adj_sparse.shape[0] >= 2:
                # return_eigenvectors=True is required to capture the actual structural vectors
                vals, vecs = eigsh(adj_sparse, k=2, which='LM', return_eigenvectors=True)
                
                # Sort to ensure e1 is always the principal eigenvalue
                sorted_idx = np.argsort(np.abs(vals))
                eigval_1, eigval_2 = np.abs(vals[sorted_idx[-1]]), np.abs(vals[sorted_idx[-2]])
                
                # Extract raw eigenvector components for mapping
                v1_raw = vecs[:, sorted_idx[-1]]
                v2_raw = vecs[:, sorted_idx[-2]]
                
                # Create lookups for the absolute values
                v1_lookup = dict(zip(nodes, np.abs(v1_raw)))
                v2_lookup = dict(zip(nodes, np.abs(v2_raw)))
                
                # ----- Derive Convergence Steps (Global and Node-Based) --------------------------
                n = adj_sparse.shape[0]
                v_curr = np.ones(n) / np.sqrt(n)
                epsilon = 1e-6
                max_iter = 500000  # Increased for better resolution
                damp = 0.85        # Damping factor to ensure stability
                
                node_steps = np.zeros(n, dtype=int)
                converged_mask = np.zeros(n, dtype=bool)

                for t in range(1, max_iter + 1):
                    # Power Step with Damping
                    v_next = (damp * (adj_sparse @ v_curr)) + ((1 - damp) * v_curr)
                    
                    norm = np.linalg.norm(v_next)
                    if norm == 0: break
                    v_next = v_next / norm
                    
                    # Identify entries that have converged
                    # We use a slightly more relaxed check for "Stability"
                    newly_converged = (np.abs(v_next - v_curr) < epsilon) & (~converged_mask)
                    node_steps[newly_converged] = t
                    converged_mask[newly_converged] = True
                    
                    v_curr = v_next
                    global_steps = t
                    
                    if converged_mask.all(): break
                
                # Fill non-converged with the max
                node_steps[~converged_mask] = max_iter
                node_conv_lookup = dict(zip(nodes, node_steps))
            
            # Standard math for simple matrices
            else:
                eigval_1 = np.linalg.norm(adj_sparse.toarray()) / np.sqrt(2)
                eigval_2 = 0.0
        except:
            eigval_1, eigval_2 = 0.0, 0.0
            
        # Map results back to respective information
        idx = subset.index
        pdDataFrame.loc[idx, 'eigen_1'] = eigval_1
        pdDataFrame.loc[idx, 'eigen_2'] = eigval_2
        pdDataFrame.loc[idx, 'v1_src'] = pdDataFrame.loc[idx, 'source_ip'].map(v1_lookup)
        pdDataFrame.loc[idx, 'v2_src'] = pdDataFrame.loc[idx, 'source_ip'].map(v2_lookup)
        pdDataFrame.loc[idx, 'global_convergence_steps'] = global_steps if not is_star else 0
        pdDataFrame.loc[idx, 'node_convergence_steps'] = pdDataFrame.loc[idx, 'source_ip'].map(node_conv_lookup)
        pdDataFrame.loc[idx, 'src_pagerank'] = pdDataFrame.loc[idx, 'source_ip'].map(pr_lookup)
        pdDataFrame.loc[idx, 'dst_pagerank'] = pdDataFrame.loc[idx, 'destination_ip'].map(pr_lookup)
        
    # Create the spectral_gap feature
    pdDataFrame['spectral_gap'] = (pdDataFrame['eigen_1'] - pdDataFrame['eigen_2']).abs()
    
    # Return new dataframe
    return pdDataFrame

# =================================================================================================
# END Feature Engineering Functions
# START Preprocessing Functions
# =================================================================================================

def preprocess_complex_data(pdDataFrame: pd.DataFrame) -> pd.DataFrame:
    '''
    About
    -----
    - Applies log-scaling and normalization to spectral and topological features to ensure convergence in Neural Network training

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to normalize and scale on

    Returns
    -------
    - pd.DataFrame
        - The new Pandas dataframe that is normalized and scaled
    '''
    # ----- Apply Target Encoding -----------------------------------------------------------------
    # Map the 'attack' strings to the same 0-20 IDs from the normal dataset
    attack_mapping = {'scanning': 16,
                      'benign': 2,
                      'ddos': 5,
                      'dos': 6,
                      'xss': 20,
                      'reconnaissance': 15,
                      'password': 13,
                      'injection': 11,
                      'brute_force': 4,
                      'fuzzers': 8,
                      'bot': 3,
                      'infilteration': 10,
                      'generic': 9,
                      'backdoor': 1,
                      'exploits': 7,
                      'ransomware': 14,
                      'mitm': 12,
                      'theft': 18,
                      'shellcode': 17,
                      'analysis': 0,
                      'worms': 19
                    }
    pdDataFrame['target'] = pdDataFrame['attack'].map(attack_mapping)

    # ----- Define Features And Their Respective Scaler/Transform ---------------------------------
    # Log transforms (They have massive ranges (10^0 to 10^8))
    log_cols = [
        'eigen_1',
        'eigen_2',
        'spectral_gap', 
        'global_convergence_steps', 
        'node_convergence_steps',
        'edge_weight',
        'baseline_edge_weight_ratio'
    ]
    
    # Linear scaling (They are already small decimals or balanced ratios)
    minmax_cols = ['src_pagerank', 'dst_pagerank']
    
    # Standard scaling (Generally for centering purposes)
    z_cols = ['baseline_edge_weight_zscore', 'v1_src', 'v2_src']

    # ----- Perform Scaling/Transfroms ------------------------------------------------------------
    # Sanity check before scaling
    pdDataFrame[log_cols + minmax_cols + z_cols] = pdDataFrame[log_cols + minmax_cols + z_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Apply log scaling
    for col in log_cols:
        if col in pdDataFrame.columns:
            pdDataFrame[col] = np.log1p(pdDataFrame[col].astype(float))
    
    # Apply linear scaling
    scaler_minmax = MinMaxScaler()
    pdDataFrame[minmax_cols] = scaler_minmax.fit_transform(pdDataFrame[minmax_cols])

    # Apply standard scaling
    scaler_std = StandardScaler()
    pdDataFrame[log_cols + z_cols] = scaler_std.fit_transform(pdDataFrame[log_cols + z_cols])

    return pdDataFrame

# =================================================================================================
# END Preprocessing Functions
# START Helper Functions
# =================================================================================================

def get_nx_graph_generation_time(pdDataFrame:pd.DataFrame,
                                 source: str = 'ipv4_src_addr',
                                 target: str = 'ipv4_dst_addr',
                                 cols_for_edge_weights: list[str] = ['in_bytes', 'out_bytes', 'duration_in'],
                                 nodes_to_generate: list[int] = [10, 100, 1000, 10000, 100000, 1000000]) -> dict:
    '''
    About
    -----
    - Gets the network generation time using NetworkX's DiGraph method

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to be used during graph generation
    - source (str) :
        - Default: ipv4_src_addr
        - The starting node in a graph (In this case ideally the source IP)
    - target (str) :
        - Default: ipv4_dst_addr
        - The destination node in a graph (In this case ideally the destination IP)
    - cols_for_edge_weights (list[str]) :
        - Default: [in_bytes, out_bytes, duration_in]
        - The column names to be grouped together per edge to aggregate a single weighted edge
    - nodes_to_generate (list[int]) :
        - Default: [10, 100, 1000, 10000, 100000, 1000000]
        - The number of nodes to generate in a graph to observe time changes

    Returns
    -------
    - results (dict)
        - Information of the size of the graph and the time it took to generate
    '''
    # Instance the results dict
    results = {'sizes': [], 'times': []}

    # Start the iterations
    for total_nodes in nodes_to_generate:
        sample_df = pdDataFrame.head(total_nodes)
        start = time()

        # Group repeat source/targets and aggregate a single edge
        df_weighted = sample_df.groupby([source, target])[cols_for_edge_weights].sum().reset_index()
        
        # Create the graph
        nx_graph = nx.from_pandas_edgelist(
            df=df_weighted,
            source=source,
            target=target,
            edge_attr=cols_for_edge_weights,
            create_using=nx.DiGraph() 
        )
        end = time()
        
        # Append results
        results['sizes'].append(total_nodes)
        results['times'].append(end - start)

    # Return results
    return results


def get_ig_graph_generation_time(pdDataFrame:pd.DataFrame,
                                 source: str = 'ipv4_src_addr',
                                 target: str = 'ipv4_dst_addr',
                                 cols_for_edge_weights: list[str] = ['in_bytes', 'out_bytes', 'duration_in'],
                                 nodes_to_generate: list[int] = [10, 100, 1000, 10000, 100000, 1000000]) -> dict:
    '''
    About
    -----
    - Gets the network generation time using iGraphs' directed graph

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to be used during graph generation
    - source (str) :
        - Default: ipv4_src_addr
        - The starting node in a graph (In this case ideally the source IP)
    - target (str) :
        - Default: ipv4_dst_addr
        - The destination node in a graph (In this case ideally the destination IP)
    - cols_for_edge_weights (list[str]) :
        - Default: [in_bytes, out_bytes, duration_in]
        - The column names to be grouped together per edge to aggregate a single weighted edge
    - nodes_to_generate (list[int]) :
        - Default: [10, 100, 1000, 10000, 100000, 1000000]
        - The number of nodes to generate in a graph to observe time changes

    Returns
    -------
    - results (dict)
        - Information of the size of the graph and the time it took to generate
    '''
    # Instance the results dict
    results = {'sizes': [], 'times': []}

    # Start the iterations
    for total_nodes in nodes_to_generate:
        sample_df = pdDataFrame.head(total_nodes)
        ig_df = sample_df[[source, target] + cols_for_edge_weights]
        start = time()

        # Create the directed graph
        ig_graph = ig.Graph.DataFrame(ig_df, directed=True, use_vids=False)
        
        # Imitate the aggregation in NetworkX to derive the same graph
        ig_graph.simplify(combine_edges={col: "sum" for col in cols_for_edge_weights})
        end = time()
        
        # Append results
        results['sizes'].append(total_nodes)
        results['times'].append(end - start)

    # Return results
    return results


def get_nx_adj_matrix_generation_time(pdDataFrame:pd.DataFrame,
                                      source:str = 'ipv4_src_addr',
                                      target:str = 'ipv4_dst_addr',
                                      cols_for_edge_weights:list[str] = ['in_bytes', 'out_bytes', 'duration_in'],
                                      nodes_to_generate:list[int] = [10, 100, 1000, 10000, 100000, 1000000]) -> dict:
    '''
    About
    -----
    - Gets the time to derive the adjacency matrix in NetworkX

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to be used during graph generation
    - source (str) :
        - Default: ipv4_src_addr
        - The starting node in a graph (In this case ideally the source IP)
    - target (str) :
        - Default: ipv4_dst_addr
        - The destination node in a graph (In this case ideally the destination IP)
    - cols_for_edge_weights (list[str]) :
        - Default: [in_bytes, out_bytes, duration_in]
        - The column names to be grouped together per edge to aggregate a single weighted edge
    - nodes_to_generate (list[int]) :
        - Default: [10, 100, 1000, 10000, 100000, 1000000]
        - The number of nodes to generate in a graph to observe time changes

    Returns
    -------
    - results (dict)
        - Information of the size of the graph and the time it took to generate the adjacency matrix
    '''
    # Initialize the results dict
    results = {'sizes': [], 'times': []}

    # Start iterations
    for total_nodes in nodes_to_generate:
        sample_df = pdDataFrame.head(total_nodes)
        
        # Group repeat source/targets and aggregate a single edge
        df_weighted = sample_df.groupby([source, target])[cols_for_edge_weights].sum().reset_index()

        # Create the graph
        nx_graph = nx.from_pandas_edgelist(
            df_weighted,
            source,
            target,
            cols_for_edge_weights,
            create_using=nx.DiGraph()
        )

        # Time adjacency matrix derivation
        start = time()
        _ = nx.to_scipy_sparse_array(nx_graph, weight=cols_for_edge_weights[0])
        end = time()
        
        # Append the generation times
        results['sizes'].append(total_nodes)
        results['times'].append(end - start)

    # Return the results dict
    return results


def get_ig_adj_matrix_generation_time(pdDataFrame:pd.DataFrame,
                                      source:str = 'ipv4_src_addr',
                                      target:str = 'ipv4_dst_addr',
                                      cols_for_edge_weights:list[str] = ['in_bytes', 'out_bytes', 'duration_in'],
                                      nodes_to_generate:list[int] = [10, 100, 1000, 10000, 100000, 1000000]) -> dict:
    '''
    About
    -----
    - Gets the time to derive the adjacency matrix in iGraph

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to be used during graph generation
    - source (str) :
        - Default: ipv4_src_addr
        - The starting node in a graph (In this case ideally the source IP)
    - target (str) :
        - Default: ipv4_dst_addr
        - The destination node in a graph (In this case ideally the destination IP)
    - cols_for_edge_weights (list[str]) :
        - Default: [in_bytes, out_bytes, duration_in]
        - The column names to be grouped together per edge to aggregate a single weighted edge
    - nodes_to_generate (list[int]) :
        - Default: [10, 100, 1000, 10000, 100000, 1000000]
        - The number of nodes to generate in a graph to observe time changes

    Returns
    -------
    - results (dict)
        - Information of the size of the graph and the time it took to generate the adjacency matrix
    '''
    # Initialize the results dict
    results = {'sizes': [], 'times': []}

    # Start the iterations
    for total_nodes in nodes_to_generate:
        sample_df = pdDataFrame.head(total_nodes)
        
        # Create the directed graph similar to the NetworkX version for consistency
        ig_graph = ig.Graph.DataFrame(sample_df[[source, target] + cols_for_edge_weights], directed=True, use_vids=False)
        ig_graph.simplify(combine_edges={col: "sum" for col in cols_for_edge_weights})

        # Time adjacency matrix derivation
        start = time()
        _ = ig_graph.get_adjacency_sparse(attribute=cols_for_edge_weights[0])
        end = time()
        
        # Append the generation times
        results['sizes'].append(total_nodes)
        results['times'].append(end - start)

    # Return the results dict
    return results


def get_nx_matrix_mult_generation_time(pdDataFrame:pd.DataFrame,
                                       source:str = 'ipv4_src_addr',
                                       target:str = 'ipv4_dst_addr',
                                       cols_for_edge_weights:list[str] = ['in_bytes', 'out_bytes', 'duration_in'],
                                       nodes_to_generate:list[int] = [10, 100, 1000, 10000, 100000, 1000000]) -> dict:
    '''
    About
    -----
    - Gets the time to perform matrix multiplication (squaring) using a NetworkX-derived matrix

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to be used during graph generation
    - source (str) :
        - Default: ipv4_src_addr
        - The starting node in a graph (In this case ideally the source IP)
    - target (str) :
        - Default: ipv4_dst_addr
        - The destination node in a graph (In this case ideally the destination IP)
    - cols_for_edge_weights (list[str]) :
        - Default: [in_bytes, out_bytes, duration_in]
        - The column names to be grouped together per edge to aggregate a single weighted edge
    - nodes_to_generate (list[int]) :
        - Default: [10, 100, 1000, 10000, 100000, 1000000]
        - The number of nodes to generate in a graph to observe time changes

    Returns
    -------
    - results (dict)
        - Information of the size of the graph and the time it took to multiply the matrix
    '''
    # Initialize the results dict
    results = {'sizes': [], 'times': []}

    # Start iterations
    for total_nodes in nodes_to_generate:
        sample_df = pdDataFrame.head(total_nodes)

        # Group repeat source/targets and aggregate a single edge
        df_weighted = sample_df.groupby([source, target])[cols_for_edge_weights].sum().reset_index()

        # Create the graph
        nx_graph = nx.from_pandas_edgelist(
            df_weighted,
            source,
            target,
            cols_for_edge_weights,
            create_using=nx.DiGraph()
        )
        
        # Obtain adjacency matrix
        adj_matrix = nx.to_scipy_sparse_array(nx_graph, weight=cols_for_edge_weights[0])

        # Time the matrix multiplication (A * A)
        start = time()
        _ = adj_matrix @ adj_matrix
        end = time()
        
        # Append results
        results['sizes'].append(total_nodes)
        results['times'].append(end - start)

    # Return the results dict
    return results


def get_ig_matrix_mult_generation_time(pdDataFrame: pd.DataFrame,
                                       source: str = 'ipv4_src_addr',
                                       target: str = 'ipv4_dst_addr',
                                       cols_for_edge_weights: list[str] = ['in_bytes', 'out_bytes', 'duration_in'],
                                       nodes_to_generate: list[int] = [10, 100, 1000, 10000, 100000, 1000000]) -> dict:
    '''
    About
    -----
    - Gets the time to perform matrix multiplication (squaring) using an iGraph-derived matrix

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to be used during graph generation
    - source (str) :
        - Default: ipv4_src_addr
        - The starting node in a graph (In this case ideally the source IP)
    - target (str) :
        - Default: ipv4_dst_addr
        - The destination node in a graph (In this case ideally the destination IP)
    - cols_for_edge_weights (list[str]) :
        - Default: [in_bytes, out_bytes, duration_in]
        - The column names to be grouped together per edge to aggregate a single weighted edge
    - nodes_to_generate (list[int]) :
        - Default: [10, 100, 1000, 10000, 100000, 1000000]
        - The number of nodes to generate in a graph to observe time changes

    Returns
    -------
    - results (dict)
        - Information of the size of the graph and the time it took to multiply the matrix
    '''
    # Initialize the results dict
    results = {'sizes': [], 'times': []}

    # Start iterations
    for total_nodes in nodes_to_generate:
        sample_df = pdDataFrame.head(total_nodes)

        # Create the directed graph similar to the NetworkX version for consistency
        ig_graph = ig.Graph.DataFrame(sample_df[[source, target] + cols_for_edge_weights], directed=True, use_vids=False)
        ig_graph.simplify(combine_edges={col: "sum" for col in cols_for_edge_weights})
        
        # Derive adjacency matrix
        adj_matrix = ig_graph.get_adjacency_sparse(attribute=cols_for_edge_weights[0])

        # Time the matrix multiplication (A * A)
        start = time()
        _ = adj_matrix @ adj_matrix
        end = time()
        
        # Append results
        results['sizes'].append(total_nodes)
        results['times'].append(end - start)

    # Return the results dict
    return results


def calculate_big_o(graph_generation_results:dict)-> tuple[float, float]:
    '''
    About
    -----
    - Derives the "Big-O" exponent and the intercept of some information

    Parameters
    ----------
    - graph_generation_results (dict) :
        - The results dict ideally from "get_ig/nx_graph_generation_time", or anything that imitates that structure

    Returns
    -------
    - o_exp (float) :
        - The "Big-O" exponent from the data sample
    - intercept (float) :
        - The "Big-O" intercept from the data sample
    '''
    # Use log 10 for consistency
    log_sizes = np.log10(graph_generation_results['sizes'])
    log_times = np.log10(graph_generation_results['times'])
    
    # Derive the exponent and intercept
    o_exp, intercept = np.polyfit(log_sizes, log_times, 1)

    # Return exponent and intercept
    return o_exp, intercept

# =================================================================================================
# END Helper Functions
# =================================================================================================