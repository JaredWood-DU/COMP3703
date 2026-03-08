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
# Databasing
import numpy as np
import pandas as pd

# Networking
import networkx as nx
import igraph as ig

# Database splitting, encoding, scaling
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

# Visualizations
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# Timing
from time import time

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
    plt.figure(figsize=(14, 9))
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

    # Verification: In a star graph, the number of edges must equal the number of neighbors
    # and the diameter must be 2.
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

# =================================================================================================
# END Visualization Functions
# START Helper Function
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
# END Helper Function
# =================================================================================================