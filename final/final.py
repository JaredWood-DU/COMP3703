'''
FILE OVERVIEW:
- Underlying code intended for use in final.ipynb to keep notebook cleaner
- Generally will use visualizations and functions produced from other .py files, so anything here is because
  it is specific for this .ipynb

=================================================

MISC COMMENTS:
- NA

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Visualization Functions
    - plot_complex_graph_binary
    - plot_complex_graph_multiclass
    - plot_key_comparison
- Main Function
    - create_key_takeaways
'''
# ----- Imports -----------------------------------------------------------------------------------
import matplotlib.lines as mlines
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm

# ----- Global Variables --------------------------------------------------------------------------
ATTACK_MAPPING = {'scanning': 16,
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

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def plot_complex_graph_binary(df, g_id, attack_mapping=ATTACK_MAPPING, node_size=150):
    # 1. Filter and Create Directed Graph
    subset = df[df['graph_id'] == g_id].copy()
    G = nx.from_pandas_edgelist(subset, 'source_ip', 'destination_ip', 
                                 edge_attr=True, create_using=nx.DiGraph())
    
    # 2. Map IDs to Names for the Legend
    # Create an inverse mapping: {ID: Name}
    id_to_name = {v: k for k, v in attack_mapping.items()}

    # 3. Determine Node Colors and Identify Present Classes
    node_status = {}
    for _, row in subset.iterrows():
        node_status[row['source_ip']] = row['target']
        node_status[row['destination_ip']] = row['target']

    colors = []
    unique_targets_in_graph = set()
    
    for node in G.nodes():
        status = node_status.get(node, 2)
        unique_targets_in_graph.add(status)
        # Benign is Blue, all others are Red
        colors.append('#3498db' if status == 2 else '#e74c3c')

    # 4. Plotting
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(G) 
    
    # Draw topology
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors, alpha=0.9, edgecolors='white')
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, width=1, arrows=True, arrowsize=12)
    
    # 5. Build Dynamic Legend based on what's actually in THIS graph
    legend_handles = []
    for target_id in sorted(list(unique_targets_in_graph)):
        name = id_to_name.get(target_id, f"ID {target_id}").capitalize()
        color = '#3498db' if target_id == 2 else '#e74c3c'
        
        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                               markersize=10, label=name)
        legend_handles.append(handle)

    plt.legend(handles=legend_handles, title="Traffic Type", loc='upper right', frameon=True)
    
    # 6. Title with Graph ID and Node Count
    node_count = G.number_of_nodes()
    plt.title(f"Structural Analysis: Graph ID {g_id} (Nodes: {node_count})", fontsize=15, pad=20)
    
    plt.axis('off')
    plt.show()


def plot_complex_graph_multiclass(df, g_id, attack_mapping=ATTACK_MAPPING, node_size=150):
    subset = df[df['graph_id'] == g_id].copy()
    G = nx.from_pandas_edgelist(subset, 'source_ip', 'destination_ip', 
                                 edge_attr=True, create_using=nx.DiGraph())
    
    id_to_name = {v: k for k, v in attack_mapping.items()}
    
    # 1. Identify all unique classes in this specific graph
    node_status = {}
    for _, row in subset.iterrows():
        node_status[row['source_ip']] = row['target']
        node_status[row['destination_ip']] = row['target']
    
    unique_targets = sorted(list(set(node_status.values())))
    
    # 2. Define the Color Palette
    # Benign (Target 2) is always the same Blue
    # Other attacks get distinct colors from a colormap
    benign_color = '#3498db'
    attack_cmap = cm.get_cmap('Set1', len(unique_targets)) 
    
    # Map each target ID to a color
    color_map = {}
    for i, t_id in enumerate(unique_targets):
        if t_id == 2:
            color_map[t_id] = benign_color
        else:
            # We use a color from the map, avoiding blue-ish tones if possible
            color_map[t_id] = attack_cmap(i % 9) 

    # 3. Assign colors to nodes
    node_colors = [color_map[node_status.get(node, 2)] for node in G.nodes()]

    # 4. Plotting
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(G) 
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, 
                           alpha=0.9, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, 
                           width=1.2, arrows=True, arrowsize=15, 
                           connectionstyle='arc3,rad=0.1')
    
    # 5. Build Legend with specific colors
    legend_handles = []
    for t_id in unique_targets:
        name = id_to_name.get(t_id, f"ID {t_id}").capitalize()
        handle = mlines.Line2D([], [], color=color_map[t_id], marker='o', 
                               linestyle='None', markersize=12, label=name)
        legend_handles.append(handle)

    plt.legend(handles=legend_handles, title="Network Entities", 
               loc='upper right', frameon=True, fontsize=10)
    
    node_count = G.number_of_nodes()
    plt.title(f"Multi-Attack Structural Analysis: Graph ID {g_id} (Nodes: {node_count})", 
              fontsize=16, pad=25, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_key_comparison():
    # Data extracted from your metrics (Standardized for the 6 key narrative classes)
    labels = ['Benign', 'Worms', 'Ransomware', 'Theft', 'Scanning', 'DDoS']

    # Recall Data (Ability to catch the attack)
    rec_tab = [0.598, 0.000, 0.588, 0.750, 0.937, 0.974]
    rec_comp = [0.999, 1.000, 1.000, 1.000, 0.767, 0.300]

    # Precision Data (Reliability of the alert)
    prec_tab = [0.995, 0.000, 0.833, 0.028, 0.485, 0.993]
    prec_comp = [0.996, 1.000, 1.000, 1.000, 0.889, 0.429]

    x = np.arange(len(labels))
    width = 0.35

    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Color Palette
    tab_color = '#bdc3c7'      # Neutral Grey for Baseline
    recall_color = '#27ae60'   # Green for Detection Success
    prec_color = '#27ae60'     # Green for Precision/Reliability

    # --- Plot 1: Recall ---
    rects1 = ax1.bar(x - width/2, rec_tab, width, label='Tabular (Normal)', color=tab_color, edgecolor='black', linewidth=0.5)
    rects2 = ax1.bar(x + width/2, rec_comp, width, label='Complex (Graph)', color=recall_color, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Recall Score', fontweight='bold', fontsize=12)
    ax1.set_title('Attack Understanding: How well can we differentiate between non-attacks and attacks', fontsize=14, pad=15, fontweight='bold')
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Plot 2: Precision ---
    rects3 = ax2.bar(x - width/2, prec_tab, width, label='Tabular (Normal)', color=tab_color, edgecolor='black', linewidth=0.5)
    rects4 = ax2.bar(x + width/2, prec_comp, width, label='Complex (Graph)', color=prec_color, edgecolor='black', linewidth=0.5)

    ax2.set_ylabel('Precision Score', fontweight='bold', fontsize=12)
    ax2.set_title('Alert Reliability: How many alerts were actually true?', fontsize=14, pad=15, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # Function to add value labels on top of bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax2)
    autolabel(rects4, ax2)

    plt.xlabel('Attack Category', fontsize=12, labelpad=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def create_key_takeaways(metrics_dir='metrics/'):
    # Helper to load and standardize
    def load_metrics(filename, suffix):
        df = pd.read_parquet(f"{metrics_dir}{filename}")
        # Reset index if attack_name is the index
        if 'attack_name' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'attack_name'})
        
        # Clean naming (lowercase, strip whitespace)
        df['attack_name'] = df['attack_name'].astype(str).str.strip().str.lower()
        
        # Filter for key columns and drop meta-rows (accuracy, etc.)
        df = df[~df['attack_name'].isin(['accuracy', 'macro avg', 'weighted avg'])]
        return df[['attack_name', 'recall']].rename(columns={'recall': f'Recall_{suffix}'}).set_index('attack_name')

    # 1. Load the 4 DataFrames
    ml_norm = load_metrics('ml_normal_metrics.parquet', 'ML_Tabular')
    ml_comp = load_metrics('ml_complex_metrics.parquet', 'ML_Complex')
    nn_norm = load_metrics('nn_normal_metrics.parquet', 'NN_Tabular')
    nn_comp = load_metrics('nn_complex_metrics.parquet', 'NN_Complex')

    # 2. Join all into one summary table
    summary = ml_norm.join([ml_comp, nn_norm, nn_comp], how='outer').fillna(0)

    # 3. Calculate the "Structural Impact" (The Delta for ML)
    summary['ML_Delta'] = summary['Recall_ML_Complex'] - summary['Recall_ML_Tabular']
    
    # 4. Filter for the "Main Narrative" categories
    # Focusing on stealth improvements and volumetric trade-offs
    key_categories = ['benign', 'worms', 'ransomware', 'theft', 'ddos', 'bot']
    takeaway_df = summary.loc[summary.index.intersection(key_categories)].copy()

    # 5. Add a narrative column for the report
    def get_insight(row):
        if row['ML_Delta'] > 0.2: return "Significant Structural Discovery"
        if row['ML_Delta'] < -0.2: return "Volumetric Trade-off (Tabular Better)"
        return "Consistent Performance"
    
    takeaway_df['Key_Insight'] = takeaway_df.apply(get_insight, axis=1)

    return takeaway_df.sort_values(by='ML_Delta', ascending=False)

# =================================================================================================
# END Main Function
# =================================================================================================