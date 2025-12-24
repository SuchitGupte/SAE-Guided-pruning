import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv('sae_analysis_results.csv')

# Plot 1: Effect of TopK on Unique Features (aggregated over tasks for clarity, or specific task)
def plot_topk_impact_faceted(df):
    """
    Replaces the single messy line plot with a FacetGrid (one subplot per task).
    Allows independent scales or clearer view of trends per task.
    """
    subset = df[df['Metric'] == 'Unique']
    
    # FacetGrid: Row per Task (or Col), Hue is TopK
    g = sns.FacetGrid(subset, col="Task", col_wrap=2, height=4, aspect=1.5, sharey=False)
    g.map_dataframe(
        sns.lineplot, 
        x="Layer", 
        y="Value", 
        hue="TopK", 
        palette="viridis", 
        marker="o",
        linewidth=2
    )
    
    g.add_legend(title="Top K")
    g.set_titles("{col_name}")
    g.set_axis_labels("Layer", "Count of Unique Features")
    
    # Add a global title
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Effect of TopK on Unique Features (Split by Task)")
    plt.savefig('plot_topk_faceted.png')
    plt.show()

# Plot 2: Unique vs Shared Stacked View (For a specific TopK, e.g., 100)
def plot_unique_vs_shared(df, fixed_k=100):
    plt.figure(figsize=(12, 6))
    subset = df[df['TopK'] == fixed_k]
    
    # We use a FacetGrid to show tasks side-by-side
    g = sns.FacetGrid(subset, col="Task", height=5, aspect=1)
    g.map_dataframe(
        sns.histplot,
        x="Layer",
        weights="Value",
        hue="Metric",
        multiple="stack",
        shrink=0.8,
        binwidth=2,
        palette={"Unique": "#FF6B6B", "Shared": "#4ECDC4"} # Red/Teal contrast
    )
    g.add_legend()
    g.fig.suptitle(f"Unique vs Shared Features (TopK={fixed_k})", y=1.05)
    plt.savefig('plot_unique_vs_shared.png')

def plot_composition_stacked(df, fixed_k=100):
    """
    Replaces the side-by-side bars with a 100% Stacked Bar Chart.
    Focuses on the RATIO of Unique vs Shared, which is crucial for pruning.
    """
    subset = df[df['TopK'] == fixed_k].copy()
    
    # Pivot to wide format for easier stacking calculation
    wide = subset.pivot_table(index=['Task', 'Layer'], columns='Metric', values='Value').reset_index()
    wide['Total'] = wide['Unique'] + wide['Shared']
    
    # Normalize to percentage
    wide['Unique_Pct'] = wide['Unique'] / wide['Total']
    wide['Shared_Pct'] = wide['Shared'] / wide['Total']
    
    # We plot this using a bar chart where 'Unique' is the bottom and 'Shared' is the top
    # But seaborn doesn't do stacked bars natively well, so we use pandas plot or raw matplotlib
    
    tasks = wide['Task'].unique()
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10, 3 * len(tasks)), sharex=True)
    
    if len(tasks) == 1: axes = [axes] # Handle single task case
    
    for ax, task in zip(axes, tasks):
        task_data = wide[wide['Task'] == task]
        
        # Plotting
        ax.bar(task_data['Layer'], task_data['Unique_Pct'], label='Unique', color='#FF6B6B', edgecolor='white')
        ax.bar(task_data['Layer'], task_data['Shared_Pct'], bottom=task_data['Unique_Pct'], label='Shared', color='#4ECDC4', edgecolor='white')
        
        ax.set_ylabel("Proportion")
        ax.set_title(f"Task: {task}")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Add legend only to the first plot to avoid clutter
        if task == tasks[0]:
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

    plt.xlabel("Layer")
    plt.tight_layout()
    plt.savefig('plot_composition_stacked.png')
    plt.show()


# Plot 3: Layer-wise "Task Specificity" (Ratio of Unique/Total)
def plot_specificity_ratio(df, fixed_k=100):
    # Pivot to get columns for Unique and Shared
    subset = df[df['TopK'] == fixed_k].pivot_table(
        index=['Task', 'Layer'], 
        columns='Metric', 
        values='Value'
    ).reset_index()
    
    subset['Specificity'] = subset['Unique'] / (subset['Unique'] + subset['Shared'])
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset, x="Layer", y="Specificity", hue="Task", marker="o", linewidth=2.5)
    plt.title(f"Task Specificity Ratio (Unique / Total Active) @ TopK={fixed_k}")
    plt.ylabel("Specificity (0 = Generic, 1 = Unique)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.savefig('plot_specificity_ratio.png')

sns.set_theme(style="whitegrid")

# Run plots
plot_topk_impact_faceted(df)
plot_composition_stacked(df, fixed_k=100)