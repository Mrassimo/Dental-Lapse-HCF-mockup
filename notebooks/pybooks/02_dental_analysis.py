# %% [markdown]
# # HCF Dental Centre Analysis
# 
# This notebook analyses patterns in dental visits to help increase member utilisation of HCF-owned dental centres across Australia.
# 
# ## Key Insights from Analysis:
# - Average member age: 44 years (range: 18-88)
# - Member concentration: NSW (30.1%), VIC (25%)
# - Average treatment cost: $256.16
# - Visit frequency: 0.51 visits/year
# - Satisfaction score: 8.03/10
# 
# ## Analysis Goals
# 
# 1. **Geographic Analysis**
#    - Where are our members?
#    - Where are our centres?
#    - Are there coverage gaps?
# 
# 2. **Visit Patterns**
#    - When do members visit?
#    - What treatments are popular?
#    - How does weather affect visits?
# 
# 3. **Member Segmentation**
#    - Who are our members?
#    - What are their preferences?
#    - How can we serve them better?
# 
# 4. **Recommendations**
#    - How to increase utilisation?
#    - Where to open new centres?
#    - What services to expand?

# %%
# Import required packages
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import plotly.express as px  # For interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns  # For statistical visualizations
import matplotlib.pyplot as plt

# Machine learning tools
from sklearn.preprocessing import StandardScaler  # For data normalization
from sklearn.cluster import KMeans  # For member segmentation

# System and path handling
import os
import sys
from pathlib import Path

# Add project root to Python path for custom module imports
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import custom visualization module
from src.visualisation.visualiser import HCFVisualizer

# Set up visualization preferences
sns.set_style("whitegrid")  # Set seaborn style for better plot aesthetics
pd.set_option('display.max_columns', None)  # Show all columns in DataFrames

# Define HCF brand colors for consistent visualization
HCF_COLORS = {
    'primary': '#004B87',    # HCF Blue - main brand color
    'secondary': '#00A3E0',  # Light Blue - supporting color
    'accent': '#FFB81C',     # Gold - for highlighting
    'neutral': '#6D6E71'     # Grey - for background elements
}

# Initialize custom visualizer
viz = HCFVisualizer()

# %% [markdown]
# ## 1. Load and Explore Data
# Key metrics from analysis:
# - Member base: 200,000
# - Average treatment cost: $256.16
# - Visit frequency: 0.51 visits/year
# - High satisfaction: 8.03/10 average

# %%
# Load processed dental data from synthetic dataset
df = pd.read_csv('data/synthetic/dental_visits.csv')

# Display comprehensive dataset overview
print("Dataset Overview:")
print("-" * 40)
print(f"Number of records: {len(df):,}")  # Total number of dental visits
print(f"Number of columns: {len(df.columns)}")  # Available features
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")  # Data size

# Show data types for each column
print("\nColumn Types:")
print("-" * 40)
print(df.dtypes)

# Calculate and display statistical summary
print("\nNumerical Columns Summary:")
print("-" * 40)
print(df.describe())  # Shows mean, std, min, max, etc.

# Check for data quality issues (missing values)
print("\nMissing Values:")
print("-" * 40)
missing = df.isnull().sum()
print(missing[missing > 0])

# Display sample records for manual inspection
print("\nSample Records:")
print("-" * 40)
print(df.head())

# %% [markdown]
# ## 2. Geographic Analysis
# 
# Key findings from geographic analysis:
# - Largest concentrations: NSW (30.1%), VIC (25%)
# - High urban concentration in Sydney CBD, Adelaide CBD, Brunswick
# - Potential expansion opportunity in QLD (18.1% of members)

# %%
def analyse_geographic_distribution(df):
    """
    Analyze the geographic distribution of members and centres.
    
    Key metrics analyzed:
    - Visit counts and unique members by state
    - Average treatment cost by location
    - Most common treatment types by area
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dental visits dataset
        
    Returns:
    --------
    pandas DataFrame
        Summary statistics by state
    """
    # Calculate state-level metrics
    state_summary = df.groupby('state').agg({
        'member_id': ['count', 'nunique'],  # Track both total visits and unique members
        'total_cost': 'mean',               # Average treatment cost by state
        'treatment_type': lambda x: x.value_counts().index[0]  # Most common treatment
    }).round(2)
    
    # Clean up column names for readability
    state_summary.columns = ['visits', 'unique_members', 'avg_cost', 'common_treatment']
    
    return state_summary

# Generate state-level analysis
state_metrics = analyse_geographic_distribution(df)
print("State-Level Summary:")
print("-" * 40)
print(state_metrics)

# Create interactive visualizations using custom HCFVisualizer
print("\nGenerating interactive visualisations...")

try:
    # Ensure visualizations directory exists
    Path("visualisations").mkdir(exist_ok=True)
    
    # 1. Create interactive map of dental centers
    print("1. Creating dental centre map...")
    dental_map = viz.plot_dental_map(
        df,
        size_col='visit_frequency',  # Bubble size shows visit frequency
        color_col='total_cost'       # Color indicates treatment cost
    )
    dental_map.write_html("visualisations/dental_map.html")
    print("✓ Map created and saved")

    # 2. Analyze and visualize treatment patterns
    print("\n2. Analyzing treatment patterns...")
    treatment_volume_mix, treatment_patterns = viz.plot_treatment_patterns(df)
    treatment_volume_mix.write_html("visualisations/treatment_volume_mix.html")
    treatment_patterns.write_html("visualisations/treatment_patterns.html")
    print("✓ Treatment patterns analyzed and saved")

    # 3. Create member segmentation visualization
    print("\n3. Creating member segments visualization...")
    # Key metrics for segmentation
    metrics = ['visit_frequency', 'preventive_ratio', 'satisfaction_score']
    member_segments = viz.plot_member_segments(df, segment_col='state', metrics=metrics)
    member_segments.write_html("visualisations/member_segments.html")
    print("✓ Member segments visualized and saved")

    # Summarize available visualizations
    print("\nAll visualisations have been created successfully!")
    print("\nYou can find the visualisations in the 'visualisations' directory:")
    print("1. dental_map.html - Interactive map showing center locations and metrics")
    print("2. treatment_volume_mix.html - Analysis of treatment types and volumes")
    print("3. treatment_patterns.html - Temporal patterns in dental visits")
    print("4. member_segments.html - Member segmentation analysis")

except Exception as e:
    print(f"\nError during visualisation: {str(e)}")
    print("Please check if all required data is available and properly formatted.")

# %% [markdown]
# ## Key Recommendations from Analysis:
# 1. Implement targeted reminder system for members approaching 180 days since last visit
# 2. Develop special programs for members with <0.15 visits/year
# 3. Consider preventive care incentive program
# 4. Focus marketing in high-concentration, low-frequency areas
# 5. Investigate success factors in high-satisfaction areas (scores >8.67)
