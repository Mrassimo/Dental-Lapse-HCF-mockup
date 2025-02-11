"""
HCF Data Visualisation Module
---------------------------

This module creates beautiful, interactive visualisations for the HCF data science project.
It's designed to be beginner-friendly with detailed explanations of plotting libraries.

Key Features:
- Interactive maps of dental centres
- Member segmentation visualisations
- Treatment pattern analysis
- Time series visualisations

Author: [Your Name]
Last Updated: [Date]
"""

# Essential Python Packages
# -----------------------
# pandas: Data manipulation (like Excel on steroids)
import pandas as pd

# numpy: Numerical computations (fast arrays and math)
import numpy as np

# plotly: Interactive visualizations (great for sharing)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# seaborn: Statistical visualizations (built on matplotlib)
import seaborn as sns

# matplotlib: The foundation for Python plotting
import matplotlib.pyplot as plt

# For Australian postcodes and locations
from geopy.geocoders import Nominatim


class HCFVisualizer:
    """
    Main visualisation class for HCF data analysis.
    
    This class creates various plots and charts to help understand:
    - Dental centre utilisation
    - Member behaviour patterns
    - Geographic distributions
    - Treatment trends
    """
    
    def __init__(self, style='whitegrid'):
        """
        Initialise the visualiser with a specific style.
        
        Parameters:
        -----------
        style : str
            The seaborn style to use (default: 'whitegrid')
            Other options: 'darkgrid', 'white', 'dark', 'ticks'
        """
        sns.set_style(style)
        self.geocoder = Nominatim(user_agent="hcf_analysis")
        
        # Set up colour schemes
        self.hcf_colors = {
            'primary': '#004B87',    # HCF Blue
            'secondary': '#00A3E0',  # Light Blue
            'accent': '#FFB81C',     # Gold
            'neutral': '#6D6E71'     # Grey
        }
    
    def plot_dental_map(self, df, suburb_col='suburb', size_col=None, color_col=None):
        """
        Create an interactive map of dental centres or member locations.
        """
        # Get coordinates for each suburb
        df = df.copy()
        
        if 'lat' not in df.columns:
            print("Geocoding suburbs (this may take a few minutes)...")
            
            # Create a dictionary to cache coordinates for each unique suburb
            unique_suburbs = df[suburb_col].unique()
            print(f"Total unique suburbs to geocode: {len(unique_suburbs)}")
            
            coords_cache = {}
            for i, suburb in enumerate(unique_suburbs, 1):
                if i % 10 == 0:  # Show progress every 10 suburbs
                    print(f"Progress: {i}/{len(unique_suburbs)} suburbs geocoded")
                try:
                    if suburb not in coords_cache:
                        location = self.geocoder.geocode(f"{suburb}, Australia")
                        coords_cache[suburb] = {
                            'lat': location.latitude if location else None,
                            'lon': location.longitude if location else None
                        }
                except Exception as e:
                    print(f"Warning: Could not geocode {suburb}: {str(e)}")
                    coords_cache[suburb] = {'lat': None, 'lon': None}
            
            # Apply cached coordinates to the dataframe
            df['lat'] = df[suburb_col].map(lambda x: coords_cache[x]['lat'])
            df['lon'] = df[suburb_col].map(lambda x: coords_cache[x]['lon'])
            
            print("Geocoding complete!")
        
        # Remove any rows with missing coordinates
        df = df.dropna(subset=['lat', 'lon'])
        print(f"Plotting {len(df)} locations with valid coordinates")
        
        # Create the map
        fig = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            size=size_col if size_col else None,
            color=color_col if color_col else None,
            hover_name=suburb_col,
            zoom=3,  # Zoom out to show all of Australia
            title='HCF Dental Centre Analysis',
            color_continuous_scale='viridis'
        )
        
        # Use OpenStreetMap style (doesn't require token)
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":30,"l":0,"b":0},
            height=800  # Make the map taller
        )
        
        return fig
    
    def plot_treatment_patterns(self, df, date_col='visit_date', treatment_col='treatment_type'):
        """
        Analyze and visualize treatment patterns over time.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create separate figures for different visualizations
        
        # 1. Monthly volume and treatment mix (2x1 subplot)
        fig1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Monthly Treatment Volume', 'Treatment Mix'),
            specs=[[{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # Monthly volume
        monthly = df.groupby(df[date_col].dt.to_period('M')).size()
        fig1.add_trace(
            go.Scatter(
                x=monthly.index.astype(str),
                y=monthly.values,
                name='Total Visits',
                line=dict(color=self.hcf_colors['primary'])
            ),
            row=1, col=1
        )
        
        # Treatment mix
        treatment_mix = df[treatment_col].value_counts()
        fig1.add_trace(
            go.Pie(
                labels=treatment_mix.index,
                values=treatment_mix.values,
                name='Treatment Mix',
                marker=dict(colors=[self.hcf_colors['primary'], 
                                  self.hcf_colors['secondary'],
                                  self.hcf_colors['accent'],
                                  self.hcf_colors['neutral']])
            ),
            row=1, col=2
        )
        
        fig1.update_layout(
            height=500,
            title_text='HCF Dental Treatment Volume and Mix',
            showlegend=True
        )
        
        # 2. Seasonal and daily patterns (2x1 subplot)
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Seasonal Patterns', 'Day of Week Patterns')
        )
        
        # Seasonal patterns
        seasonal = df.groupby(df[date_col].dt.month).size()
        fig2.add_trace(
            go.Bar(
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=seasonal.values,
                name='Monthly Pattern',
                marker_color=self.hcf_colors['primary']
            ),
            row=1, col=1
        )
        
        # Day of week patterns
        daily = df.groupby(df[date_col].dt.day_name()).size()
        fig2.add_trace(
            go.Bar(
                x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                y=daily.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']).values,
                name='Daily Pattern',
                marker_color=self.hcf_colors['secondary']
            ),
            row=1, col=2
        )
        
        fig2.update_layout(
            height=500,
            title_text='HCF Dental Visit Patterns',
            showlegend=True
        )
        
        return fig1, fig2
    
    def plot_member_segments(self, df, segment_col='segment', metrics=None):
        """
        Visualize member segments and their characteristics.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Member data with segment information
        segment_col : str
            Column containing segment labels
        metrics : list
            List of metrics to compare across segments
        
        Example:
        --------
        >>> metrics = ['visits_per_year', 'avg_cost', 'preventive_ratio']
        >>> visualizer.plot_member_segments(member_df, metrics=metrics)
        """
        if metrics is None:
            metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create subplot for each metric
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=metrics
        )
        
        for i, metric in enumerate(metrics, 1):
            # Calculate segment averages
            segment_avg = df.groupby(segment_col)[metric].mean().sort_values()
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=segment_avg.index,
                    y=segment_avg.values,
                    name=metric,
                    showlegend=False
                ),
                row=i, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=300 * len(metrics),
            title_text='Member Segment Analysis',
            showlegend=False
        )
        
        return fig
    
    def plot_retention_analysis(self, df, tenure_col='membership_years', event_col='churned'):
        """
        Analyze and visualize member retention patterns.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Member data
        tenure_col : str
            Column containing membership duration
        event_col : str
            Column indicating if member has left
        """
        # Create retention curve
        retention_data = (
            df.groupby(tenure_col)[event_col]
            .mean()
            .sort_index()
            .cumsum()
        )
        
        fig = go.Figure()
        
        # Add retention curve
        fig.add_trace(
            go.Scatter(
                x=retention_data.index,
                y=(1 - retention_data.values) * 100,
                mode='lines+markers',
                name='Retention Rate',
                line=dict(color=self.hcf_colors['primary'])
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Member Retention Analysis',
            xaxis_title='Membership Years',
            yaxis_title='Retention Rate (%)',
            showlegend=True
        )
        
        return fig


def main():
    """
    Example usage of the visualizer.
    
    This shows how to:
    1. Load data
    2. Create various visualizations
    3. Save or display the results
    """
    # Create visualizer
    viz = HCFVisualizer()
    
    try:
        # Load data
        dental_df = pd.read_csv('data/processed/dental_clean.csv')
        member_df = pd.read_csv('data/processed/retention_clean.csv')
        
        # Create visualizations
        dental_map = viz.plot_dental_map(dental_df)
        treatment_patterns = viz.plot_treatment_patterns(dental_df)
        member_segments = viz.plot_member_segments(member_df)
        retention_analysis = viz.plot_retention_analysis(member_df)
        
        # Save visualizations
        dental_map.write_html('visualisations/dental_map.html')
        treatment_patterns[0].write_html('visualisations/treatment_patterns_volume_mix.html')
        treatment_patterns[1].write_html('visualisations/treatment_patterns_seasonal_daily.html')
        member_segments.write_html('visualisations/member_segments.html')
        retention_analysis.write_html('visualisations/retention_analysis.html')
        
        print("✓ Visualisations created successfully")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("\nPlease ensure:")
        print("1. The processed data files exist")
        print("2. You have an internet connection (for maps)")
        print("3. The visualisations directory exists")


if __name__ == "__main__":
    main()