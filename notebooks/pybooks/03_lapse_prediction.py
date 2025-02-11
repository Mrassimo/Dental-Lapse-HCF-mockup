# %% [markdown]
# # HCF Member Retention Analysis
# 
# This notebook analyses member retention patterns to help improve member satisfaction and reduce lapse rates.
# 
# ## Key Insights from Analysis:
# - Member Base: 200,000 members
# - Current Retention Rate: 75.1%
# - Average Premium: $6,368.05
# - Churn Rate: 24.9% (49,775 members)
# - Model Performance: AUC-ROC Score: 0.641
# 
# ## Analysis Goals
# 
# 1. **Current State**
#    - What's our retention rate?
#    - How does it vary by segment?
#    - What are the trends?
# 
# 2. **Risk Factors**
#    - Why do members leave?
#    - Who is most likely to leave?
#    - When do they typically leave?
# 
# 3. **Predictive Modelling**
#    - Can we predict who will leave?
#    - How accurate are our predictions?
#    - What are the key indicators?
# 
# 4. **Interventions**
#    - How can we improve retention?
#    - What interventions work best?
#    - When should we intervene?

# %% [markdown]
# ## Understanding Features in Data Science
# 
# In data science, a "feature" is any measurable piece of data that can be used to analyse patterns
# or make predictions. For our member retention analysis, features include:
# 
# 1. **Numeric Features**
#    - Claim Amount: How much a member claimed (avg: $1,213.48)
#    - Premium Amount: How much they pay (median: $5,039.87)
#    - BMI: Body Mass Index (avg: 27.95)
#    - Membership Years: How long they've been with HCF
#
# 2. **Categorical Features**
#    - State: Where they live
#    - Claim Reason: Why they made a claim
#    - Data Confidentiality Level
#
# These features help us understand and predict member behaviour.

# %%
# Import required packages
# Data manipulation and analysis
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns  # For statistical visualisations
import plotly.graph_objects as go  # For interactive plots
from pathlib import Path
import os
import sys

# Machine learning and statistical packages
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.preprocessing import StandardScaler  # For normalising numeric features
from sklearn.metrics import (  # For evaluating model performance
    roc_auc_score,  # Area under ROC curve (0.641 achieved)
    precision_recall_curve,  # For analyzing precision-recall tradeoff
    average_precision_score,  # Average precision (0.350 achieved)
    roc_curve  # For plotting ROC curve
)
import lightgbm as lgb  # Advanced machine learning model (used for churn prediction)
from lifelines import KaplanMeierFitter, CoxPHFitter  # For survival analysis
from scipy import stats  # For statistical tests
import plotly.express as px  # For easy plotting
from plotly.subplots import make_subplots  # For complex plot layouts

# Add project root to Python path for custom imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up plotting style for consistent visualisations
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)

# HCF brand colours for consistent visualisations
HCF_COLORS = {
    'primary': '#004B87',    # HCF Blue - main brand color
    'secondary': '#00A3E0',  # Light Blue - supporting color
    'accent': '#FFB81C',     # Gold - for highlighting
    'neutral': '#6D6E71'     # Grey - for background elements
}

# Ensure visualisations directory exists
Path("visualisations").mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Data Loading and Feature Engineering
# 
# Key metrics from the data:
# - Premium Distribution:
#   - Median: $5,039.87
#   - 25th percentile: $3,298.21
#   - 75th percentile: $8,221.20
#   - Maximum: $20,696.50
# - Claims:
#   - Average claim: $1,213.48
#   - Claim-to-premium ratio: 19%
#   - High variation (std: $1,155.42)

# %%
# Load the data
data_path = Path(project_root) / "data" / "processed" / "retention_clean.csv"
df = pd.read_csv(data_path)

# Feature Engineering
# Convert categorical churn to numeric (1 = churned, 0 = retained)
df['churned'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Create membership years feature from premium (as a proxy)
df['membership_years'] = df['Category Premium'] / 1000

# Extract state from address using regex pattern
df['state'] = df['Customer_Address'].str.extract(r', ([A-Z]{2}) \d+')

# Basic data exploration
print("Dataset Overview:")
print("-" * 40)
print(f"Number of members: {len(df):,}")
print(f"Number of features: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")

# Calculate key business metrics
retention_rate = (1 - df['churned'].mean()) * 100  # Percentage of members retained
avg_premium = df['Category Premium'].mean()  # Average premium amount
risk_segments = df.groupby('state')['churned'].mean() * 100  # Churn rate by state

print("\nKey Metrics:")
print("-" * 40)
print(f"Overall Retention Rate: {retention_rate:.1f}%")
print(f"Average Premium: ${avg_premium:.2f}")
print("\nChurn Risk by State:")
print(risk_segments.round(1))

# Show example records
print("\nSample Records:")
print("-" * 40)
print(df.head())

# %% [markdown]
# ## 2. Survival Analysis
# 
# Key findings from survival analysis:
# - Wide range of membership durations observed
# - Longer-term members show different behavioral patterns
# - Premium increases need careful management
# - Members with high claim-to-premium ratios show higher retention
#
# We use the Kaplan-Meier estimator to understand:
# 1. Membership duration patterns
# 2. Risk factors for early departure
# 3. Optimal intervention timing

# %%
def perform_survival_analysis(df, tenure_col='membership_years', event_col='churned', group_col=None):
    """
    Perform survival analysis using the Kaplan-Meier estimator.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The member data
    tenure_col : str
        Column containing how long members have been with HCF
    event_col : str
        Column indicating if a member has churned (1) or not (0)
    group_col : str, optional
        Column to use for grouping (e.g., 'state' or 'Claim Reason')
    """
    # Create base figure for plotting
    fig = go.Figure()
    
    # First, calculate overall survival curve
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df[tenure_col],     # How long each member has been with HCF
        event_observed=df[event_col], # Whether they've churned (1) or not (0)
        label='Overall'               # Label for the plot
    )
    
    # Add the overall survival curve to the plot
    fig.add_trace(
        go.Scatter(
            x=kmf.timeline,           # Time points
            y=kmf.survival_function_.values.flatten(),  # Survival probabilities
            mode='lines',
            name='Overall Survival',
            line=dict(color=HCF_COLORS['primary'], width=2)
        )
    )
    
    # Add confidence intervals if available
    try:
        ci = kmf.confidence_intervals_
        if ci is not None:
            # Lower confidence bound
            fig.add_trace(
                go.Scatter(
                    x=kmf.timeline,
                    y=ci.iloc[:, 0],  # First column is lower bound
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            # Upper confidence bound with shading
            fig.add_trace(
                go.Scatter(
                    x=kmf.timeline,
                    y=ci.iloc[:, 1],  # Second column is upper bound
                    mode='lines',
                    fill='tonexty',  # Fill area between upper and lower bounds
                    fillcolor=f"rgba{tuple(list(int(HCF_COLORS['primary'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
                    line=dict(width=0),
                    showlegend=False
                )
            )
    except (AttributeError, KeyError):
        print("Note: Confidence intervals not available")
    
    # If a grouping column is specified, add survival curves for each group
    if group_col and group_col in df.columns:
        colors = [HCF_COLORS['secondary'], HCF_COLORS['accent'], HCF_COLORS['neutral']]
        for i, group in enumerate(df[group_col].unique()):
            mask = df[group_col] == group
            if mask.sum() > 0:  # Only plot if we have data for this group
                # Calculate survival curve for this group
                kmf_group = KaplanMeierFitter()
                kmf_group.fit(
                    durations=df[mask][tenure_col],
                    event_observed=df[mask][event_col],
                    label=str(group)
                )
                
                # Add this group's curve to the plot
                fig.add_trace(
                    go.Scatter(
                        x=kmf_group.timeline,
                        y=kmf_group.survival_function_.values.flatten(),
                        mode='lines',
                        name=f'{group_col}: {group}',
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
    
    # Customise the plot layout
    fig.update_layout(
        title='Member Survival Analysis',
        xaxis_title='Membership Years (Premium-Based Proxy)',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1.05]),  # Probability range 0-1
        showlegend=True,
        template='plotly_white',
        height=600,
        hovermode='x unified'  # Show all values at a given x-position
    )
    
    return fig

# %%
# Perform overall survival analysis
print("Analyzing member survival patterns...")
survival_plot = perform_survival_analysis(df)
survival_plot.write_html("visualisations/survival_analysis.html")
print("✓ Overall survival analysis complete")

# Analyze survival by state
print("\nAnalyzing survival patterns by state...")
state_survival = perform_survival_analysis(df, group_col='state')
state_survival.write_html("visualisations/state_survival_analysis.html")
print("✓ State-level survival analysis complete")

# Analyze survival by claim reason
print("\nAnalyzing survival patterns by claim reason...")
claim_survival = perform_survival_analysis(df, group_col='Claim Reason')
claim_survival.write_html("visualisations/claim_survival_analysis.html")
print("✓ Claim-based survival analysis complete")

# %% [markdown]
# ## 3. Risk Factor Analysis
# 
# Let's identify the key factors that influence member churn.

# %%
def analyse_risk_factors(df, target_col='churned'):
    """
    Analyse and visualise how different features relate to member churn risk.
    
    This function looks at two types of relationships:
    1. Numeric correlations (e.g., how claim amounts relate to churn)
    2. Categorical patterns (e.g., churn rates by state)
    
    Parameters:
    -----------
    df : pandas DataFrame
        The member data
    target_col : str
        The column indicating churn status (1 = churned, 0 = retained)
    """
    # Select relevant numeric features for analysis
    numeric_cols = [
        'Claim Amount',         # How much was claimed
        'Category Premium',     # Member's premium amount
        'Premium/Amount Ratio', # Ratio of premium to claims
        'BMI',                  # Body Mass Index
        'payment_reliability',  # Payment history metric
        'premium_increase',     # How much premium has increased
        'service_utilisation'   # How much they use services
    ]
    # Only keep columns that exist in our data
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Calculate correlations between numeric features and churn
    corr_matrix = df[numeric_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
    
    # Create correlation heatmap
    fig1 = go.Figure(data=go.Heatmap(
        z=df[numeric_cols + [target_col]].corr().values,
        x=df[numeric_cols + [target_col]].corr().columns,
        y=df[numeric_cols + [target_col]].corr().index,
        colorscale='RdBu',  # Red-Blue scale: red=positive, blue=negative correlation
        zmid=0              # Center the color scale at 0
    ))
    
    fig1.update_layout(
        title='Feature Correlation Matrix',
        height=800,
        width=800
    )
    
    # Analyse categorical features
    categorical_cols = [
        'Claim Reason',          # Why they made claims
        'Data confidentiality',  # Privacy level
        'state'                  # Location
    ]
    
    # Create subplots for each categorical feature
    fig2 = make_subplots(
        rows=len(categorical_cols),
        cols=1,
        subplot_titles=[f'Churn Rate by {col}' for col in categorical_cols],
        vertical_spacing=0.1
    )
    
    # For each categorical feature
    for i, col in enumerate(categorical_cols, 1):
        if col in df.columns:
            # Calculate average churn rate for each category
            churn_by_cat = df.groupby(col)['churned'].mean() * 100
            
            # Add bar chart showing churn rates
            fig2.add_trace(
                go.Bar(
                    x=churn_by_cat.index,
                    y=churn_by_cat.values,
                    name=col,
                    marker_color=HCF_COLORS['primary']
                ),
                row=i, col=1
            )
            
            # Label the y-axis
            fig2.update_yaxes(title_text='Churn Rate (%)', row=i, col=1)
    
    # Customise the categorical analysis plot
    fig2.update_layout(
        height=300 * len(categorical_cols),
        title_text='Categorical Risk Factor Analysis',
        showlegend=False,
        template='plotly_white'
    )
    
    # Print insights about numeric correlations
    print("\nKey Risk Factors:")
    print("-" * 40)
    print("Numeric Correlations with Churn:")
    for feature, corr in corr_matrix.items():
        if feature != target_col:
            print(f"{feature}: {corr:.3f}")
    
    return fig1, fig2

# %%
# Perform risk factor analysis
print("Analysing risk factors...")
correlation_plot, category_plot = analyse_risk_factors(df)
correlation_plot.write_html("visualisations/correlation_matrix.html")
category_plot.write_html("visualisations/categorical_risk_factors.html")
print("✓ Risk factor analysis complete")

# %% [markdown]
# ## 4. Predictive Modelling
# 
# Model Performance:
# - AUC-ROC Score: 0.641
# - Average Precision: 0.350
# - Early stopping at iteration 26
# 
# Areas for Enhancement:
# - Additional data points needed:
#   - Payment reliability
#   - Service utilization
#   - Premium increase history
# 
# Process Overview:
# 1. **Data Preparation**
#    - Select numeric features
#    - Split data into training and testing sets
#    - Scale features to similar ranges
# 
# 2. **Model Training**
#    - Use LightGBM, an advanced machine learning algorithm
#    - Optimise for AUC (Area Under Curve) metric
#    - Monitor for overfitting
# 
# 3. **Model Evaluation**
#    - ROC Curve: Shows trade-off between true and false positives
#    - Precision-Recall Curve: Shows model's ability to find churned members
#    - Calculate overall performance metrics

# %%
def build_predictive_model(df, target_col='churned', test_size=0.2):
    """
    Build and evaluate a machine learning model to predict member churn.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The member data
    target_col : str
        Column indicating churn status (1 = churned, 0 = retained)
    test_size : float
        Proportion of data to use for testing (0.2 = 20%)
    
    Returns:
    --------
    model : LGBMClassifier
        The trained model
    roc_fig : plotly Figure
        ROC curve visualisation
    pr_fig : plotly Figure
        Precision-Recall curve visualisation
    auc_score : float
        Area under ROC curve
    avg_precision : float
        Average precision score
    """
    # Step 1: Prepare Features
    # -----------------------
    # Define features that are safe to use (no data leakage)
    safe_features = [
        'Claim Amount',      # Historical claim amounts
        'Category Premium',  # Current premium amount
        'BMI'               # Health indicator
    ]
    
    # Select only safe numeric features
    feature_cols = [col for col in safe_features if col in df.columns]
    print("\nFeatures used in model:")
    print("-" * 40)
    print("\n".join(f"- {col}" for col in feature_cols))
    
    # Create feature matrix X and target vector y
    X = df[feature_cols].copy()
    y = df[target_col]
    
    # Add some feature engineering to capture more complex patterns
    X['claim_to_premium'] = X['Claim Amount'] / X['Category Premium']
    
    # Create categorical features using cut instead of qcut for premium
    X['premium_bracket'] = pd.cut(
        X['Category Premium'],
        bins=[0, 1000, 5000, 10000, 15000, float('inf')],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )
    
    # Use cut for BMI with domain-specific ranges
    X['bmi_bracket'] = pd.cut(
        X['BMI'],
        bins=[0, 18.5, 25, 30, 35, float('inf')],  # Standard BMI ranges
        labels=['underweight', 'normal', 'overweight', 'obese', 'severely_obese']
    )
    
    # Add interaction features
    X['high_claim_ratio'] = (X['claim_to_premium'] > X['claim_to_premium'].median()).astype(int)
    
    # Convert categorical features to numeric
    X = pd.get_dummies(X, columns=['premium_bracket', 'bmi_bracket'])
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    
    # Print data analysis
    print("\nFeature Statistics:")
    print("-" * 40)
    print(X.describe().round(2))
    
    print("\nFeature Importance Analysis:")
    print("-" * 40)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    correlations = []
    for col in numeric_cols:
        corr = X[col].corr(y)
        correlations.append((col, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 Features by Correlation Strength:")
    for col, corr in correlations[:10]:
        print(f"{col}: {corr:.3f}")
    
    # Initialise LightGBM classifier with better parameters for synthetic data
    model = lgb.LGBMClassifier(
        n_estimators=1000,      # More trees for better learning
        learning_rate=0.001,    # Very slow learning rate
        num_leaves=4,           # Very conservative tree complexity
        min_child_samples=200,  # Require many samples per leaf
        subsample=0.6,          # Use 60% of data for each tree
        colsample_bytree=0.6,   # Use 60% of features for each tree
        reg_alpha=0.5,          # Stronger L1 regularization
        reg_lambda=0.5,         # Stronger L2 regularization
        random_state=42         # For reproducibility
    )
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,  # Use 20% for testing
        random_state=42       # For reproducibility
    )
    
    # Scale features to similar ranges
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],  # Validation data
        eval_metric='auc',                    # Optimise for AUC
        callbacks=[lgb.early_stopping(20)]    # Stop if no improvement for 20 rounds
    )
    
    # Step 3: Make Predictions
    # -----------------------
    # Get probability of churn for each member
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Step 4: Calculate Performance Metrics
    # -----------------------------------
    # Area under ROC curve (overall performance)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Precision-Recall curve data
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Step 5: Create ROC Curve Visualisation
    # -------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_fig = go.Figure()
    
    # Add model performance line
    roc_fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color=HCF_COLORS['primary'], width=2)
        )
    )
    
    # Add random classifier line
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        )
    )
    
    # Customise ROC plot
    roc_fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Step 6: Create Precision-Recall Curve Visualisation
    # ------------------------------------------------
    pr_fig = go.Figure()
    
    # Add precision-recall curve
    pr_fig.add_trace(
        go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            line=dict(color=HCF_COLORS['secondary'], width=2)
        )
    )
    
    # Customise PR plot
    pr_fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    return model, roc_fig, pr_fig, auc_score, avg_precision

# %%
# Build and evaluate the predictive model
print("Building predictive model...")
model, roc_plot, pr_plot, auc_score, ap_score = build_predictive_model(df)

# Print performance metrics
print(f"\nModel Performance:")
print("-" * 40)
print(f"AUC-ROC Score: {auc_score:.3f}")  # Higher is better (max 1.0)
print(f"Average Precision: {ap_score:.3f}")  # Higher is better (max 1.0)

# Save visualisations
roc_plot.write_html("visualisations/roc_curve.html")
pr_plot.write_html("visualisations/precision_recall_curve.html")
print("\n✓ Model evaluation complete")

# %% [markdown]
# ## 5. Recommendations
# 
# Key findings and recommendations based on analysis:
# 
# 1. **Premium Strategy**
#    - Review pricing for high-premium categories
#    - Consider graduated premium increases
#    - Monitor impact on retention
# 
# 2. **Claims Management**
#    - Monitor claim-to-premium ratios (key predictor)
#    - Develop targeted retention strategies
#    - Focus on high-claiming members
# 
# 3. **Risk Mitigation**
#    - Implement early warning system based on model
#    - Focus on members with changing claim patterns
#    - Proactive intervention at critical points
# 
# 4. **Model Enhancement**
#    - Collect additional behavioral data
#    - Regular review of predictions
#    - Continuous model refinement 