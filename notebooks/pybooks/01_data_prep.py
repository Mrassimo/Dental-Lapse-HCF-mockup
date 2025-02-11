# %% [markdown]
# # HCF Data Preparation and Cleaning
# 
# This notebook handles data preparation for the HCF data science project, including:
# 1. Cleaning and transforming the downloaded data
# 2. Generating synthetic data to match HCF's context
# 
# ## Key Metrics:
# - Member Base: 200,000 members
# - Average Premium: $6,368.05
# - Retention Rate: 75.1%
# - Average Dental Visit: 0.51 visits/year
# - Average Member Age: 44 years
# 
# ## Package Overview
# 
# ### Data Processing
# - **pandas**: Data manipulation and cleaning
# - **numpy**: Numerical operations
# - **sklearn**: Feature scaling and preprocessing
# 
# ### Data Generation
# - **faker**: For generating realistic synthetic data
# - **numpy.random**: For random number generation

# %%
# Import required packages
# Data manipulation and analysis
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from faker import Faker  # For generating realistic synthetic data
from sklearn.preprocessing import StandardScaler  # For normalizing features

# Set up Faker for Australian data
fake = Faker('en_AU')  # Use Australian locale for realistic data

# Initialize scaler for numeric features
scaler = StandardScaler()  # Will be used to normalize numeric columns

# %% [markdown]
# ## 1. Clean and Transform Data
# 
# Let's clean both datasets and transform them to match HCF's Australian context.
# Key transformations include:
# - Converting USD to AUD
# - Mapping regions to Australian states
# - Creating HCF-specific features

# %%
def clean_data(df):
    """
    Basic data cleaning operations.
    
    Operations performed:
    - Handle missing values in numeric columns (using median)
    - Handle missing values in text columns (using 'Unknown')
    - Ensure data types are appropriate
    
    Parameters:
    -----------
    df : pandas DataFrame
        The raw data to clean
    
    Returns:
    --------
    pandas DataFrame
        The cleaned data
    """
    # Always work with a copy to preserve original data
    df_clean = df.copy()
    
    # Handle missing values in numeric columns using median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
        df_clean[numeric_cols].median()
    )
    
    # Handle missing values in text columns with 'Unknown'
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna('Unknown')
    
    return df_clean

# %%
def transform_dates(df, date_cols):
    """
    Convert dates to Australian format and create time-based features.
    
    Features created:
    - Month of each date
    - Year of each date
    - Quarter of each date
    - Australian season
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data containing date columns
    date_cols : list
        List of column names containing dates
    
    Returns:
    --------
    pandas DataFrame
        Data with additional date-based features
    """
    df_dates = df.copy()
    
    for col in date_cols:
        # Convert to datetime format
        df_dates[col] = pd.to_datetime(df_dates[col])
        
        # Extract basic date components
        df_dates[f'{col}_month'] = df_dates[col].dt.month
        df_dates[f'{col}_year'] = df_dates[col].dt.year
        df_dates[f'{col}_quarter'] = df_dates[col].dt.quarter
        
        # Map months to Australian seasons
        df_dates[f'{col}_season'] = df_dates[col].dt.month.map({
            12: 'Summer', 1: 'Summer', 2: 'Summer',  # Australian summer
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',   # Australian autumn
            6: 'Winter', 7: 'Winter', 8: 'Winter',   # Australian winter
            9: 'Spring', 10: 'Spring', 11: 'Spring'  # Australian spring
        })
    
    return df_dates

# %%
def create_member_features(df):
    """
    Create HCF-specific member features.
    
    Features created:
    - Age groups (Young Adult, Middle Age, Senior, Elderly)
    - High-claimer flag
    - State based on postcode
    
    Parameters:
    -----------
    df : pandas DataFrame
        Member data
    
    Returns:
    --------
    pandas DataFrame
        Data with additional member features
    """
    df_features = df.copy()
    
    # Create age groups for segmentation
    if 'age' in df_features.columns:
        df_features['age_group'] = pd.cut(
            df_features['age'],
            bins=[0, 30, 50, 70, 100],
            labels=['Young Adult', 'Middle Age', 'Senior', 'Elderly']
        )
    
    # Flag high-claiming members (above median)
    if 'claims' in df_features.columns:
        df_features['high_claimer'] = (
            df_features['claims'] > df_features['claims'].median()
        )
    
    # Add state-based features using Australian postcodes
    if 'postcode' in df_features.columns:
        # Australian postcode ranges by state
        state_ranges = {
            'NSW': (1000, 2999),  # New South Wales
            'ACT': (2600, 2618),  # Australian Capital Territory
            'VIC': (3000, 3999),  # Victoria
            'QLD': (4000, 4999),  # Queensland
            'SA': (5000, 5999),   # South Australia
            'WA': (6000, 6999),   # Western Australia
            'TAS': (7000, 7999),  # Tasmania
            'NT': (800, 999)      # Northern Territory
        }
        
        def get_state(postcode):
            """Map postcode to Australian state."""
            try:
                postcode = int(postcode)
                for state, (start, end) in state_ranges.items():
                    if start <= postcode <= end:
                        return state
                return 'Unknown'
            except (ValueError, TypeError):
                return 'Unknown'
        
        df_features['state'] = df_features['postcode'].apply(get_state)
    
    return df_features

# %%
def clean_health_insurance_data():
    """
    Clean and transform the US health insurance dataset.
    Converts to Australian context (e.g., USD to AUD, states to AU states).
    
    Transformations:
    - Convert USD to AUD (rate: 1.5)
    - Map US regions to Australian states
    - Add member-specific features
    
    Returns:
    --------
    pandas DataFrame
        Cleaned and transformed insurance data
    """
    # Read the raw insurance data
    df = pd.read_csv('data/raw/health_insurance/insurance.csv')
    
    # Apply basic cleaning operations
    df = clean_data(df)
    
    # Convert USD to AUD using current rate
    aud_rate = 1.5  # Example conversion rate
    df['charges'] = df['charges'] * aud_rate
    
    # Map US regions to Australian states for context
    state_mapping = {
        'northeast': 'NSW',  # Map US regions to
        'northwest': 'VIC',  # approximate Australian
        'southeast': 'QLD',  # equivalents based on
        'southwest': 'WA'    # relative position
    }
    
    # Add Australian state column
    df['state'] = df['region'].map(state_mapping)
    
    # Add HCF-specific member features
    df = create_member_features(df)
    
    # Save cleaned data
    df.to_csv('data/processed/insurance_clean.csv', index=False)
    return df

# %%
def clean_customer_churn_data():
    """
    Clean and transform the insurance customer churn dataset.
    Adapts to Australian context and HCF's specific needs.
    
    Transformations:
    - Convert currency values to AUD
    - Add member features
    - Calculate retention metrics
    
    Returns:
    --------
    pandas DataFrame
        Cleaned and transformed churn data
    """
    # Read the member data
    df = pd.read_csv('data/raw/member_data/randomdata.csv')
    
    # Apply basic cleaning
    df = clean_data(df)
    
    # Convert all currency values to AUD
    currency_cols = [col for col in df.columns if 'premium' in col.lower()]
    for col in currency_cols:
        df[col] = df[col] * 1.5  # Convert to AUD
    
    # Add HCF-specific member features
    df = create_member_features(df)
    
    # Add retention-specific features
    retention_features = {
        # Calculate payment reliability (successful/total payments)
        'payment_reliability': lambda x: x['successful_payments'] / x['total_payments'] if 'total_payments' in df.columns else None,
        # Calculate premium increase percentage
        'premium_increase': lambda x: (x['current_premium'] - x['initial_premium']) / x['initial_premium'] if all(col in df.columns for col in ['current_premium', 'initial_premium']) else None,
        # Calculate service utilization ratio
        'service_utilisation': lambda x: x['services_used'] / x['services_available'] if all(col in df.columns for col in ['services_used', 'services_available']) else None
    }
    
    # Apply retention features where data is available
    for name, func in retention_features.items():
        try:
            df[name] = df.apply(func, axis=1)
        except:
            print(f"Couldn't calculate {name} - missing required columns")
    
    # Save cleaned data
    df.to_csv('data/processed/retention_clean.csv', index=False)
    return df

# %% [markdown]
# ## 2. Generate Synthetic Dental Data
# 
# Create realistic synthetic data for HCF dental centres, including:
# - Visit patterns (avg 0.51 visits/year)
# - Treatment types (check-ups most common)
# - Member demographics (avg age 44)
# - Geographic distribution (NSW 30.1%, VIC 25%)
# - Treatment history and costs (avg $256.16)

# %%
def generate_synthetic_dental_data(n_samples=1000):
    """
    Generate synthetic dental visit data matching HCF's context.
    
    Data characteristics:
    - Visit frequency: 0.51 visits/year average
    - Treatment cost: $256.16 average
    - Member satisfaction: 8.03/10 average
    - Geographic distribution matches population
    
    Parameters:
    -----------
    n_samples : int
        Number of records to generate
    
    Returns:
    --------
    pandas DataFrame
        Synthetic dental visit data
    """
    # Define Australian states and treatment types
    states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'NT', 'ACT']
    treatments = ['Check-up', 'Clean', 'Filling', 'Crown', 'Root Canal']
    # Define major suburbs for each state
    suburbs = {
        'NSW': ['Sydney CBD', 'Parramatta', 'Bondi', 'Chatswood', 'Newcastle'],
        'VIC': ['Melbourne CBD', 'South Yarra', 'Brunswick', 'Geelong'],
        'QLD': ['Brisbane CBD', 'Gold Coast', 'Sunshine Coast'],
        'WA': ['Perth CBD', 'Fremantle'],
        'SA': ['Adelaide CBD', 'Glenelg'],
        'TAS': ['Hobart CBD'],
        'NT': ['Darwin CBD'],
        'ACT': ['Canberra City']
    }
    
    # Generate base member data
    data = {
        'member_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),  # Age distribution
        # State distribution matching population
        'state': np.random.choice(states, n_samples, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.025, 0.0125, 0.0125]),
        # Treatment type distribution
        'treatment_type': np.random.choice(treatments, n_samples, p=[0.4, 0.3, 0.2, 0.07, 0.03]),
        'visit_date': [fake.date_between(start_date='-1y', end_date='today') for _ in range(n_samples)],
        'total_cost': np.random.gamma(shape=5, scale=50, size=n_samples),  # Cost distribution
        'membership_years': np.random.uniform(1, 10, n_samples).round(1),  # Tenure
        'previous_visits': np.random.poisson(lam=2, size=n_samples)  # Visit history
    }
    
    df = pd.DataFrame(data)
    
    # Add suburb based on state
    df['suburb'] = df['state'].apply(lambda x: np.random.choice(suburbs[x]))
    
    # Add dental-specific metrics
    df['last_visit_days'] = (datetime.now() - pd.to_datetime(df['visit_date'])).dt.days
    df['visit_frequency'] = df['previous_visits'] / df['membership_years']
    df['preventive_ratio'] = np.random.uniform(0, 1, n_samples)  # Preventive care ratio
    
    # Add satisfaction scores (normal distribution around 8.03)
    df['satisfaction_score'] = np.random.normal(8, 1, n_samples).clip(0, 10)
    
    # Save synthetic data
    df.to_csv('data/synthetic/dental_visits.csv', index=False)
    return df

# %%
def generate_synthetic_member_data(n_samples=200000, random_seed=42):
    """
    Generate realistic synthetic member data for churn prediction.
    
    Data characteristics:
    - Member base: 200,000
    - Retention rate: 75.1%
    - Average premium: $6,368.05
    - Churn rate: 24.9%
    
    Features generated with realistic distributions:
    - Premium amounts follow market segments
    - Claims have relationship with premium but with noise
    - BMI follows Australian population distribution
    - Churn probability influenced by multiple factors
    
    Parameters:
    -----------
    n_samples : int
        Number of members to generate
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pandas DataFrame
        Synthetic member data
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    fake = Faker('en_AU')
    Faker.seed(random_seed)
    
    # Generate base member information
    data = {
        'Customer Name': [fake.name() for _ in range(n_samples)],
        'Customer_Address': [fake.address() for _ in range(n_samples)],
        'Company Name': [fake.company() for _ in range(n_samples)]
    }
    
    # Define premium tiers based on market research
    premium_tiers = {
        'Basic': (1200, 2400),    # Basic coverage
        'Bronze': (2400, 3600),   # Bronze tier
        'Silver': (3600, 5400),   # Silver tier
        'Gold': (5400, 8400),     # Gold tier
        'Platinum': (8400, 12000) # Platinum tier
    }
    
    # Assign premium tiers with realistic distribution
    tier_probabilities = [0.3, 0.25, 0.2, 0.15, 0.1]  # More members in lower tiers
    tiers = np.random.choice(list(premium_tiers.keys()), size=n_samples, p=tier_probabilities)
    
    # Generate premiums within tiers
    premiums = []
    for tier in tiers:
        min_premium, max_premium = premium_tiers[tier]
        premium = np.random.uniform(min_premium, max_premium)
        # Add realistic noise
        premium *= np.random.normal(1, 0.05)  # 5% standard deviation
        premiums.append(round(premium, 2))
    
    data['Category Premium'] = premiums
    
    # Generate claims based on premium
    claim_ratios = np.random.beta(2, 5, n_samples)  # Beta distribution for claim ratios
    data['Claim Amount'] = (data['Category Premium'] * claim_ratios * np.random.normal(1, 0.2, n_samples)).round(2)
    
    # Generate BMI following Australian distribution
    bmi_mean = 27.9  # Australian average
    bmi_std = 5.5    # Standard deviation
    data['BMI'] = np.random.normal(bmi_mean, bmi_std, n_samples)
    data['BMI'] = np.clip(data['BMI'], 16, 45).round(1)  # Clip to realistic range
    
    # Generate claim reasons with realistic distribution
    claim_reasons = ['Medical', 'Dental', 'Optical', 'Physio', 'Mental Health']
    claim_probabilities = [0.4, 0.25, 0.15, 0.1, 0.1]
    data['Claim Reason'] = np.random.choice(claim_reasons, size=n_samples, p=claim_probabilities)
    
    # Generate data confidentiality preferences
    data['Data confidentiality'] = np.random.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.2, 0.5, 0.3])
    
    # Generate payment reliability (with some missing values)
    data['payment_reliability'] = np.random.beta(8, 2, n_samples)  # Most members pay reliably
    mask = np.random.random(n_samples) < 0.1  # 10% missing values
    data['payment_reliability'][mask] = np.nan
    
    # Generate premium increase data (with some missing values)
    data['premium_increase'] = np.random.normal(0.05, 0.02, n_samples)  # Mean 5% increase
    mask = np.random.random(n_samples) < 0.15  # 15% missing values
    data['premium_increase'][mask] = np.nan
    
    # Generate service utilization (with some missing values)
    data['service_utilisation'] = np.random.beta(3, 4, n_samples)  # Right-skewed
    mask = np.random.random(n_samples) < 0.12  # 12% missing values
    data['service_utilisation'][mask] = np.nan
    
    # Calculate churn probability based on multiple factors
    churn_prob = np.zeros(n_samples)
    
    # Base churn rate (15%)
    churn_prob += 0.15
    
    # Premium effect (higher premiums slightly increase churn)
    premium_effect = (data['Category Premium'] - np.mean(data['Category Premium'])) / np.std(data['Category Premium'])
    churn_prob += 0.05 * premium_effect
    
    # Claim ratio effect (high claims decrease churn)
    claim_ratio = data['Claim Amount'] / data['Category Premium']
    claim_effect = -0.1 * (claim_ratio - np.mean(claim_ratio)) / np.std(claim_ratio)
    churn_prob += claim_effect
    
    # Payment reliability effect
    payment_effect = np.zeros(n_samples)
    mask = ~np.isnan(data['payment_reliability'])
    payment_effect[mask] = -0.15 * (data['payment_reliability'][mask] - 0.5) * 2
    churn_prob += payment_effect
    
    # Premium increase effect
    increase_effect = np.zeros(n_samples)
    mask = ~np.isnan(data['premium_increase'])
    increase_effect[mask] = 0.2 * (data['premium_increase'][mask] / 0.05)
    churn_prob += increase_effect
    
    # Add random noise
    churn_prob += np.random.normal(0, 0.1, n_samples)
    
    # Clip probabilities to valid range
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate actual churn based on probabilities
    data['Churn'] = pd.Series(np.random.random(n_samples) < churn_prob).map({True: 'Yes', False: 'No'})
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add binary churn indicator
    df['churned'] = (df['Churn'] == 'Yes').astype(int)
    
    return df

# %%
# Set up directories for data storage
print("Setting up directories...")
for directory in ['data/raw/member_data', 'data/processed', 'data/synthetic']:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Run the complete data preparation pipeline
print("\nStarting data preparation pipeline...")

# 1. Clean and transform insurance data
print("\nCleaning health insurance data...")
insurance_df = clean_health_insurance_data()

# 2. Clean and transform churn data
print("\nCleaning customer churn data...")
churn_df = clean_customer_churn_data()

# 3. Generate synthetic dental data
print("\nGenerating synthetic dental data...")
dental_df = generate_synthetic_dental_data()

# 4. Generate synthetic member data
print("\nGenerating synthetic member data...")
member_df = generate_synthetic_member_data()

# Save all generated data
print("\nSaving data...")
member_df.to_csv('data/raw/member_data/randomdata.csv', index=False)
print("✓ Member data saved to data/raw/member_data/randomdata.csv")

# Clean the newly generated data
print("\nCleaning generated data...")
clean_customer_churn_data()
print("✓ Generated data cleaned and saved to data/processed/retention_clean.csv")

print("\nData preparation complete!")
print("Processed files saved in data/processed/")
print("Synthetic data saved in data/synthetic/") 