# HCF Data Science Project

This project aims to improve HCF's business operations through data science applications, focusing on:
1. Increasing member utilisation of HCF-owned dental centres
2. Predicting and preventing member lapse rates

## Key Findings

### Member Retention Analysis
- Member Base: 200,000 members
- Current Retention Rate: 75.1%
- Average Premium: $6,368.05
- Churn Rate: 24.9% (49,775 members)
- Model Performance: AUC-ROC Score: 0.641

### Dental Analysis
- Average Visit Frequency: 0.51 visits/year
- Average Treatment Cost: $1,213.48
- Member Satisfaction Score: 7.8/10
- Geographic Coverage: 85% within 10km of centre

## Project Structure

```
hcf_project/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Cleaned data
│   └── synthetic/           # Generated mock data
├── notebooks/
│   ├── 01_data_prep.py         # Data cleaning and preparation
│   ├── 02_dental_analysis.py   # Dental centre analysis
│   └── 03_lapse_prediction.py  # Lapse rate prediction
├── src/
│   └── data_processing/
│       └── download_data.py     # Dataset download script
└── requirements.txt
```

## Setup Instructions

### 1. Environment Setup

#### On macOS:
```bash
# Install Python 3 if not already installed
brew install python3

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### On Windows:
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Kaggle API Setup

#### On macOS:
```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Download kaggle.json from your Kaggle account:
# 1. Go to https://www.kaggle.com
# 2. Click on your profile picture → "Account"
# 3. Scroll to "API" section and click "Create New API Token"
# 4. Move the downloaded kaggle.json to ~/.kaggle/
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

#### On Windows:
```bash
# Create Kaggle directory
mkdir %USERPROFILE%\.kaggle

# Move kaggle.json to the correct location
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### 3. Download and Process Data

```bash
# First, download the datasets
python src/data_processing/download_data.py

# Then, run the data preparation notebook
python notebooks/01_data_prep.py
```

The data preparation notebook will:
1. Clean and transform the downloaded data
2. Generate synthetic dental data
3. Save everything in the correct locations

### 4. Start Analysis

The analysis is split into three main notebooks:
1. `01_data_prep.py` - Data cleaning and preparation
   - Processes raw insurance and dental data
   - Generates synthetic data for testing
   - Implements feature engineering

2. `02_dental_analysis.py` - Analyzing dental centre utilization
   - Geographic distribution analysis
   - Visit pattern analysis
   - Treatment cost analysis
   - Member satisfaction metrics

3. `03_lapse_prediction.py` - Predicting member lapse rates
   - Survival analysis using Kaplan-Meier estimator
   - Risk factor analysis
   - Predictive modeling (LightGBM)
   - Recommendations for retention

Run these notebooks in VS Code to take advantage of the interactive Python features.

## Datasets Used

1. US Health Insurance Dataset
   - Source: https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset
   - Used for dental centre analysis

2. Insurance Customer Churn Dataset
   - Source: https://www.kaggle.com/datasets/usmanfarid/customer-churn-dataset-for-life-insurance-industry
   - Used for lapse rate prediction

3. Synthetic Dental Visit Data
   - Generated internally to simulate HCF's scenario
   - Used to augment the analysis

## Next Steps

1. Run the exploratory analysis notebook
2. Review initial insights
3. Develop predictive models
4. Create visualization dashboard

## Dependencies

- pandas>=2.0.0
- numpy>=1.24.3
- scikit-learn>=1.2.2
- xgboost>=1.7.5
- lifelines>=0.27.4
- lightgbm>=3.3.5
- seaborn>=0.12.2
- plotly>=5.13.1
- jupyter>=1.0.0
- kagglehub>=0.1.4
- kaggle>=1.6.17
- matplotlib>=3.7.1

## Troubleshooting

1. If you get permission errors on macOS:
   ```bash
   # Fix Kaggle API permissions
   chmod 600 ~/.kaggle/kaggle.json
   
   # Fix data directory permissions
   chmod -R 755 data/
   ```

2. If Python command not found on macOS:
   ```bash
   # Add Python to PATH
   export PATH="/usr/local/opt/python/libexec/bin:$PATH"
   ```

3. If Jupyter doesn't start:
   ```bash
   # Install Jupyter separately
   pip install jupyter notebook
   ``` 