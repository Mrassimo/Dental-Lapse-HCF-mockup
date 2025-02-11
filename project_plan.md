# HCF Data Science Project Plan

## Project Overview
This project demonstrates the potential of data science applications at HCF, focusing on two key business problems:
1. Increasing member utilization of HCF-owned dental centres
2. Predicting and preventing member lapse rates

## Current Results

### Member Retention
- Achieved 75.1% retention rate
- Identified key churn factors:
  - Premium amount (median: $5,039.87)
  - Claim patterns (avg claim: $1,213.48)
  - Membership duration
- Developed predictive model (AUC-ROC: 0.641)

### Dental Centre Analysis
- Current utilisation: 0.51 visits/year
- Average treatment cost: $1,213.48
- Member satisfaction: 7.8/10
- Geographic coverage: 85% within 10km

## Data Sources and Setup

### Dental Centre Analysis
- Primary dataset: [US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset)
- We'll augment this with synthetic dental visit data to simulate HCF's scenario

### Lapse Rate Analysis
- Primary dataset: [Insurance Customer Churn Dataset](https://www.kaggle.com/datasets/usmanfarid/customer-churn-dataset-for-life-insurance-industry)
- Will be modified to match HCF's context

## Project Structure

```
hcf_project/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Cleaned and transformed data
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

## Analytical Approaches

### Data Preparation (01_data_prep.py)
1. **Data Cleaning**
   - Handled missing values
   - Standardized currencies to AUD
   - Mapped regions to Australian states
   - Created consistent date formats

2. **Feature Engineering**
   - Created age groups and tenure segments
   - Calculated claim-to-premium ratios
   - Generated geographic distance metrics
   - Added seasonal indicators

3. **Synthetic Data Generation**
   - Created realistic dental visit patterns
   - Simulated Australian geographic distribution
   - Generated treatment preferences
   - Modeled member satisfaction scores

### Dental Centre Analysis (02_dental_analysis.py)
1. **Geographic Analysis**
   - Member distribution by state
   - Centre accessibility metrics
   - Coverage optimization

2. **Visit Patterns**
   - Frequency analysis
   - Treatment type distribution
   - Cost analysis
   - Satisfaction correlation

### Lapse Rate Prediction (03_lapse_prediction.py)
1. **Survival Analysis**
   - Kaplan-Meier estimation
   - Risk factor identification
   - Tenure pattern analysis

2. **Predictive Modeling**
   - LightGBM implementation
   - Feature importance analysis
   - Model performance:
     - AUC-ROC: 0.641
     - Average Precision: 0.350

3. **Recommendations**
   - Premium strategy optimization
   - Claims management improvements
   - Risk mitigation approaches
   - Model enhancement roadmap

## Required Python Packages
```python
pandas>=2.0.0
numpy>=1.24.3
scikit-learn>=1.2.2
xgboost>=1.7.5
lifelines>=0.27.4
lightgbm>=3.3.5
seaborn>=0.12.2
plotly>=5.13.1
kaggle>=1.6.17
faker>=16.0.0
matplotlib>=3.7.1
```

## Snowflake Integration Points
- Data ingestion pipelines
- Feature store implementation
- Model serving infrastructure
- Real-time scoring capabilities

## Next Steps
1. **Model Enhancement**
   - Collect additional behavioral data
   - Implement early warning system
   - Refine predictive features

2. **Implementation**
   - Deploy model in production
   - Set up monitoring dashboard
   - Create intervention workflow

3. **Business Integration**
   - Train staff on new tools
   - Establish feedback loops
   - Track ROI metrics

## Success Metrics
- Dental Centre Utilization: % increase in member visits
- Lapse Rate: Improvement in prediction accuracy
- ROI: Projected cost savings from better retention