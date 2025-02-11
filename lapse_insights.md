# HCF Lapse Analysis Insights

## Overall Metrics
- Member Base: 200,000 members analyzed
- Current Retention Rate: 75.1%
- Average Premium: $6,368.05
- Churn Rate: 24.9% (49,775 members)

## Premium & Claims Patterns
- Premium Distribution:
  - Median: $5,039.87
  - 25% of members pay less than $3,298.21
  - 25% of members pay more than $8,221.20
  - Maximum premium: $20,696.50
- Claims:
  - Average claim amount: $1,213.48
  - Claim-to-premium ratio averages 19%
  - High variation in claims (std dev: $1,155.42)

## Risk Factors
### Strongest Predictors of Churn (by correlation):
1. Claim-to-premium ratio (0.197)
2. High claim ratio (0.162)
3. Category Premium (0.108)
4. Claim Amount (0.059)

### Member Health Indicators
- Average BMI: 27.95
- BMI shows minimal correlation with churn (-0.002)
- Mental health claims present in dataset

## Model Performance
- AUC-ROC Score: 0.641
- Average Precision: 0.350
- Early stopping at iteration 26
- Model shows moderate predictive power

## Key Findings
1. Premium Sensitivity:
   - Higher premiums correlate with increased churn risk
   - Premium increases need careful management

2. Claims Behavior:
   - Members with high claim-to-premium ratios show higher retention
   - Dental claims are frequent in the dataset

3. Member Tenure:
   - Wide range of membership durations observed
   - Longer-term members show different behavioral patterns

## Business Recommendations
1. Premium Strategy:
   - Review pricing strategy for high-premium categories
   - Consider graduated premium increases

2. Claims Management:
   - Monitor claim-to-premium ratios
   - Develop targeted retention strategies for high-claiming members

3. Risk Mitigation:
   - Implement early warning system based on identified risk factors
   - Focus on members with changing claim patterns

4. Model Enhancement:
   - Consider collecting additional data points:
     - Payment reliability
     - Service utilization
     - Premium increase history
   - These fields could improve model performance

## Action Items
1. Develop targeted retention programs for high-risk segments
2. Implement premium sensitivity analysis before increases
3. Create monitoring dashboard for claim-to-premium ratios
4. Enhance data collection for key missing metrics
5. Regular review of churn predictions for early intervention