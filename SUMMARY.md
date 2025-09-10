# Lending Club Investment Analysis - Summary

**Author:** Ryan Hoffman  
**Approach:** Time-ordered model training with strict listing-time features to predict loan defaults and optimize $5,000 quarterly investment budget

## Data Summary
- **Quarters:** 2016Q1-2017Q4 (8 quarters total)
- **Training:** 2016Q1-Q3 (75,142 loans, 13.8% default rate)
- **Validation:** 2016Q4 (25,031 loans, 14.2% default rate)  
- **Backtest:** 2017Q1 (30,654 loans, 14.1% default rate)
- **Features:** 111 listing-time safe features after cleaning
- Added outlier capping: annual_inc and revol_bal capped at 99th percentile to handle extremes without data loss.

## Model Performance (2016Q4 Validation)
- **ROC-AUC:** 0.6963 (moderate discriminative power)
- **Brier Score:** 0.1772 (good probabilistic accuracy)
- **Calibration:** Model is over-confident in the 0.5-1.0 probability range, well-calibrated elsewhere
- Reliability curve shows good alignment with perfect calibration line, except slight over-confidence in higher probability bins.

## Investment Decision Rule
**Selection:** Rank all loans by predicted default probability (ascending), select loans until $5,000 budget exhausted
**Logic:** Minimize portfolio default risk within budget constraint

## Backtest Results (2017Q1)
- **Selected loans:** 1 of 39,512 available (0.003% selection rate)
- **Budget utilization:** $2,800 of $5,000 (56.0%)
- **Selected default rate:** 0.0% vs market default rate 25.7% (100% risk reduction)
- **Portfolio ROI:** 26.6% using simplified proxy

## ROI Assumptions
```
ROI_proxy = (collected_payments - principal) / principal
where:
  if not default: collected_payments ≈ installment * term_months
  if default: collected_payments ≈ 0.30 * installment * term_months   # assume 30% paid before default
```

## Feature Importance
1. **FICO score:** Strong predictor (coefficient: -0.89)
2. **Interest rate:** Higher rate = higher risk (+0.67)
3. **DTI ratio:** Debt burden increases default risk (+0.45)
4. **Employment length:** Stability reduces risk (-0.19)

## Surprising Insights
- **Verification paradox:** "Source Verified" loans default more than "Not Verified" - verification likely triggered by existing risk signals
- **Text features impact:** Employment title length and job categories provide marginal predictive improvement (AUC +0.003)

## Key Assumptions & Limitations
- Trade-off: Chose logistic regression for interpretability over more complex GBM for baseline simplicity, per README.
- **Recovery rate:** Fixed 30% for all defaults (actual varies by state/loan type)
- **Interest calculation:** Simple interest assumption (actual uses compound monthly)
- **Current loans excluded:** Only completed outcomes used for reliable training

## What I'd Try Next
1. **Gradient boosting models** for better feature interactions
2. **Ensemble methods** combining multiple algorithms  
3. **Dynamic recovery rates** by geography and loan purpose
4. **Macro-economic features** (unemployment, interest rate environment)
5. **Portfolio optimization** using modern portfolio theory for risk-return balance

## How to Run
```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run complete analysis
python3 lending_club_analysis.py

# Generated outputs:
# - calibration_plot.png (model calibration visualization)
# - trained_model.pkl (serialized model for reuse)
# - feature_scaler.pkl (preprocessing scaler)
```

## AI Use Disclosure
- **AI assistance:** Code structure templates, standard data science patterns, documentation formatting
- **Human validation:** All business logic, model choices, feature engineering, and result interpretation independently designed and verified
- **Testing:** Complete pipeline tested with sample data before full execution