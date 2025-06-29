import pandas as pd
import numpy as np
from typing import Tuple, Dict
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Plot style configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

def run_macroprudential_policy_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze the impact of macroeconomic indicators on macroprudential policy implementation.
    
    Analysis includes:
    1. Policy activation probability model (logit)
    2. Policy activation intensity model (negative binomial/OLS)
    3. Marginal effects calculation
    4. Model performance evaluation
    
    Parameters:
        df (pd.DataFrame): Analysis dataset
        
    Returns:
        Dict: Analysis results dictionary
    """
    
    # More comprehensive macroeconomic indicators
    macro_indicators = [
        'credit_gap_actual',  # Credit gap (main financial imbalance indicator)
        'residential_property_price_yoy_changes_pct',  # Housing price growth rate
        'debt_service_ratio_private_non_financial_sector',  # Private debt service ratio
        'central_bank_policy_rate_obs_value',  # Monetary policy stance
        'total_credit_private_non_financial_sector_all_sector',  # Private credit total
        'effective_exchange_rate_real',  # Real effective exchange rate
        'commercial_property_price_obs_value'  # Commercial real estate prices
    ]
    
    cols_to_use = ['obs_bin', 'obs_agg', 'country_code', 'year'] + macro_indicators

    # Data preprocessing
    print("=== Data Preprocessing ===")
    
    # Select only available indicators
    available_indicators = [col for col in macro_indicators if col in df.columns]
    cols_to_use = ['obs_bin', 'obs_agg', 'country_code', 'year'] + available_indicators
    
    print(f"Using macroeconomic indicators: {available_indicators}")
    
    # Handle missing values
    df_clean = df[cols_to_use].copy()
    
    # Convert to numeric
    numeric_cols = ['obs_bin', 'obs_agg'] + available_indicators
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Missing value handling before/after sample size
    print(f"Sample size before processing: {len(df_clean)}")
    df_clean = df_clean.dropna()
    print(f"Sample size after processing: {len(df_clean)}")
    print(f"Policy activation observations: {df_clean['obs_bin'].sum()}")
    
    if len(df_clean) == 0:
        raise ValueError("No analyzable data available")
    
    # Standardize macroeconomic indicators (for easier interpretation)
    scaler = StandardScaler()
    df_clean[available_indicators] = scaler.fit_transform(df_clean[available_indicators])
    
    # Multicollinearity check: remove variables with high correlation
    corr_matrix = df_clean[available_indicators].corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print(f"High correlation variable pairs: {high_corr_pairs}")
        # Remove the second variable in high correlation pairs
        vars_to_remove = set()
        for pair in high_corr_pairs:
            vars_to_remove.add(pair[1])
        available_indicators = [var for var in available_indicators if var not in vars_to_remove]
        print(f"Indicators after removal: {available_indicators}")
    
    # Fixed effects (year only - country fixed effects omitted for small sample sizes)
    df_clean = pd.get_dummies(df_clean, columns=['year'], prefix='year', drop_first=True)
    
    # Prepare explanatory variables (macro indicators + year fixed effects)
    macro_cols = available_indicators
    year_cols = [col for col in df_clean.columns if col.startswith('year_')]
    X_cols = macro_cols + year_cols
    
    # Perfect separation check (remove variables that lead to 0 or 1 only for policy activation)
    X_data = df_clean[X_cols]
    policy_activated = df_clean['obs_bin'] == 1
    policy_not_activated = df_clean['obs_bin'] == 0
    
    vars_to_keep = []
    for col in X_cols:
        if col.startswith('year_'):
            # Keep year dummies with variation
            if X_data[col].nunique() > 1:
                vars_to_keep.append(col)
        else:
            # Check for perfect separation in macro indicators
            activated_values = X_data.loc[policy_activated, col]
            not_activated_values = X_data.loc[policy_not_activated, col]
            if len(activated_values) > 0 and len(not_activated_values) > 0:
                # Check if value ranges overlap
                if (activated_values.min() <= not_activated_values.max() and 
                    not_activated_values.min() <= activated_values.max()):
                    vars_to_keep.append(col)
    
    X_cols = vars_to_keep
    print(f"Final number of explanatory variables: {len(X_cols)}")
    
    if len(X_cols) == 0:
        raise ValueError("No usable explanatory variables available")
    
    X = sm.add_constant(df_clean[X_cols].astype(float))
    y_binary = df_clean['obs_bin'].astype(int)
    y_count = df_clean['obs_agg'].astype(int)
    
    # Variance inflation factor (VIF) check
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        high_vif = vif_data[vif_data["VIF"] > 10]
        if len(high_vif) > 0:
            print(f"Variables with high VIF values: \\n{high_vif}")
    except:
        pass
    
    # === Model 1: Policy activation probability (logit model) ===
    print("\n=== Model 1: Policy Activation Probability Analysis ===")
    try:
        logit_model = sm.Logit(y_binary, X).fit(disp=False, maxiter=1000)
    except:
        print("Logit model did not converge. Trying regularized logit.")
        # Regularized logit (L1 regularization)
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        
        lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        lr.fit(X.iloc[:, 1:], y_binary)  # Exclude constant term
        
        # Create statsmodels-like object
        class SimpleLogitResult:
            def __init__(self, lr_model, X, y):
                self.params = pd.Series(lr_model.coef_[0], index=X.columns[1:])
                self.params['const'] = lr_model.intercept_[0]
                self.params = self.params[X.columns]  # Match order
                
                # Calculate predicted values
                y_pred_proba = lr_model.predict_proba(X.iloc[:, 1:])[:, 1]
                self.fittedvalues = y_pred_proba
                
                # Simple statistics
                from sklearn.metrics import log_loss
                self.llf = -log_loss(y, y_pred_proba) * len(y)
                self.pvalues = pd.Series([0.5] * len(self.params), index=self.params.index)  # Dummy values
                
        logit_model = SimpleLogitResult(lr, X, y_binary)
    
    # === Model 2: Policy activation intensity (negative binomial model) ===
    print("\n=== Model 2: Policy Activation Intensity Analysis ===")
    try:
        nb_model = sm.NegativeBinomial(y_count, X).fit(disp=False)
    except:
        print("Negative binomial model did not converge. Using OLS model.")
        nb_model = sm.OLS(y_count.astype(float), X).fit()
    
    # === Model 3: Comparison OLS model ===
    ols_model = sm.OLS(y_count.astype(float), X).fit()
    
    # === Marginal effects calculation (main indicators only) ===
    print("\n=== Marginal Effects Calculation ===")
    marginal_effects = {}
    
    for indicator in available_indicators:
        if indicator in X.columns and indicator in logit_model.params.index:
            # Marginal effects for logit model (marginal effects at mean)
            coef = logit_model.params[indicator]
            mean_prob = y_binary.mean()
            marginal_effect = coef * mean_prob * (1 - mean_prob)
            pvalue = logit_model.pvalues[indicator] if hasattr(logit_model, 'pvalues') else 0.5
            marginal_effects[indicator] = {
                'coefficient': coef,
                'marginal_effect_probability': marginal_effect,
                'pvalue': pvalue
            }
    
    # Create results dictionary
    results = {
        'data_info': {
            'sample_size': len(df_clean),
            'policy_activations': int(y_binary.sum()),
            'policy_activation_rate': float(y_binary.mean()),
            'indicators_used': available_indicators,
            'y_binary': y_binary  # Added for visualization
        },
        'logit_model': logit_model,
        'count_model': nb_model,
        'ols_model': ols_model,
        'marginal_effects': marginal_effects,
        'scaler': scaler
    }
    
    return results

def create_visualizations(results: Dict, df_original: pd.DataFrame, output_dir: str = "../results"):
    """
    Create and save analysis result visualizations
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Macroeconomic indicator coefficient plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Logit model coefficients
    logit_coeffs = []
    logit_indicators = []
    for indicator in results['data_info']['indicators_used']:
        if indicator in results['logit_model'].params.index:
            logit_coeffs.append(results['logit_model'].params[indicator])
            logit_indicators.append(indicator.replace('_', '\n'))
    
    ax1.barh(range(len(logit_indicators)), logit_coeffs)
    ax1.set_yticks(range(len(logit_indicators)))
    ax1.set_yticklabels(logit_indicators, fontsize=10)
    ax1.set_xlabel('Coefficient (Log-Odds)', fontsize=12)
    ax1.set_title('Policy Activation Probability\n(Logit Model Coefficients)', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Marginal effects plot
    marginal_effects = []
    me_indicators = []
    for indicator, me in results['marginal_effects'].items():
        marginal_effects.append(me['marginal_effect_probability'])
        me_indicators.append(indicator.replace('_', '\n'))
    
    colors = ['green' if x > 0 else 'red' for x in marginal_effects]
    ax2.barh(range(len(me_indicators)), marginal_effects, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(me_indicators)))
    ax2.set_yticklabels(me_indicators, fontsize=10)
    ax2.set_xlabel('Marginal Effect on Probability', fontsize=12)
    ax2.set_title('Marginal Effects on Policy\nActivation Probability', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/policy_coefficients_and_marginal_effects.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Policy activation time series trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Annual policy activation rate
    yearly_stats = df_original.groupby('year').agg({
        'obs_bin': ['count', 'sum', 'mean'],
        'obs_agg': 'mean'
    }).round(4)
    yearly_stats.columns = ['total_obs', 'activations', 'activation_rate', 'avg_intensity']
    
    ax1.plot(yearly_stats.index, yearly_stats['activation_rate'] * 100, marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Policy Activation Rate (%)', fontsize=12)
    ax1.set_title('Macroprudential Policy Activation Rate Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Major financial crisis annotations
    crisis_years = [1997, 2008, 2020]
    crisis_labels = ['Asian Crisis', '2008 Crisis', 'COVID-19']
    for year, label in zip(crisis_years, crisis_labels):
        if year in yearly_stats.index:
            ax1.axvline(x=year, color='red', linestyle='--', alpha=0.7)
            ax1.text(year, ax1.get_ylim()[1] * 0.8, label, rotation=90, 
                    verticalalignment='bottom', fontsize=9, color='red')
    
    # Policy activation intensity trends
    ax2.bar(yearly_stats.index, yearly_stats['avg_intensity'], alpha=0.7, color='steelblue')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Average Policy Intensity', fontsize=12)
    ax2.set_title('Average Policy Activation Intensity Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/policy_trends_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Macroeconomic indicator correlation heatmap
    indicators = results['data_info']['indicators_used']
    corr_data = df_original[indicators].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Macroeconomic Indicators', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/indicators_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Policy activation vs non-activation indicator distribution comparison
    n_indicators = len(indicators)
    n_cols = 3
    n_rows = (n_indicators + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, indicator in enumerate(indicators):
        if i < len(axes):
            activated = df_original[df_original['obs_bin'] == 1][indicator].dropna()
            not_activated = df_original[df_original['obs_bin'] == 0][indicator].dropna()
            
            axes[i].hist(not_activated, bins=30, alpha=0.7, label='No Policy (0)', color='lightblue', density=True)
            axes[i].hist(activated, bins=30, alpha=0.7, label='Policy Activated (1)', color='orange', density=True)
            axes[i].set_xlabel(indicator.replace('_', ' ').title(), fontsize=10)
            axes[i].set_ylabel('Density', fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(len(indicators), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribution of Macro Indicators: Policy vs No Policy', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/indicators_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Model prediction performance visualization
    if hasattr(results['logit_model'], 'fittedvalues'):
        predicted_probs = results['logit_model'].fittedvalues
        actual = results['data_info']['y_binary'] if 'y_binary' in results['data_info'] else None
        
        if actual is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Predicted probability distribution
            ax1.hist(predicted_probs[actual == 0], bins=50, alpha=0.7, label='No Policy', color='lightblue', density=True)
            ax1.hist(predicted_probs[actual == 1], bins=50, alpha=0.7, label='Policy Activated', color='orange', density=True)
            ax1.set_xlabel('Predicted Probability', fontsize=12)
            ax1.set_ylabel('Density', fontsize=12)
            ax1.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ROC-style scatter plot
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(actual, predicted_probs)
            roc_auc = auc(fpr, tpr)
            
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate', fontsize=12)
            ax2.set_ylabel('True Positive Rate', fontsize=12)
            ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_performance_evaluation.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nVisualization files saved to {output_dir}:")
    print("  - policy_coefficients_and_marginal_effects.png")
    print("  - policy_trends_over_time.png") 
    print("  - indicators_correlation_matrix.png")
    print("  - indicators_distribution_comparison.png")
    print("  - model_performance_evaluation.png")

def print_analysis_results(results: Dict):
    """
    Organize and output analysis results
    """
    print("\n" + "="*60)
    print("Macroprudential Policy Reaction Function Analysis Results")
    print("="*60)
    
    # Data overview
    info = results['data_info']
    print(f"\n【Data Overview】")
    print(f"Analysis sample size: {info['sample_size']:,}")
    print(f"Policy activation observations: {info['policy_activations']:,}")
    print(f"Policy activation rate: {info['policy_activation_rate']:.2%}")
    print(f"Number of indicators used: {len(info['indicators_used'])}")
    
    # Main results (logit model)
    print(f"\n【Impact on Policy Activation Probability (Logit Model)】")
    logit = results['logit_model']
    if hasattr(logit, 'prsquared'):
        print(f"Pseudo R-squared: {logit.prsquared:.4f}")
    if hasattr(logit, 'aic'):
        print(f"AIC: {logit.aic:.2f}")
    
    print("\nCoefficients and significance of main macroeconomic indicators:")
    for indicator in results['data_info']['indicators_used']:
        if indicator in logit.params.index:
            coef = logit.params[indicator]
            if hasattr(logit, 'pvalues'):
                pval = logit.pvalues[indicator]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"  {indicator}: {coef:8.4f} {sig:3s} (p={pval:.3f})")
            else:
                print(f"  {indicator}: {coef:8.4f} (regularized model)")
    
    # Marginal effects
    print(f"\n【Marginal Effects (Impact on Activation Probability)】")
    for indicator, me in results['marginal_effects'].items():
        effect = me['marginal_effect_probability']
        pval = me['pvalue']
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {indicator}: {effect:8.6f} {sig:3s}")
    
    # Policy intensity model
    print(f"\n【Impact on Policy Activation Intensity】")
    count_model = results['count_model']
    if hasattr(count_model, 'prsquared'):
        print(f"Pseudo R-squared: {count_model.prsquared:.4f}")
    else:
        print(f"R-squared: {count_model.rsquared:.4f}")
    
    print("\nCoefficients of main macroeconomic indicators:")
    for indicator in results['data_info']['indicators_used']:
        if indicator in count_model.params.index:
            coef = count_model.params[indicator]
            pval = count_model.pvalues[indicator]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {indicator}: {coef:8.4f} {sig:3s} (p={pval:.3f})")
    
    print("\nNote: ***, **, * indicate statistical significance at 1%, 5%, 10% levels respectively")
    print("    Coefficients are standardized values (effect of 1 standard deviation change)")

def main():
    """Main analysis execution"""
    print("Starting macroprudential policy reaction function analysis...")
    
    # Data loading
    df = pd.read_csv("../data/dataset/final.csv")
    
    # Execute analysis
    results = run_macroprudential_policy_analysis(df)
    
    # Output results
    print_analysis_results(results)
    
    # Create and save visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    create_visualizations(results, df)
    
    # Detailed results (if needed)
    print("\n" + "="*60)
    print("Detailed Statistical Results")
    print("="*60)
    print("\n【Logit Model Details】")
    if hasattr(results['logit_model'], 'summary2'):
        print(results['logit_model'].summary2())
    else:
        print("Regularized logit model coefficients:")
        print(results['logit_model'].params)
    
    return results

if __name__ == "__main__":
    results = main()