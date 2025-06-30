"""
Analysis Engine
Comprehensive macroprudential policy reaction function analysis
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Plot style configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

class AnalysisEngine:
    """Engine for comprehensive macroprudential policy analysis"""
    
    def __init__(self):
        """Initialize analysis engine"""
        self.macro_indicators = [
            'CRG',          # credit_gap_trend_hp_filter
            'CRNF',         # total_credit_private_non_financial_sector_all_sector
            'DSR',          # debt_service_ratio_private_non_financial_sector
            'RPP',          # residential_property_price_yoy_changes_pct
            'CPP',          # commercial_property_price_average_index
            'REER',         # effective_exchange_rate_real
            'CBPOL'         # central_bank_policy_rate_quarterly_max
        ]
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """Preprocess data for analysis"""
        print("=== Data Preprocessing ===")
        
        # Select available indicators
        available_indicators = [col for col in self.macro_indicators if col in df.columns]
        cols_to_use = ['policy_binary', 'policy_intensity', 'country_code', 'period'] + available_indicators
        
        print(f"Using macroeconomic indicators: {available_indicators}")
        
        # Handle missing values
        df_clean = df[cols_to_use].copy()
        
        # Convert to numeric
        numeric_cols = ['policy_binary', 'policy_intensity'] + available_indicators
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Drop missing values
        print(f"Sample size before processing: {len(df_clean)}")
        df_clean = df_clean.dropna()
        print(f"Sample size after processing: {len(df_clean)}")
        print(f"Policy activation observations: {df_clean['policy_binary'].sum()}")
        
        if len(df_clean) == 0:
            raise ValueError("No analyzable data available")
        
        # Standardize indicators
        scaler = StandardScaler()
        df_clean[available_indicators] = scaler.fit_transform(df_clean[available_indicators])
        
        # Remove highly correlated variables
        available_indicators = self._remove_multicollinearity(df_clean, available_indicators)
        
        return df_clean, available_indicators
    
    def _remove_multicollinearity(self, df: pd.DataFrame, indicators: list) -> list:
        """Remove variables with high correlation"""
        corr_matrix = df[indicators].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print(f"High correlation variable pairs: {high_corr_pairs}")
            vars_to_remove = set()
            for pair in high_corr_pairs:
                vars_to_remove.add(pair[1])
            indicators = [var for var in indicators if var not in vars_to_remove]
            print(f"Indicators after removal: {indicators}")
        
        return indicators
    
    def _prepare_model_data(self, df_clean: pd.DataFrame, available_indicators: list) -> Dict:
        """Prepare data for modeling"""
        # Add year fixed effects
        df_clean = pd.get_dummies(df_clean, columns=['period'], prefix='year', drop_first=True)
        
        # Prepare explanatory variables
        macro_cols = available_indicators
        year_cols = [col for col in df_clean.columns if col.startswith('year_')]
        X_cols = macro_cols + year_cols
        
        # Remove variables causing perfect separation
        X_data = df_clean[X_cols]
        policy_activated = df_clean['policy_binary'] == 1
        policy_not_activated = df_clean['policy_binary'] == 0
        
        vars_to_keep = []
        for col in X_cols:
            if col.startswith('year_'):
                if X_data[col].nunique() > 1:
                    vars_to_keep.append(col)
            else:
                activated_values = X_data.loc[policy_activated, col]
                not_activated_values = X_data.loc[policy_not_activated, col]
                if len(activated_values) > 0 and len(not_activated_values) > 0:
                    if (activated_values.min() <= not_activated_values.max() and 
                        not_activated_values.min() <= activated_values.max()):
                        vars_to_keep.append(col)
        
        X_cols = vars_to_keep
        print(f"Final number of explanatory variables: {len(X_cols)}")
        
        if len(X_cols) == 0:
            raise ValueError("No usable explanatory variables available")
        
        X = sm.add_constant(df_clean[X_cols].astype(float))
        y_binary = df_clean['policy_binary'].astype(int)
        y_count = df_clean['policy_intensity'].astype(int)
        
        return {
            'X': X,
            'y_binary': y_binary,
            'y_count': y_count,
            'df_clean': df_clean,
            'available_indicators': available_indicators
        }
    
    def _fit_logit_model(self, X: pd.DataFrame, y_binary: pd.Series):
        """Fit logit model for policy activation probability"""
        print("\\n=== Model 1: Policy Activation Probability Analysis ===")
        try:
            logit_model = sm.Logit(y_binary, X).fit(disp=False, maxiter=1000)
        except:
            print("Logit model did not converge. Trying regularized logit.")
            # Use regularized logit
            lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            lr.fit(X.iloc[:, 1:], y_binary)
            
            # Create statsmodels-like object
            class SimpleLogitResult:
                def __init__(self, lr_model, X, y):
                    self.params = pd.Series(lr_model.coef_[0], index=X.columns[1:])
                    self.params['const'] = lr_model.intercept_[0]
                    self.params = self.params[X.columns]
                    
                    y_pred_proba = lr_model.predict_proba(X.iloc[:, 1:])[:, 1]
                    self.fittedvalues = y_pred_proba
                    
                    self.llf = -log_loss(y, y_pred_proba) * len(y)
                    self.pvalues = pd.Series([0.5] * len(self.params), index=self.params.index)
                    
            logit_model = SimpleLogitResult(lr, X, y_binary)
        
        return logit_model
    
    def _fit_count_model(self, X: pd.DataFrame, y_count: pd.Series):
        """Fit count model for policy activation intensity"""
        print("\\n=== Model 2: Policy Activation Intensity Analysis ===")
        try:
            nb_model = sm.NegativeBinomial(y_count, X).fit(disp=False)
        except:
            print("Negative binomial model did not converge. Using OLS model.")
            nb_model = sm.OLS(y_count.astype(float), X).fit()
        
        return nb_model
    
    def _calculate_marginal_effects(self, logit_model, available_indicators: list, y_binary: pd.Series) -> Dict:
        """Calculate marginal effects for main indicators"""
        print("\\n=== Marginal Effects Calculation ===")
        marginal_effects = {}
        
        for indicator in available_indicators:
            if indicator in logit_model.params.index:
                coef = logit_model.params[indicator]
                mean_prob = y_binary.mean()
                marginal_effect = coef * mean_prob * (1 - mean_prob)
                pvalue = logit_model.pvalues[indicator] if hasattr(logit_model, 'pvalues') else 0.5
                marginal_effects[indicator] = {
                    'coefficient': coef,
                    'marginal_effect_probability': marginal_effect,
                    'pvalue': pvalue
                }
        
        return marginal_effects
    
    def run_comprehensive_analysis(self, df: pd.DataFrame) -> Dict:
        """Run comprehensive macroprudential policy analysis"""
        # Data preprocessing
        df_clean, available_indicators = self._preprocess_data(df)
        
        # Prepare model data
        model_data = self._prepare_model_data(df_clean, available_indicators)
        
        # Fit models
        logit_model = self._fit_logit_model(model_data['X'], model_data['y_binary'])
        count_model = self._fit_count_model(model_data['X'], model_data['y_count'])
        ols_model = sm.OLS(model_data['y_count'].astype(float), model_data['X']).fit()
        
        # Calculate marginal effects
        marginal_effects = self._calculate_marginal_effects(logit_model, available_indicators, model_data['y_binary'])
        
        # Compile results
        results = {
            'data_info': {
                'sample_size': len(df_clean),
                'policy_activations': int(model_data['y_binary'].sum()),
                'policy_activation_rate': float(model_data['y_binary'].mean()),
                'indicators_used': available_indicators,
                'y_binary': model_data['y_binary']
            },
            'logit_model': logit_model,
            'count_model': count_model,
            'ols_model': ols_model,
            'marginal_effects': marginal_effects
        }
        
        return results
    
    def create_visualizations(self, results: Dict, df_original: pd.DataFrame, output_dir: str = "../results"):
        """Create and save analysis visualizations"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Coefficients and marginal effects
        self._plot_coefficients_and_effects(results, output_dir)
        
        # 2. Policy trends over time
        self._plot_policy_trends(df_original, output_dir)
        
        # 3. Indicator correlations
        self._plot_indicator_correlations(results, df_original, output_dir)
        
        # 4. Distribution comparisons
        self._plot_distribution_comparisons(results, df_original, output_dir)
        
        # 5. Model performance
        self._plot_model_performance(results, output_dir)
        
        print(f"\\nVisualization files saved to {output_dir}:")
        print("  - policy_coefficients_and_marginal_effects.png")
        print("  - policy_trends_over_time.png") 
        print("  - indicators_correlation_matrix.png")
        print("  - indicators_distribution_comparison.png")
        print("  - model_performance_evaluation.png")
    
    def _plot_coefficients_and_effects(self, results: Dict, output_dir: str):
        """Plot coefficients and marginal effects"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Logit coefficients
        logit_coeffs = []
        logit_indicators = []
        for indicator in results['data_info']['indicators_used']:
            if indicator in results['logit_model'].params.index:
                logit_coeffs.append(results['logit_model'].params[indicator])
                logit_indicators.append(indicator.replace('_', '\\n'))
        
        ax1.barh(range(len(logit_indicators)), logit_coeffs)
        ax1.set_yticks(range(len(logit_indicators)))
        ax1.set_yticklabels(logit_indicators, fontsize=10)
        ax1.set_xlabel('Coefficient (Log-Odds)', fontsize=12)
        ax1.set_title('Policy Activation Probability\\n(Logit Model Coefficients)', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Marginal effects
        marginal_effects = []
        me_indicators = []
        for indicator, me in results['marginal_effects'].items():
            marginal_effects.append(me['marginal_effect_probability'])
            me_indicators.append(indicator.replace('_', '\\n'))
        
        colors = ['green' if x > 0 else 'red' for x in marginal_effects]
        ax2.barh(range(len(me_indicators)), marginal_effects, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(me_indicators)))
        ax2.set_yticklabels(me_indicators, fontsize=10)
        ax2.set_xlabel('Marginal Effect on Probability', fontsize=12)
        ax2.set_title('Marginal Effects on Policy\\nActivation Probability', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/policy_coefficients_and_marginal_effects.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_policy_trends(self, df_original: pd.DataFrame, output_dir: str):
        """Plot policy trends over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Annual policy activation rate
        yearly_stats = df_original.groupby('period').agg({
            'policy_binary': ['count', 'sum', 'mean'],
            'policy_intensity': 'mean'
        }).round(4)
        yearly_stats.columns = ['total_obs', 'activations', 'activation_rate', 'avg_intensity']
        
        ax1.plot(yearly_stats.index, yearly_stats['activation_rate'] * 100, marker='o', linewidth=2, markersize=4)
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Policy Activation Rate (%)', fontsize=12)
        ax1.set_title('Macroprudential Policy Activation Rate Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Crisis annotations
        crisis_years = [1997, 2008, 2020]
        crisis_labels = ['Asian Crisis', '2008 Crisis', 'COVID-19']
        for year, label in zip(crisis_years, crisis_labels):
            if year in yearly_stats.index:
                ax1.axvline(x=year, color='red', linestyle='--', alpha=0.7)
                ax1.text(year, ax1.get_ylim()[1] * 0.8, label, rotation=90, 
                        verticalalignment='bottom', fontsize=9, color='red')
        
        # Policy intensity trends
        ax2.bar(yearly_stats.index, yearly_stats['avg_intensity'], alpha=0.7, color='steelblue')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Average Policy Intensity', fontsize=12)
        ax2.set_title('Average Policy Activation Intensity Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/policy_trends_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_indicator_correlations(self, results: Dict, df_original: pd.DataFrame, output_dir: str):
        """Plot indicator correlation heatmap"""
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
    
    def _plot_distribution_comparisons(self, results: Dict, df_original: pd.DataFrame, output_dir: str):
        """Plot indicator distribution comparisons"""
        indicators = results['data_info']['indicators_used']
        n_indicators = len(indicators)
        n_cols = 3
        n_rows = (n_indicators + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, indicator in enumerate(indicators):
            if i < len(axes):
                activated = df_original[df_original['policy_binary'] == 1][indicator].dropna()
                not_activated = df_original[df_original['policy_binary'] == 0][indicator].dropna()
                
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
    
    def _plot_model_performance(self, results: Dict, output_dir: str):
        """Plot model performance evaluation"""
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
                
                # ROC curve
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
    
    def print_analysis_summary(self, results: Dict):
        """Print comprehensive analysis summary"""
        print("\\n" + "="*60)
        print("Macroprudential Policy Reaction Function Analysis Results")
        print("="*60)
        
        # Data overview
        info = results['data_info']
        print(f"\\n【Data Overview】")
        print(f"Analysis sample size: {info['sample_size']:,}")
        print(f"Policy activation observations: {info['policy_activations']:,}")
        print(f"Policy activation rate: {info['policy_activation_rate']:.2%}")
        print(f"Number of indicators used: {len(info['indicators_used'])}")
        
        # Main results (logit model)
        print(f"\\n【Impact on Policy Activation Probability (Logit Model)】")
        logit = results['logit_model']
        if hasattr(logit, 'prsquared'):
            print(f"Pseudo R-squared: {logit.prsquared:.4f}")
        if hasattr(logit, 'aic'):
            print(f"AIC: {logit.aic:.2f}")
        
        print("\\nCoefficients and significance of main macroeconomic indicators:")
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
        print(f"\\n【Marginal Effects (Impact on Activation Probability)】")
        for indicator, me in results['marginal_effects'].items():
            effect = me['marginal_effect_probability']
            pval = me['pvalue']
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {indicator}: {effect:8.6f} {sig:3s}")
        
        # Policy intensity model
        print(f"\\n【Impact on Policy Activation Intensity】")
        count_model = results['count_model']
        if hasattr(count_model, 'prsquared'):
            print(f"Pseudo R-squared: {count_model.prsquared:.4f}")
        else:
            print(f"R-squared: {count_model.rsquared:.4f}")
        
        print("\\nCoefficients of main macroeconomic indicators:")
        for indicator in results['data_info']['indicators_used']:
            if indicator in count_model.params.index:
                coef = count_model.params[indicator]
                pval = count_model.pvalues[indicator]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"  {indicator}: {coef:8.4f} {sig:3s} (p={pval:.3f})")
        
        print("\\nNote: ***, **, * indicate statistical significance at 1%, 5%, 10% levels respectively")
        print("    Coefficients are standardized values (effect of 1 standard deviation change)")