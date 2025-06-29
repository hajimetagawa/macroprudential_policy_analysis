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

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

def run_macroprudential_policy_analysis(df: pd.DataFrame) -> Dict:
    """
    マクロ経済指標がマクロプルーデンス政策実施に与える影響を分析する。
    
    分析内容：
    1. 政策発動確率モデル（ロジット）
    2. 政策発動強度モデル（負の二項分布・OLS）
    3. 限界効果の計算
    4. 予測精度の評価
    
    Parameters:
        df (pd.DataFrame): 分析用のDataFrame
        
    Returns:
        Dict: 分析結果辞書
    """
    
    # より包括的なマクロ経済指標を使用
    macro_indicators = [
        'credit_gap_actual',  # クレジットギャップ（金融不均衡の主要指標）
        'residential_property_price_yoy_changes_pct',  # 住宅価格上昇率
        'debt_service_ratio_private_non_financial_sector',  # 民間債務返済比率
        'central_bank_policy_rate_obs_value',  # 金融政策スタンス
        'total_credit_private_non_financial_sector_all_sector',  # 民間信用総額
        'effective_exchange_rate_real',  # 実質実効為替レート
        'commercial_property_price_obs_value'  # 商業用不動産価格
    ]
    
    cols_to_use = ['obs_bin', 'obs_agg', 'country_code', 'year'] + macro_indicators

    # データ前処理
    print("=== データ前処理 ===")
    
    # 利用可能な指標のみを選択
    available_indicators = [col for col in macro_indicators if col in df.columns]
    cols_to_use = ['obs_bin', 'obs_agg', 'country_code', 'year'] + available_indicators
    
    print(f"使用するマクロ経済指標: {available_indicators}")
    
    # 欠損値処理
    df_clean = df[cols_to_use].copy()
    
    # 数値型変換
    numeric_cols = ['obs_bin', 'obs_agg'] + available_indicators
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 欠損値除外前後のサンプル数
    print(f"処理前サンプル数: {len(df_clean)}")
    df_clean = df_clean.dropna()
    print(f"処理後サンプル数: {len(df_clean)}")
    print(f"政策発動観測数: {df_clean['obs_bin'].sum()}")
    
    if len(df_clean) == 0:
        raise ValueError("分析可能なデータがありません")
    
    # マクロ経済指標の標準化（解釈しやすくするため）
    scaler = StandardScaler()
    df_clean[available_indicators] = scaler.fit_transform(df_clean[available_indicators])
    
    # 多重共線性対策：高い相関を持つ変数を除去
    corr_matrix = df_clean[available_indicators].corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print(f"高い相関を持つ変数ペア: {high_corr_pairs}")
        # 相関の高い変数ペアの後者を除去
        vars_to_remove = set()
        for pair in high_corr_pairs:
            vars_to_remove.add(pair[1])
        available_indicators = [var for var in available_indicators if var not in vars_to_remove]
        print(f"除去後の指標: {available_indicators}")
    
    # 固定効果（年のみ - 国固定効果は標本サイズが少ない場合は省略）
    df_clean = pd.get_dummies(df_clean, columns=['year'], prefix='year', drop_first=True)
    
    # 説明変数準備（マクロ指標のみ + 年固定効果）
    macro_cols = available_indicators
    year_cols = [col for col in df_clean.columns if col.startswith('year_')]
    X_cols = macro_cols + year_cols
    
    # 完全分離チェック（政策発動が0または1のみの変数を除去）
    X_data = df_clean[X_cols]
    policy_activated = df_clean['obs_bin'] == 1
    policy_not_activated = df_clean['obs_bin'] == 0
    
    vars_to_keep = []
    for col in X_cols:
        if col.startswith('year_'):
            # 年ダミーは基本的に保持（ただし変動があるもののみ）
            if X_data[col].nunique() > 1:
                vars_to_keep.append(col)
        else:
            # マクロ指標：完全分離をチェック
            activated_values = X_data.loc[policy_activated, col]
            not_activated_values = X_data.loc[policy_not_activated, col]
            if len(activated_values) > 0 and len(not_activated_values) > 0:
                # 重複する値域があるかチェック
                if (activated_values.min() <= not_activated_values.max() and 
                    not_activated_values.min() <= activated_values.max()):
                    vars_to_keep.append(col)
    
    X_cols = vars_to_keep
    print(f"最終的な説明変数数: {len(X_cols)}")
    
    if len(X_cols) == 0:
        raise ValueError("使用可能な説明変数がありません")
    
    X = sm.add_constant(df_clean[X_cols].astype(float))
    y_binary = df_clean['obs_bin'].astype(int)
    y_count = df_clean['obs_agg'].astype(int)
    
    # 分散膨張因子（VIF）チェック
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        high_vif = vif_data[vif_data["VIF"] > 10]
        if len(high_vif) > 0:
            print(f"高いVIF値を持つ変数: \\n{high_vif}")
    except:
        pass
    
    # === モデル1: 政策発動確率（ロジットモデル） ===
    print("\n=== モデル1: 政策発動確率分析 ===")
    try:
        logit_model = sm.Logit(y_binary, X).fit(disp=False, maxiter=1000)
    except:
        print("ロジットモデルが収束しませんでした。正則化ロジットを試します。")
        # 正則化ロジット（L1正則化）
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        
        lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        lr.fit(X.iloc[:, 1:], y_binary)  # 定数項を除く
        
        # statsmodelsライクなオブジェクトを作成
        class SimpleLogitResult:
            def __init__(self, lr_model, X, y):
                self.params = pd.Series(lr_model.coef_[0], index=X.columns[1:])
                self.params['const'] = lr_model.intercept_[0]
                self.params = self.params[X.columns]  # 順序を合わせる
                
                # 予測値計算
                y_pred_proba = lr_model.predict_proba(X.iloc[:, 1:])[:, 1]
                self.fittedvalues = y_pred_proba
                
                # 簡易的な統計量
                from sklearn.metrics import log_loss
                self.llf = -log_loss(y, y_pred_proba) * len(y)
                self.pvalues = pd.Series([0.5] * len(self.params), index=self.params.index)  # ダミー値
                
        logit_model = SimpleLogitResult(lr, X, y_binary)
    
    # === モデル2: 政策発動強度（負の二項分布モデル） ===
    print("\n=== モデル2: 政策発動強度分析 ===")
    try:
        nb_model = sm.NegativeBinomial(y_count, X).fit(disp=False)
    except:
        print("負の二項分布モデルが収束しませんでした。OLSモデルを使用します。")
        nb_model = sm.OLS(y_count.astype(float), X).fit()
    
    # === モデル3: 比較用OLSモデル ===
    ols_model = sm.OLS(y_count.astype(float), X).fit()
    
    # === 限界効果計算（主要指標のみ） ===
    print("\n=== 限界効果計算 ===")
    marginal_effects = {}
    
    for indicator in available_indicators:
        if indicator in X.columns and indicator in logit_model.params.index:
            # ロジットモデルの限界効果（平均での限界効果）
            coef = logit_model.params[indicator]
            mean_prob = y_binary.mean()
            marginal_effect = coef * mean_prob * (1 - mean_prob)
            pvalue = logit_model.pvalues[indicator] if hasattr(logit_model, 'pvalues') else 0.5
            marginal_effects[indicator] = {
                'coefficient': coef,
                'marginal_effect_probability': marginal_effect,
                'pvalue': pvalue
            }
    
    # 結果辞書作成
    results = {
        'data_info': {
            'sample_size': len(df_clean),
            'policy_activations': int(y_binary.sum()),
            'policy_activation_rate': float(y_binary.mean()),
            'indicators_used': available_indicators,
            'y_binary': y_binary  # 可視化用に追加
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
    分析結果の可視化を作成し、画像として保存する
    """
    # 出力ディレクトリ作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. マクロ経済指標の係数プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ロジットモデルの係数
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
    
    # 限界効果プロット
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
    
    # 2. 政策発動の時系列トレンド
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 年別政策発動率
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
    
    # 主要金融危機の注釈
    crisis_years = [1997, 2008, 2020]
    crisis_labels = ['Asian Crisis', '2008 Crisis', 'COVID-19']
    for year, label in zip(crisis_years, crisis_labels):
        if year in yearly_stats.index:
            ax1.axvline(x=year, color='red', linestyle='--', alpha=0.7)
            ax1.text(year, ax1.get_ylim()[1] * 0.8, label, rotation=90, 
                    verticalalignment='bottom', fontsize=9, color='red')
    
    # 政策発動強度の推移
    ax2.bar(yearly_stats.index, yearly_stats['avg_intensity'], alpha=0.7, color='steelblue')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Average Policy Intensity', fontsize=12)
    ax2.set_title('Average Policy Activation Intensity Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/policy_trends_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. マクロ経済指標の相関ヒートマップ
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
    
    # 4. 政策発動vs非発動での指標分布比較
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
    
    # 空いているサブプロットを非表示
    for j in range(len(indicators), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribution of Macro Indicators: Policy vs No Policy', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/indicators_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. モデル予測性能の可視化
    if hasattr(results['logit_model'], 'fittedvalues'):
        predicted_probs = results['logit_model'].fittedvalues
        actual = results['data_info']['y_binary'] if 'y_binary' in results['data_info'] else None
        
        if actual is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 予測確率分布
            ax1.hist(predicted_probs[actual == 0], bins=50, alpha=0.7, label='No Policy', color='lightblue', density=True)
            ax1.hist(predicted_probs[actual == 1], bins=50, alpha=0.7, label='Policy Activated', color='orange', density=True)
            ax1.set_xlabel('Predicted Probability', fontsize=12)
            ax1.set_ylabel('Density', fontsize=12)
            ax1.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ROC様の散布図
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
    
    print(f"\n可視化ファイルが {output_dir} に保存されました:")
    print("  - policy_coefficients_and_marginal_effects.png")
    print("  - policy_trends_over_time.png") 
    print("  - indicators_correlation_matrix.png")
    print("  - indicators_distribution_comparison.png")
    print("  - model_performance_evaluation.png")

def print_analysis_results(results: Dict):
    """
    分析結果を整理して出力する
    """
    print("\n" + "="*60)
    print("マクロプルーデンス政策反応関数分析結果")
    print("="*60)
    
    # データ概要
    info = results['data_info']
    print(f"\n【データ概要】")
    print(f"分析サンプル数: {info['sample_size']:,}")
    print(f"政策発動観測数: {info['policy_activations']:,}")
    print(f"政策発動率: {info['policy_activation_rate']:.2%}")
    print(f"使用指標数: {len(info['indicators_used'])}")
    
    # 主要結果（ロジットモデル）
    print(f"\n【政策発動確率への影響（ロジットモデル）】")
    logit = results['logit_model']
    if hasattr(logit, 'prsquared'):
        print(f"疑似決定係数: {logit.prsquared:.4f}")
    if hasattr(logit, 'aic'):
        print(f"AIC: {logit.aic:.2f}")
    
    print("\n主要マクロ経済指標の係数と有意性:")
    for indicator in results['data_info']['indicators_used']:
        if indicator in logit.params.index:
            coef = logit.params[indicator]
            if hasattr(logit, 'pvalues'):
                pval = logit.pvalues[indicator]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"  {indicator}: {coef:8.4f} {sig:3s} (p={pval:.3f})")
            else:
                print(f"  {indicator}: {coef:8.4f} (正則化モデル)")
    
    # 限界効果
    print(f"\n【限界効果（発動確率への影響）】")
    for indicator, me in results['marginal_effects'].items():
        effect = me['marginal_effect_probability']
        pval = me['pvalue']
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {indicator}: {effect:8.6f} {sig:3s}")
    
    # 政策強度モデル
    print(f"\n【政策発動強度への影響】")
    count_model = results['count_model']
    if hasattr(count_model, 'prsquared'):
        print(f"疑似決定係数: {count_model.prsquared:.4f}")
    else:
        print(f"決定係数: {count_model.rsquared:.4f}")
    
    print("\n主要マクロ経済指標の係数:")
    for indicator in results['data_info']['indicators_used']:
        if indicator in count_model.params.index:
            coef = count_model.params[indicator]
            pval = count_model.pvalues[indicator]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {indicator}: {coef:8.4f} {sig:3s} (p={pval:.3f})")
    
    print("\n注: ***, **, *はそれぞれ1%, 5%, 10%水準で統計的有意")
    print("    係数は標準化後の値（1標準偏差変化の効果）")

def main():
    """メイン分析実行"""
    print("マクロプルーデンシャル政策反応関数分析を開始します...")
    
    # データ読み込み
    df = pd.read_csv("../data/dataset/final.csv")
    
    # 分析実行
    results = run_macroprudential_policy_analysis(df)
    
    # 結果出力
    print_analysis_results(results)
    
    # 可視化作成・保存
    print("\n" + "="*60)
    print("可視化作成中...")
    print("="*60)
    create_visualizations(results, df)
    
    # 詳細結果（必要に応じて）
    print("\n" + "="*60)
    print("詳細統計結果")
    print("="*60)
    print("\n【ロジットモデル詳細】")
    if hasattr(results['logit_model'], 'summary2'):
        print(results['logit_model'].summary2())
    else:
        print("正則化ロジットモデルの係数:")
        print(results['logit_model'].params)
    
    return results

if __name__ == "__main__":
    results = main()
