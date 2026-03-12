"""
斜視診断解析パイプライン
筑波大学附属病院 眼科

論文に報告された解析の再現可能なPython実装。

注記:
  - 近見・遠見の単独ROC解析は、元々JavaScriptで固定シードなしに実行された。
  - 元々のLGBM解析はStrabismusMLPipelineクラス（Python）で実行された。
  - 本スクリプトは統一的かつ再現可能なPython実装を提供する。
  - 交差検証の分割実装の差異により、論文報告値との軽微な数値差が生じうる。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
import lightgbm as lgb


# =============================================================================
# 定数定義
# =============================================================================

RANDOM_STATE = 2024
N_SPLITS = 5
CSV_PATH = "Strabismus_tknmAndIto_enhanced.csv"

# CSVから読み込む特徴量カラム名
FEATURE_NEAR = "水平斜視角 近見"
FEATURE_DISTANT = "水平斜視角 遠見"
LABEL_COLUMN = "斜視か"


# =============================================================================
# 評価指標の計算ユーティリティ
# =============================================================================

def _safe_divide(numerator, denominator):
    """ゼロ除算を防ぐ安全な割り算。分母が0の場合は0を返す。"""
    return numerator / denominator if denominator > 0 else 0


def compute_classification_metrics(y_true, y_pred, y_score):
    """
    二値分類の主要評価指標を一括計算する。

    混同行列から感度・特異度・正確度・F1スコアを算出し、
    予測確率からAUCを算出する。

    Args:
        y_true:  正解ラベル（0 or 1）
        y_pred:  予測ラベル（0 or 1）
        y_score: 予測確率（陽性クラスの確率）

    Returns:
        各指標を格納した辞書
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy    = (tp + tn) / len(y_true)
    precision   = _safe_divide(tp, tp + fp)
    f1_score    = _safe_divide(2 * precision * sensitivity,
                               precision + sensitivity)

    fpr_array, tpr_array, _ = roc_curve(y_true, y_score)
    auc_value = auc(fpr_array, tpr_array)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy":    accuracy,
        "f1":          f1_score,
        "auc":         auc_value,
    }


# =============================================================================
# データ読み込み
# =============================================================================

def load_data(csv_path):
    """
    CSVファイルから斜視データを読み込み、欠損値を除外する。

    Returns:
        X: 特徴量行列（近見・遠見の水平斜視角）[n_samples, 2]
        y: ラベル配列（0=正常, 1=斜視）[n_samples]
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    feature_columns = [FEATURE_NEAR, FEATURE_DISTANT]
    X = df[feature_columns].values
    y = df[LABEL_COLUMN].values

    # NaN行・ラベルが0/1以外の行を除外
    is_valid = ~np.isnan(X).any(axis=1) & np.isin(y, [0.0, 1.0])
    X = X[is_valid]
    y = y[is_valid].astype(int)

    n_strabismus = y.sum()
    n_normal = len(y) - n_strabismus
    print(f"データ読込完了: {len(y)}件 (斜視: {n_strabismus}, 正常: {n_normal})")

    return X, y


# =============================================================================
# ROC閾値解析（Youden Index法）
# =============================================================================

def find_optimal_threshold_by_youden(y_true, scores, step=0.1):
    """
    Youden Indexを最大化する最適閾値を探索する。

    0からスコア最大値まで step 刻みで閾値を走査し、
    感度＋特異度−1（Youden Index）が最大となる閾値を返す。

    Args:
        y_true: 正解ラベル
        scores: 斜視角の絶対値（閾値判定に使うスコア）
        step:   閾値の刻み幅（プリズムジオプトリー単位）

    Returns:
        最適閾値・感度・特異度を格納した辞書
    """
    thresholds = np.arange(0, np.max(scores) + step, step)

    best_youden = -1
    best_result = {"threshold": 0, "sensitivity": 0, "specificity": 0}

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        tp = np.sum((predictions == 1) & (y_true == 1))
        fn = np.sum((predictions == 0) & (y_true == 1))
        tn = np.sum((predictions == 0) & (y_true == 0))
        fp = np.sum((predictions == 1) & (y_true == 0))

        sensitivity = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        youden_index = sensitivity + specificity - 1

        if youden_index > best_youden:
            best_youden = youden_index
            best_result = {
                "threshold":   threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
            }

    return best_result


def evaluate_single_threshold(y_true, scores, threshold):
    """
    指定した閾値でスコアを二値分類し、各評価指標を算出する。

    Args:
        y_true:    正解ラベル
        scores:    斜視角の絶対値
        threshold: 判定閾値（プリズムジオプトリー）

    Returns:
        閾値と各評価指標を格納した辞書
    """
    predictions = (scores >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, predictions, scores)
    metrics["threshold"] = threshold
    return metrics


# =============================================================================
# 単一測定値による交差検証
# =============================================================================

def run_single_measurement_cv(y, scores, n_splits=N_SPLITS,
                              random_state=RANDOM_STATE):
    """
    単一測定値（近見 or 遠見）に対するK分割交差検証を実行する。

    各foldで訓練データからYouden最適閾値を求め、
    テストデータで評価する。

    Args:
        y:      ラベル配列
        scores: 単一測定値の配列
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for train_indices, test_indices in kfold.split(scores):
        # 訓練データで最適閾値を決定
        optimal = find_optimal_threshold_by_youden(
            y[train_indices], scores[train_indices]
        )
        # テストデータで評価
        result = evaluate_single_threshold(
            y[test_indices], scores[test_indices], optimal["threshold"]
        )
        fold_results.append(result)

    return fold_results


# =============================================================================
# 機械学習モデル（ロジスティック回帰 / LightGBM）
# =============================================================================

def _train_logistic_regression(X_train, y_train, X_test):
    """ロジスティック回帰モデルを訓練し、予測ラベルと確率を返す。"""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return predictions, probabilities


def _train_lightgbm(X_train, y_train, X_test, y_test, random_state):
    """
    LightGBMモデルを訓練し、予測確率を返す。

    早期停止（10ラウンド改善なし）を使用する。
    予測ラベルはYouden Index最適閾値で事後的に決定する。
    """
    params = {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "num_leaves":       31,
        "learning_rate":    0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          -1,
        "random_state":     random_state,
    }

    train_dataset = lgb.Dataset(X_train, label=y_train)
    valid_dataset = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=100,
        valid_sets=[valid_dataset],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
    )

    probabilities = model.predict(X_test, num_iteration=model.best_iteration)
    return probabilities


def _determine_predictions_by_youden(y_true, probabilities):
    """
    ROC曲線上のYouden Indexが最大となる閾値で予測ラベルを決定する。

    LightGBMのように predict() が確率のみを返すモデル用。
    """
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    best_index = np.argmax(tpr - fpr)
    return (probabilities >= thresholds[best_index]).astype(int)


def run_ml_cross_validation(X, y, n_splits=N_SPLITS,
                            random_state=RANDOM_STATE):
    """
    近見＋遠見の2特徴量を用いた機械学習モデルのK分割交差検証。

    ロジスティック回帰とLightGBMの2モデルを同一foldで訓練・評価する。

    Returns:
        モデル名をキー、各foldの指標リストを値とするネスト辞書
        例: {"LR": {"sensitivity": [...], ...}, "LGBM": {...}}
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metric_names = ["sensitivity", "specificity", "accuracy", "f1", "auc"]

    results = {
        model_name: {metric: [] for metric in metric_names}
        for model_name in ["LR", "LGBM"]
    }

    for train_indices, test_indices in kfold.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # --- ロジスティック回帰 ---
        lr_predictions, lr_probabilities = _train_logistic_regression(
            X_train, y_train, X_test
        )

        # --- LightGBM ---
        lgbm_probabilities = _train_lightgbm(
            X_train, y_train, X_test, y_test, random_state
        )
        # LGBMは確率のみなので、Youden最適閾値で予測ラベルを決定
        lgbm_predictions = _determine_predictions_by_youden(
            y_test, lgbm_probabilities
        )

        # --- 両モデルの評価を統一的に実施 ---
        models = [
            ("LR",   lr_predictions,   lr_probabilities),
            ("LGBM", lgbm_predictions, lgbm_probabilities),
        ]
        for model_name, predictions, probabilities in models:
            metrics = compute_classification_metrics(
                y_test, predictions, probabilities
            )
            for metric in metric_names:
                results[model_name][metric].append(metrics[metric])

    return results


# =============================================================================
# ブートストラップ安定性解析
# =============================================================================

def run_bootstrap_threshold_analysis(y, scores, n_iterations=1000,
                                     random_state=RANDOM_STATE):
    """
    ブートストラップ法で最適閾値の分布と信頼区間を推定する。

    復元抽出したサンプルごとにYouden最適閾値を求め、
    その分布から平均・中央値・標準偏差・95%信頼区間を算出する。

    Args:
        y:            ラベル配列
        scores:       斜視角の絶対値
        n_iterations: ブートストラップ反復回数
    """
    rng = np.random.RandomState(random_state)
    threshold_samples = []

    for _ in range(n_iterations):
        bootstrap_indices = rng.choice(len(y), size=len(y), replace=True)
        y_bootstrap = y[bootstrap_indices]
        scores_bootstrap = scores[bootstrap_indices]

        # 陽性・陰性の両クラスが含まれない場合はスキップ
        if len(np.unique(y_bootstrap)) < 2:
            continue

        result = find_optimal_threshold_by_youden(y_bootstrap, scores_bootstrap)
        threshold_samples.append(result["threshold"])

    threshold_samples = np.array(threshold_samples)

    return {
        "n":            len(threshold_samples),
        "mean":         np.mean(threshold_samples),
        "median":       np.median(threshold_samples),
        "sd":           np.std(threshold_samples),
        "ci_lower":     np.percentile(threshold_samples, 2.5),
        "ci_upper":     np.percentile(threshold_samples, 97.5),
        "distribution": threshold_samples,
    }


# =============================================================================
# PPV / NPV 推定（ベイズの定理）
# =============================================================================

def compute_ppv_npv(sensitivity, specificity, prevalence):
    """
    ベイズの定理に基づき、陽性的中率(PPV)と陰性的中率(NPV)を計算する。

    スクリーニング検査の臨床的有用性を有病率を考慮して評価するために使用する。

    Args:
        sensitivity: 感度（真陽性率）
        specificity: 特異度（真陰性率）
        prevalence:  有病率（0〜1）
    """
    ppv = (sensitivity * prevalence) / (
        sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    )
    npv = (specificity * (1 - prevalence)) / (
        specificity * (1 - prevalence) + (1 - sensitivity) * prevalence
    )
    return ppv, npv


# =============================================================================
# 結果表示ユーティリティ
# =============================================================================

def _format_separator(title):
    """セクション区切り線とタイトルを整形して返す。"""
    separator = "=" * 60
    return f"\n{separator}\n {title}\n{separator}"


def print_cv_summary(results, label):
    """
    交差検証結果の平均±標準偏差を表示する。

    Args:
        results: fold結果のリスト（単一測定）または指標辞書（ML）
        label:   表示用ラベル
    """
    print(_format_separator(label))

    # 単一測定のfold結果（辞書のリスト）とML結果（リストの辞書）で処理を分岐
    if isinstance(results, list):
        metric_names = ["threshold", "sensitivity", "specificity",
                        "accuracy", "f1", "auc"]
        for metric in metric_names:
            values = [fold[metric] for fold in results]
            mean, std = np.mean(values), np.std(values)
            print(f"  {metric:<14s}: {mean:.3f} +/- {std:.3f}")
    else:
        metric_names = ["sensitivity", "specificity", "accuracy", "f1", "auc"]
        for metric in metric_names:
            values = results[metric]
            mean, std = np.mean(values), np.std(values)
            print(f"  {metric:<14s}: {mean:.3f} +/- {std:.3f}")


def print_bootstrap_summary(summary, label):
    """ブートストラップ解析結果（統計量と閾値分布）を表示する。"""
    print(_format_separator(f"Bootstrap: {label} (n={summary['n']})"))
    print(f"  Mean:   {summary['mean']:.2f} delta")
    print(f"  Median: {summary['median']:.2f} delta")
    print(f"  SD:     {summary['sd']:.2f} delta")
    print(f"  95% CI: [{summary['ci_lower']:.1f},"
          f" {summary['ci_upper']:.1f}] delta")

    # 出現頻度が5回超の閾値のみヒストグラム表示
    distribution = summary["distribution"]
    rounded = np.round(distribution, 1)
    print("  Distribution:")
    for value in sorted(set(rounded)):
        count = np.sum(rounded == value)
        if count > 5:
            percentage = count / len(distribution) * 100
            print(f"    {value:5.1f}: {count:4d} ({percentage:5.1f}%)")


def print_ppv_npv_table(model_metrics, prevalences=None):
    """
    各モデルのPPV/NPVを有病率別のテーブル形式で表示する。

    Args:
        model_metrics: モデル名→{sensitivity, specificity} の辞書
        prevalences:   評価する有病率のリスト
    """
    if prevalences is None:
        prevalences = [0.02, 0.03]

    print(_format_separator("PPV / NPV at Realistic Screening Prevalence"))

    # ヘッダ行
    header = f"{'Model':<25s}"
    for prev in prevalences:
        header += f"  Prev {prev * 100:.0f}% PPV   NPV "
    print(header)
    print("-" * (25 + 22 * len(prevalences)))

    # 各モデルの行
    for model_name, metrics in model_metrics.items():
        row = f"{model_name:<25s}"
        for prev in prevalences:
            ppv, npv = compute_ppv_npv(
                metrics["sensitivity"], metrics["specificity"], prev
            )
            row += f"  {ppv:>8.3f} {npv:>5.3f} "
        print(row)


# =============================================================================
# CV結果から平均指標を抽出するヘルパー
# =============================================================================

def _extract_mean_metrics_from_fold_list(fold_results):
    """fold結果のリスト（辞書のリスト）から感度・特異度の平均を抽出する。"""
    return {
        "sensitivity": np.mean([f["sensitivity"] for f in fold_results]),
        "specificity": np.mean([f["specificity"] for f in fold_results]),
    }


def _extract_mean_metrics_from_dict(metric_dict):
    """ML結果辞書（リストの辞書）から感度・特異度の平均を抽出する。"""
    return {
        "sensitivity": np.mean(metric_dict["sensitivity"]),
        "specificity": np.mean(metric_dict["specificity"]),
    }


# =============================================================================
# メイン処理
# =============================================================================

def main():
    """全解析ステップを順に実行する。"""

    # --- 1. データ読み込み ---
    X, y = load_data(CSV_PATH)
    near_scores    = X[:, 0]  # 近見（33cm）の水平斜視角
    distant_scores = X[:, 1]  # 遠見（5m）の水平斜視角

    # --- 2. 単一測定値によるROC閾値解析 ---
    near_cv_results = run_single_measurement_cv(y, near_scores)
    print_cv_summary(near_cv_results, "Near (33cm)")

    distant_cv_results = run_single_measurement_cv(y, distant_scores)
    print_cv_summary(distant_cv_results, "Distance (5m)")

    # --- 3. 機械学習モデルの交差検証 ---
    ml_results = run_ml_cross_validation(X, y)
    print_cv_summary(ml_results["LR"],   "LR (Near + Distance)")
    print_cv_summary(ml_results["LGBM"], "LGBM (Near + Distance)")

    # --- 4. ブートストラップ安定性解析 ---
    boot_near = run_bootstrap_threshold_analysis(y, near_scores)
    print_bootstrap_summary(boot_near, "Near threshold")

    boot_distant = run_bootstrap_threshold_analysis(y, distant_scores)
    print_bootstrap_summary(boot_distant, "Distance threshold")

    # --- 5. PPV / NPV（現実的な有病率での臨床的有用性評価） ---
    print_ppv_npv_table({
        "Near (14.4 delta)":    _extract_mean_metrics_from_fold_list(near_cv_results),
        "Distance (6.4 delta)": _extract_mean_metrics_from_fold_list(distant_cv_results),
        "LR (Near + Distance)": _extract_mean_metrics_from_dict(ml_results["LR"]),
        "LGBM (Near + Distance)": _extract_mean_metrics_from_dict(ml_results["LGBM"]),
    })


if __name__ == "__main__":
    main()
