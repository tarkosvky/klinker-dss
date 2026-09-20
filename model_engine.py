"""
Limak Çimento - Klinker Kalite Karar Destek Sistemi (DSS)
Maksimum Doğruluk ve Genelleme Başarılı Model Turnuvası Motoru
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CV_SPLITS,
    MODELS_DIR,
    OVERFITTING_THRESHOLD,
    RANDOM_STATE,
    TRAIN_SPLIT_RATIO,
)
from data_loader import load_and_preprocess_data, prepare_mode_datasets

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_candidate_models() -> Dict[str, Any]:
    """
    En yüksek genelleme başarısına sahip 8 makine öğrenimi modelini oluşturur.
    """
    models = {
        "BayesianRidge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", BayesianRidge()),
        ]),
        "Ridge (alpha=10.0)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]),
        "HuberRegressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(max_iter=1000, alpha=10.0)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.1, random_state=RANDOM_STATE)),
        ]),
        "StackingRegressor": StackingRegressor(
            estimators=[
                ("bayesian", Pipeline([("scaler", StandardScaler()), ("model", BayesianRidge())])),
                ("huber", Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(max_iter=1000, alpha=10.0))])),
                ("ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE))])),
            ],
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
            cv=5,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.85,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=6,
            l2_regularization=2.0,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def evaluate_model_cv(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = CV_SPLITS,
) -> Tuple[float, float]:
    """Zaman serisi çapraz doğrulama (TimeSeriesSplit) ile CV R2 ortalama ve standart sapması."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores: List[float] = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        val_pred = fold_model.predict(X_val)
        fold_r2 = r2_score(y_val, val_pred)
        cv_scores.append(fold_r2)

    return float(np.mean(cv_scores)), float(np.std(cv_scores))


def run_model_tournament(
    X: pd.DataFrame,
    y: pd.Series,
    mode_name: str = "Mod C",
    train_ratio: float = TRAIN_SPLIT_RATIO,
    overfit_threshold: float = OVERFITTING_THRESHOLD,
    save_champion: bool = True,
) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """Model turnuvasını çalıştırır, ezber denetimi yapar ve şampiyonu kaydeder."""
    logger.info(f"=== {mode_name} MODEL TURNUVASI BAŞLATILIYOR (Örneklem: {len(X)}) ===")

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values

    candidate_models = build_candidate_models()
    results_records: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Any] = {}

    for name, model in candidate_models.items():
        try:
            cv_mean, cv_std = evaluate_model_cv(model, X_train_np, y_train_np, n_splits=CV_SPLITS)

            model.fit(X_train_np, y_train_np)
            fitted_models[name] = model

            train_preds = model.predict(X_train_np)
            test_preds = model.predict(X_test_np)

            train_r2 = r2_score(y_train_np, train_preds)
            test_r2 = r2_score(y_test_np, test_preds)
            test_rmse = np.sqrt(mean_squared_error(y_test_np, test_preds))
            test_mae = mean_absolute_error(y_test_np, test_preds)

            overfitting_gap = train_r2 - test_r2

            if test_r2 < 0:
                status = "DISQUALIFIED (Negative R²)"
            elif overfitting_gap > overfit_threshold:
                status = "OVERFITTING_DISQUALIFIED"
            else:
                status = "Valid"

            results_records.append({
                "Model Name": name,
                "Train R²": round(train_r2, 4),
                "Test R²": round(test_r2, 4),
                "CV R² Mean": round(cv_mean, 4),
                "CV R² Std": round(cv_std, 4),
                "Test RMSE": round(test_rmse, 4),
                "Test MAE": round(test_mae, 4),
                "Overfitting Gap": round(overfitting_gap, 4),
                "Status": status,
            })
        except Exception as e:
            logger.error(f"{name} hatası: {e}")

    results_df = pd.DataFrame(results_records)

    # Şampiyon Seçimi: Valid modeller arasından en yüksek Test R2 (veya en düşük Test RMSE)
    valid_models = results_df[results_df["Status"] == "Valid"]

    if not valid_models.empty:
        best_row = valid_models.sort_values(by=["Test R²", "CV R² Mean"], ascending=[False, False]).iloc[0]
    else:
        best_row = results_df.sort_values(by=["Test RMSE"], ascending=[True]).iloc[0]

    champion_name = best_row["Model Name"]
    logger.info(f"🏆 {mode_name} ŞAMPİYONU: {champion_name} (Test R²: {best_row['Test R²']}, CV R²: {best_row['CV R² Mean']}, RMSE: {best_row['Test RMSE']} MPa)")

    champion_model = fitted_models[champion_name]
    test_preds = champion_model.predict(X_test_np)
    train_preds = champion_model.predict(X_train_np)

    metadata = {
        "mode_name": mode_name,
        "champion_name": champion_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": list(X.columns),
        "metrics": best_row.to_dict(),
        "train_r2": float(r2_score(y_train_np, train_preds)),
        "test_r2": float(r2_score(y_test_np, test_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test_np, test_preds))),
        "test_mae": float(mean_absolute_error(y_test_np, test_preds)),
        "y_test_actual": y_test_np.tolist(),
        "y_test_pred": test_preds.tolist(),
    }

    if save_champion:
        clean_mode = mode_name.replace(" ", "_")
        model_path = MODELS_DIR / f"champion_{clean_mode}.joblib"
        meta_path = MODELS_DIR / f"champion_{clean_mode}_metadata.json"
        
        joblib.dump(champion_model, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n==================== {mode_name} TURNUVA SONUÇLARI ====================")
    print(tabulate(results_df, headers="keys", tablefmt="psql", showindex=False))
    print(f"======================================================================")

    return results_df, champion_model, metadata


def extract_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Öznitelik önem düzeylerini çıkarır, fiziksel parametrelere konsolide eder ve toplamı %100 yapar."""
    importances = None
    core_model = model

    if isinstance(model, Pipeline):
        core_model = model.named_steps.get("model", model)

    if hasattr(core_model, "feature_importances_"):
        importances = core_model.feature_importances_
    elif hasattr(core_model, "coef_"):
        importances = np.abs(core_model.coef_).flatten()
    elif isinstance(core_model, StackingRegressor):
        for _, est in core_model.named_estimators_.items():
            sub = est.named_steps.get("model", est) if isinstance(est, Pipeline) else est
            if hasattr(sub, "coef_"):
                importances = np.abs(sub.coef_).flatten()
                break

    if importances is None or len(importances) != len(feature_names):
        importances = np.ones(len(feature_names))

    df_raw_imp = pd.DataFrame({
        "Feature": feature_names,
        "Raw_Imp": importances,
    })

    def map_to_core_param(f_name: str) -> str:
        f = f_name.lower()
        if f == "sio2":
            return "SiO2 (Silis Oksit)"
        elif f == "al2o3":
            return "Al2O3 (Alümin Oksit)"
        elif f == "fe2o3":
            return "Fe2O3 (Demir Oksit)"
        elif f == "cao":
            return "CaO (Kalsiyum Oksit)"
        elif "s.cao" in f or "scao" in f:
            return "s.CaO (Serbest Kireç)"
        elif "so3" in f:
            return "SO3 (Sülfat Oksit)"
        elif "k2o" in f or "na2o" in f or "alkali" in f:
            return "Alkaliler (K2O, Na2O)"
        elif f == "lsf":
            return "LSF (Kireç Doygunluk Faktörü)"
        elif f == "sim":
            return "SIM (Silis Modülü)"
        elif f == "alm":
            return "ALM (Alümin Modülü)"
        elif "blaine" in f:
            return "Blaine İnceliği"
        elif "dansite" in f:
            return "Dansite (Litre Ağırlığı)"
        elif "7 gn" in f or "7g" in f:
            return "7 Günlük Mukavemet (7G)"
        elif "2 gn" in f or "2g" in f:
            return "2 Günlük Mukavemet (2G)"
        elif "1 gn" in f or "1g" in f:
            return "1 Günlük Mukavemet (1G)"
        elif "c3s" in f or "alite" in f:
            return "C3S (Alit Fazı)"
        elif "c2s" in f or "belit" in f:
            return "C2S (Belit Fazı)"
        elif "c3a" in f:
            return "C3A (Alüminat Fazı)"
        elif "c4af" in f:
            return "C4AF (Ferrit Fazı)"
        else:
            return f_name

    df_raw_imp["Core_Group"] = df_raw_imp["Feature"].apply(map_to_core_param)
    df_grouped = df_raw_imp.groupby("Core_Group")["Raw_Imp"].sum().reset_index()
    df_grouped = df_grouped.rename(columns={"Core_Group": "Feature", "Raw_Imp": "Importance"})

    total_imp = df_grouped["Importance"].sum()
    if total_imp > 0:
        df_grouped["Importance_Pct"] = (df_grouped["Importance"] / total_imp) * 100.0
    else:
        df_grouped["Importance_Pct"] = 100.0 / len(df_grouped)

    df_grouped = df_grouped.sort_values(by="Importance_Pct", ascending=False).reset_index(drop=True)
    return df_grouped


def run_all_tournaments() -> Dict[str, Dict[str, Any]]:
    """Mod A, Mod B ve Mod C turnuvalarını koşturur ve şampiyonları kaydeder."""
    raw_df = load_and_preprocess_data()
    datasets = prepare_mode_datasets(raw_df)
    tournaments_summary = {}

    for mode in ["Mod A", "Mod B", "Mod C"]:
        ds = datasets[mode]
        res_df, champ, meta = run_model_tournament(
            X=ds["X"],
            y=ds["y"],
            mode_name=mode,
            save_champion=True,
        )
        imp_df = extract_feature_importance(champ, ds["feature_names"])
        tournaments_summary[mode] = {
            "results_df": res_df,
            "champion_model": champ,
            "metadata": meta,
            "importance_df": imp_df,
            "feature_names": ds["feature_names"],
            "dates": ds["dates"],
            "raw_df": ds["raw_df"],
            "X": ds["X"],
            "y": ds["y"],
        }

    summary_meta_path = MODELS_DIR / "tournaments_summary.json"
    serializable_summary = {
        m: {
            "champion_name": d["metadata"]["champion_name"],
            "metrics": d["metadata"]["metrics"],
            "features_count": len(d["metadata"]["features"]),
        }
        for m, d in tournaments_summary.items()
    }
    with open(summary_meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Tüm turnuvalar tamamlandı. Özet: {summary_meta_path}")
    return tournaments_summary


if __name__ == "__main__":
    summary = run_all_tournaments()
