"""
Limak Çimento - Klinker Kalite Karar Destek Sistemi (DSS)
Veri Yükleme, Ön İşleme ve Çimento Hidrasyon Kinetiği Modülü
"""

from typing import Dict, List, Optional, Tuple
import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    COLUMN_RENAMING_MAP,
    DATA_DIR,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_column_name(col: str) -> str:
    """Sütun isimlerini normalize eder, Türkçe karakter ve boşluk uyumsuzluklarını giderir."""
    col_str = str(col).strip().replace("\ufeff", "")
    for k, v in COLUMN_RENAMING_MAP.items():
        if col_str == k or col_str.lower() == k.lower():
            return v
    c_lower = col_str.lower().replace(" ", "").replace("ı", "i").replace("ý", "i")
    if "blaine" in c_lower:
        return "Blaine"
    if c_lower in ["gun", "gn"]:
        return "Gn"
    if "toplamalkali" in c_lower:
        return "Toplam Alkali"
    if c_lower in ["1gun", "1gn", "1g"]:
        return "1 Gn"
    if c_lower in ["2gun", "2gn", "2g"]:
        return "2 Gn"
    if c_lower in ["7gun", "7gn", "7g"]:
        return "7 Gn"
    if c_lower in ["28gun", "28gn", "28g"]:
        return "28 Gn"
    return col_str


def load_single_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Farklı encoding ve Türkçe ondalık virgül formatlarını destekleyerek CSV dosyasını okur."""
    encodings = ["utf-8-sig", "iso-8859-9", "windows-1254", "utf-8", "latin1"]
    
    for enc in encodings:
        try:
            df = pd.read_csv(
                filepath,
                sep=";",
                encoding=enc,
                decimal=",",
                skiprows=[1],
                low_memory=False,
            )
            df.columns = [clean_column_name(c) for c in df.columns]
            
            # Klinker dosyası doğrulama: Gn ve temel klinker sütunları (SiO2/CaO ve mukavemet/dansite) içermelidir
            is_klinker = (
                "Gn" in df.columns
                and ("SiO2" in df.columns or "CaO" in df.columns)
                and any(c in df.columns for c in ["1 Gn", "2 Gn", "7 Gn", "28 Gn", "Dansite"])
            )
            if is_klinker:
                logger.info(f"Klinker dosyası yüklendi: {filepath.name} ({enc}, {len(df)} satır)")
                return df
        except Exception:
            continue

    logger.warning(f"Dosya okunamadı: {filepath}")
    return None


def parse_date_series(series: pd.Series) -> pd.Series:
    """Tarih sütununu ('dd.mm.yyyy' veya 'd.m.yyyy') hızlı ve güvenli şekilde datetime formatına çevirir."""
    s = series.astype(str).str.strip()
    s = s.replace({"nan": None, "None": None, "": None})
    return pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")


def load_and_preprocess_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Tüm klinker CSV verilerini yükler, birleştirir, temizler ve kronolojik sıralar."""
    all_csvs = [Path(p) for p in sorted(glob.glob(str(data_dir / "*.csv")))]
    klinker_files = [
        p for p in all_csvs
        if p.name.lower().startswith("veriler")
        or p.name.lower().startswith("2023 1.dönem")
        or p.name.lower().startswith("klinker")
    ]
    if not klinker_files:
        klinker_files = [p for p in all_csvs if "besleme" not in p.name.lower() and p.stat().st_size < 5_000_000]

    dfs: List[pd.DataFrame] = []
    for p_path in klinker_files:
        df_single = load_single_csv(p_path)
        if df_single is not None:
            dfs.append(df_single)

    if not dfs:
        raise ValueError("Hiçbir CSV verisi başarıyla yüklenemedi!")

    combined = pd.concat(dfs, ignore_index=True)
    combined["Gn_dt"] = parse_date_series(combined["Gn"])
    combined = combined.dropna(subset=["Gn_dt"]).copy()

    numeric_cols = [c for c in combined.columns if c not in ["Gn", "Gn_dt"]]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined[combined[numeric_cols].notna().any(axis=1)].copy()

    combined["non_null_count"] = combined[numeric_cols].notna().sum(axis=1)
    combined = (
        combined.sort_values(["Gn_dt", "non_null_count"], ascending=[True, False])
        .drop_duplicates(subset=["Gn_dt"], keep="first")
        .drop(columns=["non_null_count"])
        .reset_index(drop=True)
    )

    logger.info(f"Tekilleştirilmiş toplam veri: {combined.shape} ({combined['Gn_dt'].min().strftime('%d.%m.%Y')} - {combined['Gn_dt'].max().strftime('%d.%m.%Y')})")
    return combined


def calculate_bogue_phases(
    sio2: float,
    al2o3: float,
    fe2o3: float,
    cao: float,
    scao: float = 0.0,
    so3: float = 0.0,
) -> Dict[str, float]:
    """
    Kılavuzdaki (kilavuz.md) formüllere göre Bogue fazlarını matematiksel olarak hesaplar:
    C3S = 4.071 * (CaO - s.CaO - 0.7 * SO3) - 7.600 * SiO2 - 6.720 * Al2O3 - 1.430 * Fe2O3
    C2S = 2.867 * SiO2 - 0.754 * C3S
    C3A = 2.650 * Al2O3 - 1.690 * Fe2O3
    C4AF = 3.043 * Fe2O3
    """
    c3s = 4.071 * (cao - scao - 0.7 * so3) - 7.600 * sio2 - 6.720 * al2o3 - 1.430 * fe2o3
    c3s = max(0.0, float(c3s))
    c2s = 2.867 * sio2 - 0.754 * c3s
    c2s = max(0.0, float(c2s))
    c3a = 2.650 * al2o3 - 1.690 * fe2o3
    c3a = max(0.0, float(c3a))
    c4af = 3.043 * fe2o3
    c4af = max(0.0, float(c4af))
    return {
        "C3S": round(c3s, 2),
        "C2S": round(c2s, 2),
        "C3A": round(c3a, 2),
        "C4AF": round(c4af, 2),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Çimento Kimyası Modülleri ve dinamik Bogue fazlarını hesaplar.
    Model girdileri olarak ham oksitler, modüller ve fiziksel parametreler kullanılır.
    """
    df = df.copy().sort_values("Gn_dt").reset_index(drop=True)

    # 1. Çimento Kimyası Modülleri
    denom_lsf = 2.8 * df["SiO2"] + 1.18 * df["Al2O3"] + 0.65 * df["Fe2O3"]
    df["LSF"] = np.where(denom_lsf > 0, (100.0 * df["CaO"]) / denom_lsf, np.nan)

    denom_sim = df["Al2O3"] + df["Fe2O3"]
    df["SIM"] = np.where(denom_sim > 0, df["SiO2"] / denom_sim, np.nan)
    df["ALM"] = np.where(df["Fe2O3"] > 0, df["Al2O3"] / df["Fe2O3"], np.nan)

    alkali_sum = (
        (df["K2O"].fillna(0) + df["Na2O"].fillna(0))
        if ("K2O" in df.columns and "Na2O" in df.columns)
        else df.get("Toplam Alkali", 0).fillna(0)
    )
    mgo_val = df.get("MgO", 0)
    if isinstance(mgo_val, pd.Series):
        mgo_val = mgo_val.fillna(0)
    df["Likit_Faz_1450"] = 3.0 * df["Al2O3"] + 2.25 * df["Fe2O3"] + mgo_val + alkali_sum

    # 2. Kılavuz Formülleriyle Bogue Fazlarının Dinamik Hesabı
    df["C3S_calc"] = np.maximum(
        0.0,
        4.071 * (df["CaO"] - df["s.CaO"].fillna(0) - 0.7 * df["SO3"].fillna(0))
        - 7.600 * df["SiO2"]
        - 6.720 * df["Al2O3"]
        - 1.430 * df["Fe2O3"]
    )
    df["C2S_calc"] = np.maximum(0.0, 2.867 * df["SiO2"] - 0.754 * df["C3S_calc"])
    df["C3A_calc"] = np.maximum(0.0, 2.650 * df["Al2O3"] - 1.690 * df["Fe2O3"])
    df["C4AF_calc"] = np.maximum(0.0, 3.043 * df["Fe2O3"])

    return df


def get_feature_columns_for_mode(mode: str) -> List[str]:
    """
    Belirli bir tahmin modu için çoklu bağlantısız, sadece ham oksitler,
    modüller, minörler ve fiziksel parametrelerden oluşan temiz öznitelik listesi döner.
    """
    base_features = [
        "SiO2", "Al2O3", "Fe2O3", "CaO", "s.CaO", "SO3", "K2O", "Na2O",
        "LSF", "SIM", "ALM", "Blaine", "Dansite"
    ]

    if mode == "Mod A":
        return base_features + ["1 Gn"]
    elif mode == "Mod B":
        return base_features + ["1 Gn", "2 Gn"]
    elif mode == "Mod C":
        return base_features + ["1 Gn", "2 Gn", "7 Gn"]
    else:
        raise ValueError(f"Geçersiz Mod: {mode}")


def prepare_mode_datasets(df: pd.DataFrame) -> Dict[str, Dict[str, any]]:
    """Mod A, Mod B ve Mod C için temizlenmiş hedef ve öznitelik matrislerini hazırlar."""
    df_feat = engineer_features(df)
    results = {}

    target_map = {
        "Mod A": "2 Gn",
        "Mod B": "7 Gn",
        "Mod C": "28 Gn",
    }

    for mode_name, target_col in target_map.items():
        feature_cols = get_feature_columns_for_mode(mode_name)
        valid_mask = df_feat[target_col].notna()
        if mode_name == "Mod A":
            valid_mask = valid_mask & df_feat["1 Gn"].notna()
        elif mode_name == "Mod B":
            valid_mask = valid_mask & df_feat["1 Gn"].notna() & df_feat["2 Gn"].notna()
        elif mode_name == "Mod C":
            valid_mask = valid_mask & df_feat["1 Gn"].notna() & df_feat["2 Gn"].notna() & df_feat["7 Gn"].notna()

        mode_df = df_feat[valid_mask].copy().reset_index(drop=True)
        avail_features = [c for c in feature_cols if c in mode_df.columns]
        
        X = mode_df[avail_features].copy()
        X = X.ffill().bfill().fillna(X.median())
        y = mode_df[target_col].copy()

        results[mode_name] = {
            "X": X,
            "y": y,
            "dates": mode_df["Gn_dt"],
            "raw_df": mode_df,
            "feature_names": avail_features,
            "target_name": target_col,
        }
        logger.info(f"{mode_name} veri seti hazırlandı: X={X.shape}, y={len(y)}")

    return results

