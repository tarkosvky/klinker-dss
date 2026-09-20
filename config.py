"""
Limak Çimento - Klinker Kalite Karar Destek Sistemi (DSS)
Konfigürasyon ve Sabit Değerler Modülü
"""

from pathlib import Path

# ==============================================================================
# 1. DOSYA VE DİZİN YOLLARI
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. MUKAVEMET MİNİMUM LİMİTLERİ (OPERASYONEL UYARI EŞİKLERİ)
# ==============================================================================
# Kullanıcı Şartı: 1 Gün min 13.0 | 2 Gün min 23.0 | 7 Gün min 38.0 | 28 Gün min 52.0 MPa
STRENGTH_MIN_LIMITS = {
    "1 Gn": 13.0,
    "2 Gn": 23.0,
    "7 Gn": 38.0,
    "28 Gn": 52.0,
}

# ==============================================================================
# 3. HEDEF MUKAVEMET KORİDORLARI (HEDEF BANTLARI - MPa)
# ==============================================================================
TARGET_CORRIDORS = {
    "Mod A": {
        "name": "Mod A (1G ➔ 2G Tahmini)",
        "target_col": "2 Gn",
        "min": 23.0,
        "max": 27.0,
        "description": "1 Günlük mukavemetten 2 Günlük mukavemet tahmini",
    },
    "Mod B": {
        "name": "Mod B (1G + 2G ➔ 7G Tahmini)",
        "target_col": "7 Gn",
        "min": 38.0,
        "max": 44.0,
        "description": "1G ve 2G mukavemetlerinden 7 Günlük mukavemet tahmini",
    },
    "Mod C": {
        "name": "Mod C (1G + 2G + 7G ➔ 28G Tahmini)",
        "target_col": "28 Gn",
        "min": 52.0,
        "max": 56.0,
        "description": "1G, 2G ve 7G mukavemetlerinden 28 Günlük nihai mukavemet tahmini",
    },
}

# ==============================================================================
# 4. ÇİMENTO KİMYASI PROSES ÇALIŞMA ARALIKLARI & FORMÜL REÇETESİ
# ==============================================================================
CEMENT_PROCESS_GUIDELINES = {
    "LSF": {"min": 97.0, "max": 98.0, "unit": "%", "name": "Kireç Doygunluk Katsayısı"},
    "SIM": {"min": 2.30, "max": 2.50, "unit": "oran", "name": "Silikat Modülü (SM)"},
    "ALM": {"min": 1.80, "max": 2.80, "unit": "oran", "name": "Alümina Modülü (AM)"},
    "AW": {"critical": 28.0, "name": "Anzast İndeksi (AW)"},
    "s.CaO": {"max": 1.5, "unit": "%", "name": "Serbest Kireç"},
    "C3S": {"min": 55.0, "opt": 60.0, "unit": "%", "name": "Alit"},
    "C2S": {"opt": 15.0, "unit": "%", "name": "Belit"},
    "C3A": {"min": 8.0, "max": 11.0, "unit": "%", "name": "Trikalsiyum Alüminat"},
    "C4AF": {"min": 8.0, "max": 10.0, "unit": "%", "name": "Tetrakalsiyum Alümina Ferrit"},
    "SO3": {"max": 3.5, "unit": "%", "name": "Kükürt Trioksit"},
    "Blaine": {"min": 3000, "unit": "cm²/g", "name": "Blaine İnceliği"},
    "Dansite": {"min": 1200, "unit": "g/L", "name": "Klinker Dansitesi"},
}

# ==============================================================================
# 5. SÜTUN İSİMLERİ VE NORMALİZASYON SÖZLÜĞÜ
# ==============================================================================
COLUMN_RENAMING_MAP = {
    "Gün": "Gn",
    "Gun": "Gn",
    "Gn": "Gn",
    "\ufeffGn": "Gn",
    "Toplam Alkali Kontenti": "Toplam Alkali",
    "Blaıne": "Blaine",
    "Blaýne": "Blaine",
    "Blane": "Blaine",
    "Blane": "Blaine",
    "1 Gün": "1 Gn",
    "1 Gn": "1 Gn",
    "1G": "1 Gn",
    "1Gn": "1 Gn",
    "2 Gün": "2 Gn",
    "2 Gn": "2 Gn",
    "2G": "2 Gn",
    "2Gn": "2 Gn",
    "7 Gün": "7 Gn",
    "7 Gn": "7 Gn",
    "7G": "7 Gn",
    "7Gn": "7 Gn",
    "28 Gün": "28 Gn",
    "28 Gn": "28 Gn",
    "28G": "28 Gn",
    "28Gn": "28 Gn",
}

RAW_FEATURE_COLUMNS = [
    "SiO2", "Al2O3", "Fe2O3", "CaO", "SO3", "K2O", "Na2O",
    "Dansite", "s.CaO", "C3S", "C2S", "C3A", "C4AF", "Alk Eq.", "Toplam Alkali", "Blaine"
]

TARGET_COLUMNS = ["1 Gn", "2 Gn", "7 Gn", "28 Gn"]

# ==============================================================================
# 6. MODEL EĞİTİMİ VE TURNUVA AYARLARI
# ==============================================================================
TRAIN_SPLIT_RATIO = 0.80  # İlk %80 Train, son %20 Test
CV_SPLITS = 5             # TimeSeriesSplit n_splits
RANDOM_STATE = 42
OVERFITTING_THRESHOLD = 0.20  # Delta R2 (Train R2 - Test R2) > 0.20 ise Disqualified
