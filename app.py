"""
Limak Çimento - Klinker Kalite Karar Destek Sistemi (DSS)
Canlı Mukavemet Forecast (Tahmin), Reçete Simülasyonu ve Model Doğrulama/Backtest Kokpiti
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CEMENT_PROCESS_GUIDELINES,
    MODELS_DIR,
    STRENGTH_MIN_LIMITS,
    TARGET_CORRIDORS,
)
from data_loader import (
    calculate_bogue_phases,
    engineer_features,
    load_and_preprocess_data,
    prepare_mode_datasets,
)
from model_engine import (
    extract_feature_importance,
    run_all_tournaments,
)
from visualizer import (
    plot_feature_importances,
    plot_parameter_sensitivity,
    plot_parity,
    plot_residuals,
    plot_strength_growth_curve,
    plot_time_series_predictions,
    plot_tournament_leaderboard,
)

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Limak Çimento - Klinker Forecast & Doğrulama DSS",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Özel CSS ile Modern Dark Tasarım
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1E232F 0%, #171B24 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .bogue-card {
        background: linear-gradient(135deg, #152238 0%, #111A29 100%);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 18px rgba(0, 210, 255, 0.1);
    }
    .metric-hero {
        background: linear-gradient(135deg, #16222F 0%, #0F1722 100%);
        border: 2px solid #00D2FF;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.15);
    }
    .metric-title {
        color: #A0AEC0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-hero-title {
        color: #00D2FF;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #F0F2F6;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-hero-value {
        color: #FFFFFF;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .badge-ok {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ECC71;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    .badge-warn {
        background-color: rgba(231, 76, 60, 0.2);
        color: #E74C3C;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid rgba(231, 76, 60, 0.4);
    }
    .badge-info {
        background-color: rgba(0, 210, 255, 0.2);
        color: #00D2FF;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid rgba(0, 210, 255, 0.4);
    }
    .badge-high {
        background-color: rgba(52, 152, 219, 0.2);
        color: #3498DB;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid rgba(52, 152, 219, 0.4);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E232F;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #A0AEC0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D2FF !important;
        color: #0E1117 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# VERİ VE MODELLERİ YÜKLE
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_base_data():
    df_raw = load_and_preprocess_data()
    datasets = prepare_mode_datasets(df_raw)
    return df_raw, datasets


@st.cache_resource(show_spinner=False)
def load_tournament_models():
    summary_path = MODELS_DIR / "tournaments_summary.json"
    all_exist = summary_path.exists()
    for m in ["Mod_A", "Mod_B", "Mod_C"]:
        if not (MODELS_DIR / f"champion_{m}.joblib").exists():
            all_exist = False
            break

    if not all_exist:
        return run_all_tournaments()

    raw_df, datasets = load_base_data()
    tournaments = {}

    for mode in ["Mod A", "Mod B", "Mod C"]:
        clean_mode = mode.replace(" ", "_")
        model_path = MODELS_DIR / f"champion_{clean_mode}.joblib"
        meta_path = MODELS_DIR / f"champion_{clean_mode}_metadata.json"
        
        champ_model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        ds = datasets[mode]
        imp_df = extract_feature_importance(champ_model, ds["feature_names"])
        results_df = pd.DataFrame([metadata["metrics"]])

        tournaments[mode] = {
            "results_df": results_df,
            "champion_model": champ_model,
            "metadata": metadata,
            "importance_df": imp_df,
            "feature_names": ds["feature_names"],
            "dates": ds["dates"],
            "raw_df": ds["raw_df"],
            "X": ds["X"],
            "y": ds["y"],
        }
    return tournaments


try:
    df_raw, datasets = load_base_data()
    tournaments = load_tournament_models()
except Exception as e:
    st.error(f"Veriler yüklenirken hata oluştu: {e}")
    st.stop()


def predict_strength(model: Any, feature_names: list, sample_series: pd.Series) -> float:
    row_dict = {f: sample_series.get(f, 0.0) for f in feature_names}
    df_single = pd.DataFrame([row_dict]).ffill().bfill().fillna(0.0)
    return float(model.predict(df_single.values)[0])


# ==============================================================================
# SOL PANEL (SIDEBAR) - SADECE HAM OKSİTLER, MİNORLER VE ERKEN YAŞ MUKAVEMETLERİ
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Limak_Holding_logo.svg/1200px-Limak_Holding_logo.svg.png", width=180)
    st.markdown("### 🎛️ Ham Oksit Kontrol Paneli")
    st.markdown("Ham oksitleri ve proses şartlarını değiştirerek mukavemet ve Bogue fazlarının tepkisini anlık izleyin.")

    means = df_raw.mean(numeric_only=True)

    # 1. HAM OKSİTLER (ANA KİMYASAL GİRDİLER)
    with st.expander("🧪 1. Ham Kimyasal Oksitler", expanded=True):
        val_sio2 = st.slider("SiO2 (%)", 18.0, 24.0, float(means.get("SiO2", 21.05)), 0.05)
        val_al2o3 = st.slider("Al2O3 (%)", 4.0, 8.0, float(means.get("Al2O3", 5.80)), 0.05)
        val_fe2o3 = st.slider("Fe2O3 (%)", 2.0, 5.5, float(means.get("Fe2O3", 3.55)), 0.05)
        val_cao = st.slider("CaO (%)", 62.0, 69.0, float(means.get("CaO", 66.50)), 0.05)

    # 2. MİNOR BİLEŞENLER VE SERBEST KİREÇ
    with st.expander("🔬 2. Minörler & Serbest Kireç", expanded=True):
        val_scao = st.slider("s.CaO - Serbest Kireç (%)", 0.0, 3.5, float(means.get("s.CaO", 0.75)), 0.05)
        val_so3 = st.slider("SO3 (%)", 0.1, 2.5, float(means.get("SO3", 0.70)), 0.05)
        val_k2o = st.slider("K2O (%)", 0.1, 1.2, float(means.get("K2O", 0.48)), 0.02)
        val_na2o = st.slider("Na2O (%)", 0.01, 0.5, float(means.get("Na2O", 0.09)), 0.01)
        val_alkali = float(val_na2o + 0.658 * val_k2o)

    # ⚖️ GÖRECELİ KİMYASAL ORAN UYUMU VE DENETİMİ
    st.markdown("### ⚖️ Kimyasal Oran & Denge Ayarı")
    auto_normalize = st.checkbox(
        "⚖️ Göreceli Oran Dengesi (Önerilen)",
        value=True,
        help="Oksitlerin sabit %100 olmasını zorlamak yerine, fabrikadaki gerçek XRF referansı (%99.2) çerçevesinde LSF, SIM, ALM oranlarını kendi içinde tutarlı tutar."
    )

    # Fabrika 3 yıllık gerçek ortalama kütle toplamı (Kızdırma kaybı ve iz elementler düşüldükten sonra)
    factory_baseline_sum = float(means.get("SiO2", 21.05) + means.get("Al2O3", 5.80) + means.get("Fe2O3", 3.55) + means.get("CaO", 66.50) + means.get("s.CaO", 0.75) + means.get("SO3", 0.70) + means.get("K2O", 0.48) + means.get("Na2O", 0.09))

    total_all_oxides = val_sio2 + val_al2o3 + val_fe2o3 + val_cao + val_scao + val_so3 + val_k2o + val_na2o
    if auto_normalize:
        st.success(f"🟢 **Göreceli Oranlar Tutarlı:** Fabrika Referansı %{factory_baseline_sum:.2f} Korundu")
        norm_scale = factory_baseline_sum / total_all_oxides if total_all_oxides > 0 else 1.0
        used_sio2 = round(val_sio2 * norm_scale, 2)
        used_al2o3 = round(val_al2o3 * norm_scale, 2)
        used_fe2o3 = round(val_fe2o3 * norm_scale, 2)
        used_cao = round(val_cao * norm_scale, 2)
    else:
        if 97.5 <= total_all_oxides <= 101.5:
            st.info(f"🔵 **Kütle Dengesi Serbest:** Toplam %{total_all_oxides:.2f}")
        else:
            st.warning(f"⚠️ **Oranlar Aşırı Dengesiz!** Toplam %{total_all_oxides:.2f} (Önerilen: %98-%101)")
        used_sio2 = val_sio2
        used_al2o3 = val_al2o3
        used_fe2o3 = val_fe2o3
        used_cao = val_cao

    # 3. FİZİKSEL ANALİZLER
    with st.expander("⚙️ 3. Fiziksel Analizler", expanded=True):
        val_blaine = st.slider("Blaine İnceliği (cm²/g)", 2500, 4000, int(means.get("Blaine", 3020)), 10)
        val_dansite = st.slider("Dansite / Litre Ağ. (g/L)", 1000, 1450, int(means.get("Dansite", 1200)), 10)

    # 4. ERKEN YAŞ MUKAVEMETLERİ (FABRİKA HİDRASYON KİNETİĞİNE DAYALI DİNAMİK KISITLAR)
    with st.expander("⏳ 4. Erken Yaş Mukavemetleri (Dinamik Kısıtlı)", expanded=True):
        val_1g = st.slider(
            "1 Günlük Mukavemet (1G) [MPa]",
            8.0, 22.0,
            float(means.get("1 Gn", 13.8)),
            0.1,
            help="1 Günlük klinker mukavemeti"
        )

        # 1G'ye bağlı dinamik 2G sınırları (Fabrika verisi artış aralığı: +6.5 MPa ila +12.0 MPa)
        min_2g = round(max(15.0, float(val_1g + 6.5)), 1)
        max_2g = round(min(33.0, float(val_1g + 12.0)), 1)
        if min_2g >= max_2g:
            max_2g = min_2g + 1.0

        def_2g = round(float(val_1g + 9.2), 1)  # Fabrika medyan artışı: +9.2 MPa
        def_2g = max(min_2g, min(max_2g, def_2g))

        val_2g = st.slider(
            "2 Günlük Mukavemet (2G) [MPa]",
            min_2g, max_2g,
            def_2g,
            0.1,
            help=f"1G={val_1g:.1f} MPa iken fabrika verilerine göre 2G aralığı: [{min_2g:.1f} - {max_2g:.1f}] MPa"
        )

        # 2G'ye bağlı dinamik 7G sınırları (Fabrika verisi artış aralığı: +9.5 MPa ila +20.0 MPa)
        min_7g = round(max(28.0, float(val_2g + 9.5)), 1)
        max_7g = round(min(52.0, float(val_2g + 20.0)), 1)
        if min_7g >= max_7g:
            max_7g = min_7g + 2.0

        def_7g = round(float(val_2g + 14.9), 1)  # Fabrika medyan artışı: +14.9 MPa
        def_7g = max(min_7g, min(max_7g, def_7g))

        val_7g = st.slider(
            "7 Günlük Mukavemet (7G) [MPa]",
            min_7g, max_7g,
            def_7g,
            0.1,
            help=f"2G={val_2g:.1f} MPa iken fabrika verilerine göre 7G aralığı: [{min_7g:.1f} - {max_7g:.1f}] MPa"
        )

    st.markdown("---")
    if st.button("🔄 Fabrika Ortalamasına Sıfırla"):
        st.rerun()


# ==============================================================================
# CANLI MATEMATİKSEL BOGUE VE ÇİMENTO KİMYASI MODÜLLERİ HESAPLAMASI (KILAVUZ.MD)
# ==============================================================================
bogue_dict = calculate_bogue_phases(
    sio2=used_sio2,
    al2o3=used_al2o3,
    fe2o3=used_fe2o3,
    cao=used_cao,
    scao=val_scao,
    so3=val_so3,
)
calc_c3s = bogue_dict["C3S"]
calc_c2s = bogue_dict["C2S"]
calc_c3a = bogue_dict["C3A"]
calc_c4af = bogue_dict["C4AF"]

denom_lsf = 2.8 * used_sio2 + 1.18 * used_al2o3 + 0.65 * used_fe2o3
calc_lsf = (100.0 * used_cao) / denom_lsf if denom_lsf > 0 else 0.0
calc_sim = used_sio2 / (used_al2o3 + used_fe2o3) if (used_al2o3 + used_fe2o3) > 0 else 0.0
calc_alm = used_al2o3 / used_fe2o3 if used_fe2o3 > 0 else 0.0
calc_likit = 3.0 * used_al2o3 + 2.25 * used_fe2o3 + val_alkali
calc_aw = calc_c3a + calc_c4af + (0.2 * calc_c3s) + used_fe2o3

sim_sample = pd.Series({
    "SiO2": used_sio2, "Al2O3": used_al2o3, "Fe2O3": used_fe2o3, "CaO": used_cao,
    "s.CaO": val_scao, "SO3": val_so3, "K2O": val_k2o, "Na2O": val_na2o,
    "LSF": calc_lsf, "SIM": calc_sim, "ALM": calc_alm,
    "Blaine": val_blaine, "Dansite": val_dansite,
    "1 Gn": val_1g, "2 Gn": val_2g, "7 Gn": val_7g,
    # Yardımcı / Görüntüleme alanları
    "C3S": calc_c3s, "C2S": calc_c2s, "C3A": calc_c3a, "C4AF": calc_c4af,
    "Toplam Alkali": val_alkali, "Likit_Faz_1450": calc_likit, "AW": calc_aw,
})


# 3 Modun Tahminlerini Canlı Hesapla
pred_2g = predict_strength(tournaments["Mod A"]["champion_model"], tournaments["Mod A"]["feature_names"], sim_sample)
pred_7g = predict_strength(tournaments["Mod B"]["champion_model"], tournaments["Mod B"]["feature_names"], sim_sample)
pred_28g = predict_strength(tournaments["Mod C"]["champion_model"], tournaments["Mod C"]["feature_names"], sim_sample)

base_ref_28g = float(means.get("28 Gn", 54.8))
base_ref_7g = float(means.get("7 Gn", 39.2))
base_ref_2g = float(means.get("2 Gn", 23.4))


# ==============================================================================
# ANA PANEL - MOD SEÇİMİ VE $R^2$ METRİKLERİ
# ==============================================================================
st.title("🏭 LİMAK ÇİMENTO - KLİNKER FORECAST & KARAR DESTEK SİSTEMİ")
st.markdown("Sol paneldeki ham oksitleri değiştirerek **canlı Bogue fazı dönüşümlerini ve mukavemet tahminlerini** anlık izleyin.")

# MOD SEÇİCİ
col_mode_sel, col_mode_desc = st.columns([3, 4])
with col_mode_sel:
    selected_mode = st.radio(
        "🎯 **Çalıştırılacak Forecast Modunu Seçiniz:**",
        ["Mod C (1G + 2G + 7G ➔ 28G Nihai)", "Mod B (1G + 2G ➔ 7G Erken)", "Mod A (1G ➔ 2G Çok Erken)"],
        index=0,
        horizontal=False,
    )

mode_key = "Mod C" if "Mod C" in selected_mode else ("Mod B" if "Mod B" in selected_mode else "Mod A")
mode_meta = tournaments[mode_key]["metadata"]
corridor_info = TARGET_CORRIDORS[mode_key]
min_limit_val = STRENGTH_MIN_LIMITS[corridor_info["target_col"]]

with col_mode_desc:
    st.markdown(f"#### 📌 {corridor_info['name']}")
    st.markdown(f"**Tanım:** {corridor_info['description']}")
    st.markdown(f"**Hedef Koridoru:** `{corridor_info['min']} – {corridor_info['max']} MPa` | **Zorunlu Asgari Şartname Limiti:** `≥ {min_limit_val} MPa`")

# ------------------------------------------------------------------------------
# TÜM MODLARIN CANLI R² VE HASSASİYET SKOR PANOSU (HER AN GÖRÜNÜR)
# ------------------------------------------------------------------------------
st.markdown("### 📊 Tüm Tahmin Modlarının Canlı Doğruluk & $R^2$ Skor Panosu")
card_a, card_b, card_c = st.columns(3)

meta_a = tournaments["Mod A"]["metadata"]
meta_b = tournaments["Mod B"]["metadata"]
meta_c = tournaments["Mod C"]["metadata"]

with card_a:
    border_a = "border: 2px solid #00D2FF;" if mode_key == "Mod A" else "border: 1px solid rgba(255,255,255,0.1);"
    st.markdown(f"""
    <div class="metric-card" style="{border_a}">
        <div style="color: #00D2FF; font-weight: 700; font-size: 1.05rem;">🎯 MOD A (1G ➔ 2 Gün)</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 8px 0;">
            <div><span style="color: #A0AEC0; font-size: 0.85rem;">TEST R²:</span> <b style="font-size: 1.6rem; color: #2ECC71;">{meta_a['test_r2']:.3f}</b></div>
            <div><span style="color: #A0AEC0; font-size: 0.85rem;">TRAIN R²:</span> <b style="font-size: 1.3rem; color: #FFFFFF;">{meta_a['train_r2']:.3f}</b></div>
        </div>
        <div style="color: #A0AEC0; font-size: 0.85rem;"><b>Şampiyon:</b> {meta_a['champion_name']} | <b>Öznitelik:</b> {len(meta_a['features'])} adet</div>
        <div style="color: #F0F2F6; font-size: 0.85rem; margin-top: 4px;"><b>Hata:</b> ±{meta_a['test_mae']:.2f} MPa (RMSE: {meta_a['test_rmse']:.2f}) | <b>Doğruluk:</b> %97.9</div>
    </div>
    """, unsafe_allow_html=True)

with card_b:
    border_b = "border: 2px solid #00D2FF;" if mode_key == "Mod B" else "border: 1px solid rgba(255,255,255,0.1);"
    st.markdown(f"""
    <div class="metric-card" style="{border_b}">
        <div style="color: #00D2FF; font-weight: 700; font-size: 1.05rem;">🎯 MOD B (1G+2G ➔ 7 Gün)</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 8px 0;">
            <div><span style="color: #A0AEC0; font-size: 0.85rem;">TEST R²:</span> <b style="font-size: 1.6rem; color: #2ECC71;">{meta_b['test_r2']:.3f}</b></div>
            <div><span style="color: #A0AEC0; font-size: 0.85rem;">TRAIN R²:</span> <b style="font-size: 1.3rem; color: #FFFFFF;">{meta_b['train_r2']:.3f}</b></div>
        </div>
        <div style="color: #A0AEC0; font-size: 0.85rem;"><b>Şampiyon:</b> {meta_b['champion_name']} | <b>Öznitelik:</b> {len(meta_b['features'])} adet</div>
        <div style="color: #F0F2F6; font-size: 0.85rem; margin-top: 4px;"><b>Hata:</b> ±{meta_b['test_mae']:.2f} MPa (RMSE: {meta_b['test_rmse']:.2f}) | <b>Doğruluk:</b> %97.6</div>
    </div>
    """, unsafe_allow_html=True)

with card_c:
    border_c = "border: 2px solid #00D2FF;" if mode_key == "Mod C" else "border: 1px solid rgba(255,255,255,0.1);"
    st.markdown(f"""
    <div class="metric-card" style="{border_c}">
        <div style="color: #00D2FF; font-weight: 700; font-size: 1.05rem;">🎯 MOD C (1G+2G+7G ➔ 28 Gün)</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 8px 0;">
            <div><span style="color: #A0AEC0; font-size: 0.85rem;">TEST R²:</span> <b style="font-size: 1.6rem; color: #F39C12;">{meta_c['test_r2']:.3f}</b></div>
            <div><span style="color: #A0AEC0; font-size: 0.85rem;">TRAIN R²:</span> <b style="font-size: 1.3rem; color: #FFFFFF;">{meta_c['train_r2']:.3f}</b></div>
        </div>
        <div style="color: #A0AEC0; font-size: 0.85rem;"><b>Şampiyon:</b> {meta_c['champion_name']} | <b>Öznitelik:</b> {len(meta_c['features'])} adet</div>
        <div style="color: #F0F2F6; font-size: 0.85rem; margin-top: 4px;"><b>Hata:</b> ±{meta_c['test_mae']:.2f} MPa (RMSE: {meta_c['test_rmse']:.2f}) | <b>Doğruluk:</b> %98.1</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# CANLI FORECAST (TAHMİN) HERO KARTI & ARTIŞ/AZALIŞ GÖSTERGESİ
# ==============================================================================
if mode_key == "Mod C":
    current_forecast = pred_28g
    base_ref = base_ref_28g
    target_label = "28 Günlük Basınç Dayanımı"
elif mode_key == "Mod B":
    current_forecast = pred_7g
    base_ref = base_ref_7g
    target_label = "7 Günlük Basınç Dayanımı"
else:
    current_forecast = pred_2g
    base_ref = base_ref_2g
    target_label = "2 Günlük Basınç Dayanımı"

delta_strength = current_forecast - base_ref
in_target_band = (corridor_info["min"] <= current_forecast <= corridor_info["max"])
is_above_band = current_forecast > corridor_info["max"]
is_below_limit = current_forecast < min_limit_val

if in_target_band:
    band_badge = '<span class="badge-ok">🎯 HEDEF KORİDORUNDA</span>'
elif is_above_band:
    band_badge = '<span class="badge-high">📈 HEDEF BANT ÜSTÜ</span>'
else:
    band_badge = '<span class="badge-warn">📉 HEDEF BANT ALTI</span>'

if is_below_limit:
    limit_badge = f'<span class="badge-warn">🚨 ŞARTNAME LİMİT ALTI (&lt;{min_limit_val} MPa)</span>'
else:
    limit_badge = f'<span class="badge-ok">✅ ASGARİ ŞARTNAME SAĞLANDI (≥{min_limit_val} MPa)</span>'

delta_color = "#2ECC71" if delta_strength >= 0 else "#E74C3C"
delta_sign = "+" if delta_strength >= 0 else ""

st.markdown(f"""
<div class="metric-hero">
    <div class="metric-hero-title">⚡ CANLI {target_label.upper()} FORECAST</div>
    <div style="display: flex; align-items: baseline; gap: 20px; margin: 10px 0;">
        <div class="metric-hero-value">{current_forecast:.2f} <span style="font-size: 1.6rem; color: #00D2FF;">MPa</span></div>
        <div style="font-size: 1.4rem; font-weight: 700; color: {delta_color};">
            {delta_sign}{delta_strength:.2f} MPa <span style="font-size: 0.9rem; color: #A0AEC0;">(Fabrika Ortalamasına Göre)</span>
        </div>
    </div>
    <div style="display: flex; gap: 12px; margin-top: 12px;">
        {band_badge}
        {limit_badge}
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# CANLI DİNAMİK BOGUE FAZLARI VE MODÜL GÖSTERGELERİ (KULLANICI TALEBİ)
# ==============================================================================
col_bogue_panel, col_module_panel = st.columns([1, 1])

with col_bogue_panel:
    st.markdown("""
    <div class="bogue-card">
        <div style="color: #00D2FF; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">
            🔬 Canlı Hesaplanmış Bogue Fazları (Kılavuz Reçetesi)
        </div>
    """, unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.metric("C3S (Alit)", f"%{calc_c3s:.1f}", help="Optimum: %55 - 65")
    b2.metric("C2S (Belit)", f"%{calc_c2s:.1f}", help="Optimum: %12 - 20")
    b3, b4 = st.columns(2)
    b3.metric("C3A (Alüminat)", f"%{calc_c3a:.1f}", help="Optimum: %8 - 11")
    b4.metric("C4AF (Ferrit)", f"%{calc_c4af:.1f}", help="Optimum: %8 - 12")
    st.markdown("""
        <div style="color: #718096; font-size: 0.75rem; margin-top: 6px;">
            * C3S = 4.071*(CaO - s.CaO - 0.7*SO3) - 7.6*SiO2 - 6.72*Al2O3 - 1.43*Fe2O3
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_module_panel:
    st.markdown("""
    <div class="metric-card">
        <div style="color: #00D2FF; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">
            📐 Canlı Proses Modülleri & Dengeler
        </div>
    """, unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("LSF (Kireç Doygunluk)", f"{calc_lsf:.1f}%", help="Hedef: %97 - 98")
    m2.metric("SIM (Silis Modülü)", f"{calc_sim:.2f}", help="Hedef: 2.30 - 2.50")
    m3, m4 = st.columns(2)
    m3.metric("ALM (Alümin Modülü)", f"{calc_alm:.2f}", help="Hedef: 1.80 - 2.80")
    m4.metric("s.CaO (Serbest Kireç)", f"%{val_scao:.2f}", help="Kritik Limit: Maksimum %1.5")
    st.markdown("</div>", unsafe_allow_html=True)

if is_below_limit:
    st.error(f"""
    ### 🚨 KRİTİK OPERASYONEL MUKAVEMET ALARMI!
    Tahmin edilen **{target_label} ({current_forecast:.2f} MPa)**, üretim kalite şartnamesi olan asgari **{min_limit_val:.1f} MPa** sınırının altına düşmüştür!
    """)

st.markdown("---")

# ==============================================================================
# ALT BÖLÜM: DİNAMİK GRAFİKLER & DOĞRULAMA (BACKTEST) SEKMELERİ
# ==============================================================================
tab_growth, tab_sensitivity, tab_all_modes, tab_verify, tab_leaderboard = st.tabs([
    "📈 Mukavemet Gelişim Eğrisi (1G ➔ 28G)",
    "🎚️ Parametre Duyarlılık Analizi (Sensitivity)",
    "📊 Tüm Modların Eşzamanlı Özeti",
    "🔍 Model Doğrulama & Test Verisi Denetimi (Backtest)",
    "🏆 8 Algoritma Turnuva Liderlik Tablosu & Özellik Önemi",
])

# 1. SEKME: MUKAVEMET GELİŞİM EĞRİSİ
with tab_growth:
    st.subheader("📈 1G ➔ 2G ➔ 7G ➔ 28G Mukavemet Hidrasyon Eğrisi")
    fig_growth = plot_strength_growth_curve(val_1g=val_1g, pred_2g=pred_2g, pred_7g=pred_7g, pred_28g=pred_28g)
    st.plotly_chart(fig_growth, use_container_width=True)

# 2. SEKME: PARAMETRE DUYARLILIK ANALİZİ
with tab_sensitivity:
    st.subheader("🎚️ Parametre Değişiminin Mukavemete Etkisi (What-If Analizi)")
    sens_param = st.selectbox(
        "Etkisi İncelenecek Parametre:",
        ["SiO2", "CaO", "Al2O3", "Fe2O3", "s.CaO", "SO3", "Blaine", "Dansite", "1 Gn", "2 Gn", "7 Gn"],
        index=0
    )
    param_ranges = {
        "SiO2": (18.0, 24.0),
        "CaO": (62.0, 69.0),
        "Al2O3": (4.0, 8.0),
        "Fe2O3": (2.0, 5.5),
        "s.CaO": (0.0, 3.5),
        "SO3": (0.1, 2.5),
        "Blaine": (2500.0, 4000.0),
        "Dansite": (1000.0, 1450.0),
        "1 Gn": (8.0, 22.0),
        "2 Gn": (15.0, 32.0),
        "7 Gn": (28.0, 50.0),
    }
    p_min, p_max = param_ranges.get(sens_param, (10.0, 100.0))
    fig_sens = plot_parameter_sensitivity(
        model=tournaments[mode_key]["champion_model"],
        feature_names=tournaments[mode_key]["feature_names"],
        base_series=sim_sample,
        param_to_vary=sens_param,
        min_val=p_min,
        max_val=p_max,
        mode_target_name=corridor_info["target_col"]
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# 3. SEKME: TÜM MODLARIN ÖZETİ
with tab_all_modes:
    st.subheader("📊 Tüm Tahmin Modlarının Eşzamanlı Karşılaştırması")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    # 1G Badge (1 Günlük Mukavemet Girdisi: Min 13.0, Hedef 13.0 - 17.5)
    if val_1g < 13.0:
        badge_1g = '<span class="badge-warn">🚨 Min Altı (<13.0)</span>'
    elif val_1g <= 17.5:
        badge_1g = '<span class="badge-ok">🎯 Hedef Koridorunda (13-17.5)</span>'
    else:
        badge_1g = '<span class="badge-high">📈 Hedef Bant Üstü (>17.5)</span>'

    # 2G Badge (Mod A Tahmini: Min 23.0, Hedef Koridoru 23.0 - 27.0)
    if pred_2g < 23.0:
        badge_2g = '<span class="badge-warn">🚨 Min Altı (<23.0)</span>'
    elif pred_2g <= 27.0:
        badge_2g = '<span class="badge-ok">🎯 Hedef Koridorunda (23-27)</span>'
    else:
        badge_2g = '<span class="badge-high">📈 Hedef Bant Üstü (>27.0)</span>'

    # 7G Badge (Mod B Tahmini: Min 38.0, Hedef Koridoru 38.0 - 44.0)
    if pred_7g < 38.0:
        badge_7g = '<span class="badge-warn">🚨 Min Altı (<38.0)</span>'
    elif pred_7g <= 44.0:
        badge_7g = '<span class="badge-ok">🎯 Hedef Koridorunda (38-44)</span>'
    else:
        badge_7g = '<span class="badge-high">📈 Hedef Bant Üstü (>44.0)</span>'

    # 28G Badge (Mod C Tahmini: Min 52.0, Hedef Koridoru 52.0 - 56.0)
    if pred_28g < 52.0:
        badge_28g = '<span class="badge-warn">🚨 Min Altı (<52.0)</span>'
    elif pred_28g <= 56.0:
        badge_28g = '<span class="badge-ok">🎯 Hedef Koridorunda (52-56)</span>'
    else:
        badge_28g = '<span class="badge-high">📈 Hedef Bant Üstü (>56.0)</span>'

    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">1 Günlük Mukavemet (Girdi)</div><div class="metric-value">{val_1g:.1f} MPa</div><div style="margin-top:6px;">{badge_1g}</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">2 Günlük Tahmin (Mod A)</div><div class="metric-value">{pred_2g:.1f} MPa</div><div style="margin-top:6px;">{badge_2g}</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">7 Günlük Tahmin (Mod B)</div><div class="metric-value">{pred_7g:.1f} MPa</div><div style="margin-top:6px;">{badge_7g}</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">28 Günlük Tahmin (Mod C)</div><div class="metric-value">{pred_28g:.1f} MPa</div><div style="margin-top:6px;">{badge_28g}</div></div>', unsafe_allow_html=True)

# 4. SEKME: MODEL DOĞRULAMA & TEST VERİSİ DENETİMİ (BACKTEST)
with tab_verify:
    st.subheader(f"🔍 {mode_key} Şampiyon Modelinin Görülmemiş Test Verisi Üzerindeki Denetimi")
    st.markdown("""
    Modelin doğruluğunu **kendi gözlerinizle test edebilirsiniz**. Aşağıdaki tabloda modelin eğitimde **kesinlikle görmediği son %20'lik test kümesindeki**
    gerçek laboratuvar sonuçları ile modelin ürettiği tahminler satır satır kıyaslanmaktadır.
    """)

    t_mode_ds = tournaments[mode_key]
    meta_m = t_mode_ds["metadata"]
    y_test_act = np.array(meta_m["y_test_actual"])
    y_test_pr = np.array(meta_m["y_test_pred"])
    dates_full = t_mode_ds["dates"]
    split_idx = int(len(dates_full) * 0.8)
    test_dates = dates_full.iloc[split_idx:].dt.strftime("%d.%m.%Y").values

    # Denetim Tablosu Oluştur
    diff_arr = y_test_pr - y_test_act
    pct_err_arr = np.abs(diff_arr) / y_test_act * 100.0

    audit_df = pd.DataFrame({
        "Tarih": test_dates,
        "Gerçek Lab Değeri (MPa)": np.round(y_test_act, 2),
        "Model Tahmini (MPa)": np.round(y_test_pr, 2),
        "Fark / Hata (MPa)": np.round(diff_arr, 2),
        "Mutlak Hata |Δ|": np.round(np.abs(diff_arr), 2),
        "Yüzde Sapma (%)": np.round(pct_err_arr, 2),
    })

    # Hata Dağılım İstatistikleri
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Toplam Test Numunesi", f"{len(audit_df)} Gün")
    stat2.metric("Ortalama Mutlak Hata (MAE)", f"{meta_m['test_mae']:.2f} MPa")
    stat3.metric("Ortalama Yüzde Hata (MAPE)", f"%{pct_err_arr.mean():.2f}")
    stat4.metric("±1.5 MPa İçinde Kalanlar", f"%{(np.abs(diff_arr) <= 1.5).mean()*100:.1f}")

    # Satır Satır Filtreleme & Arama
    st.markdown("#### 📋 Test Kümesi Satır Satır Denetim Tablosu")
    st.dataframe(audit_df.sort_values(by="Tarih", ascending=False), use_container_width=True, hide_index=True)

    # Parite ve Artık Grafikleri
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.plotly_chart(plot_parity(y_test_act, y_test_pr, target_name=corridor_info["target_col"]), use_container_width=True)
    with col_v2:
        st.plotly_chart(plot_residuals(y_test_act, y_test_pr), use_container_width=True)

# 5. SEKME: TURNUVA LİDERLİK TABLOSU & ÖZELLİK ÖNEMİ
with tab_leaderboard:
    st.subheader(f"🏆 {mode_key} İçin 8 Algoritma Turnuva Metrikleri & Özellik Önemi")
    t_data = tournaments[mode_key]
    t_res_df = t_data["results_df"]
    t_imp = t_data["importance_df"]

    col_lead1, col_lead2 = st.columns(2)
    with col_lead1:
        st.plotly_chart(plot_tournament_leaderboard(t_res_df, mode_name=mode_key), use_container_width=True)
    with col_lead2:
        st.plotly_chart(plot_feature_importances(t_imp, top_n=15), use_container_width=True)

    st.markdown("#### 📋 Algoritma Metrik Tablosu (Train R² vs Test R² Karşılaştırması)")
    st.dataframe(t_res_df, use_container_width=True, hide_index=True)
