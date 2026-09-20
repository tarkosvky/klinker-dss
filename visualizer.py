"""
Limak Çimento - Klinker Kalite Karar Destek Sistemi (DSS)
İnteraktif Görselleştirme, Canlı Duyarlılık, Parite ve Mukavemet Gelişim Grafikleri
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BG_DARK = "#0E1117"
BG_CARD = "#1E232F"
TEXT_COLOR = "#F0F2F6"
COLOR_CYAN = "#00D2FF"
COLOR_CORAL = "#FF7675"
COLOR_GREEN = "#2ECC71"
COLOR_RED = "#E74C3C"
COLOR_ORANGE = "#F39C12"
COLOR_GRID = "rgba(255, 255, 255, 0.08)"


def apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "x": 0.02,
            "y": 0.96,
            "font": {"size": 16, "color": TEXT_COLOR, "family": "Segoe UI, sans-serif"},
        },
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_CARD,
        font={"color": TEXT_COLOR, "family": "Segoe UI, sans-serif"},
        hovermode="x unified",
        legend={
            "bgcolor": "rgba(20, 24, 33, 0.8)",
            "bordercolor": "rgba(255, 255, 255, 0.1)",
            "borderwidth": 1,
            "font": {"size": 12},
        },
        margin={"l": 45, "r": 35, "t": 60, "b": 45},
        xaxis={"gridcolor": COLOR_GRID, "linecolor": "rgba(255, 255, 255, 0.2)", "zeroline": False},
        yaxis={"gridcolor": COLOR_GRID, "linecolor": "rgba(255, 255, 255, 0.2)", "zeroline": False},
    )
    return fig


def plot_strength_growth_curve(
    val_1g: float,
    pred_2g: float,
    pred_7g: float,
    pred_28g: float,
    corridor_2g: tuple = (23.0, 27.0),
    corridor_7g: tuple = (40.0, 45.0),
    corridor_28g: tuple = (52.0, 53.0),
) -> go.Figure:
    days = [1, 2, 7, 28]
    strengths = [val_1g, pred_2g, pred_7g, pred_28g]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days,
        y=strengths,
        mode="lines+markers+text",
        text=[f"<b>{s:.1f} MPa</b>" for s in strengths],
        textposition="top center",
        textfont={"size": 14, "color": COLOR_CYAN},
        line={"color": COLOR_CYAN, "width": 4, "shape": "spline"},
        marker={"size": 12, "color": COLOR_CYAN, "line": {"color": "#FFFFFF", "width": 2}},
        name="Tahmin Edilen Mukavemet Gelişimi",
    ))

    min_limits = [13.0, 23.0, 40.0, 54.0]
    fig.add_trace(go.Scatter(
        x=days,
        y=min_limits,
        mode="lines+markers",
        line={"color": COLOR_RED, "width": 2, "dash": "dash"},
        marker={"size": 6, "color": COLOR_RED},
        name="Şartname Asgari Limitleri (13, 23, 40, 54 MPa)",
    ))

    fig.add_trace(go.Scatter(x=[2, 2], y=[corridor_2g[0], corridor_2g[1]], mode="lines+markers", line={"color": COLOR_GREEN, "width": 6}, name=f"2G Hedef ({corridor_2g[0]}-{corridor_2g[1]} MPa)"))
    fig.add_trace(go.Scatter(x=[7, 7], y=[corridor_7g[0], corridor_7g[1]], mode="lines+markers", line={"color": COLOR_GREEN, "width": 6}, name=f"7G Hedef ({corridor_7g[0]}-{corridor_7g[1]} MPa)"))
    fig.add_trace(go.Scatter(x=[28, 28], y=[corridor_28g[0], corridor_28g[1]], mode="lines+markers", line={"color": COLOR_GREEN, "width": 6}, name=f"28G Hedef ({corridor_28g[0]}-{corridor_28g[1]} MPa)"))

    fig.update_xaxes(title_text="Klinker Hidrasyon Yaşı (Gün)", tickvals=[1, 2, 7, 28], ticktext=["1G", "2G", "7G", "28G"])
    fig.update_yaxes(title_text="Basınç Dayanımı (MPa)")
    return apply_dark_theme(fig, "Klinker Mukavemet Gelişim Eğrisi (1G ➔ 2G ➔ 7G ➔ 28G)")


def plot_parameter_sensitivity(
    model: any,
    feature_names: list,
    base_series: pd.Series,
    param_to_vary: str,
    min_val: float,
    max_val: float,
    mode_target_name: str = "28G",
    n_points: int = 30,
) -> go.Figure:
    param_values = np.linspace(min_val, max_val, n_points)
    predicted_strengths = []

    for val in param_values:
        temp_s = base_series.copy()
        temp_s[param_to_vary] = val
        # Oksit değişiminde modülleri dinamik güncelle
        denom_lsf = 2.8 * float(temp_s.get("SiO2", 21.0)) + 1.18 * float(temp_s.get("Al2O3", 5.8)) + 0.65 * float(temp_s.get("Fe2O3", 3.5))
        temp_s["LSF"] = (100.0 * float(temp_s.get("CaO", 66.0))) / denom_lsf if denom_lsf > 0 else 0.0
        denom_sim = float(temp_s.get("Al2O3", 5.8)) + float(temp_s.get("Fe2O3", 3.5))
        temp_s["SIM"] = float(temp_s.get("SiO2", 21.0)) / denom_sim if denom_sim > 0 else 0.0
        denom_alm = float(temp_s.get("Fe2O3", 3.5))
        temp_s["ALM"] = float(temp_s.get("Al2O3", 5.8)) / denom_alm if denom_alm > 0 else 0.0

        row_dict = {f: temp_s.get(f, 0.0) for f in feature_names}
        df_eval = pd.DataFrame([row_dict]).ffill().bfill().fillna(0.0)
        p = float(model.predict(df_eval.values)[0])
        predicted_strengths.append(p)

    current_val = float(base_series.get(param_to_vary, (min_val + max_val) / 2))
    curr_row = {f: base_series.get(f, 0.0) for f in feature_names}
    df_curr = pd.DataFrame([curr_row]).ffill().bfill().fillna(0.0)
    current_strength = float(model.predict(df_curr.values)[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=param_values, y=predicted_strengths, mode="lines", line={"color": COLOR_CYAN, "width": 3}, name="Mukavemet Tepkisi"))
    fig.add_trace(go.Scatter(x=[current_val], y=[current_strength], mode="markers", marker={"size": 14, "color": COLOR_ORANGE, "line": {"color": "#FFFFFF", "width": 2}}, name=f"Mevcut Değer: {current_val:.2f} ➔ {current_strength:.1f} MPa"))
    fig.update_xaxes(title_text=f"{param_to_vary} Değeri")
    fig.update_yaxes(title_text=f"Tahmin Edilen {mode_target_name} Mukavemeti (MPa)")
    return apply_dark_theme(fig, f"{param_to_vary} Değişiminin {mode_target_name} Mukavemetine Etkisi (Duyarlılık Analizi)")


def plot_parity(y_actual: np.ndarray, y_pred: np.ndarray, target_name: str = "28G") -> go.Figure:
    min_val = min(np.min(y_actual), np.min(y_pred)) * 0.95
    max_val = max(np.max(y_actual), np.max(y_pred)) * 1.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", line={"color": "#F39C12", "width": 2, "dash": "dash"}, name="İdeal Tahmin (y = x)"))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val * 1.05, max_val * 1.05], mode="lines", line={"color": "rgba(255, 255, 255, 0.2)", "dash": "dot"}, name="+%5 Sapma", showlegend=False))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val * 0.95, max_val * 0.95], mode="lines", line={"color": "rgba(255, 255, 255, 0.2)", "dash": "dot"}, fill="tonexty", fillcolor="rgba(255, 255, 255, 0.04)", name="±%5 Tolerans"))

    diff = np.abs(y_actual - y_pred)
    fig.add_trace(go.Scatter(
        x=y_actual,
        y=y_pred,
        mode="markers",
        marker={"size": 7, "color": diff, "colorscale": "Plasma", "colorbar": {"title": "Hata |Δ|"}, "showscale": True, "line": {"color": "#FFFFFF", "width": 0.5}},
        name="Test Numuneleri",
        hovertemplate="Gerçek: %{x:.2f} MPa<br>Tahmin: %{y:.2f} MPa<extra></extra>",
    ))
    fig.update_xaxes(title_text="Laboratuvar Gerçek Değeri (MPa)", range=[min_val, max_val])
    fig.update_yaxes(title_text="Model Tahmini (MPa)", range=[min_val, max_val])
    return apply_dark_theme(fig, f"{target_name} Test Kümesi Parite Grafiği (Gerçek vs Tahmin)")


def plot_residuals(y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    residuals = y_actual - y_pred
    mean_err = float(np.mean(residuals))
    std_err = float(np.std(residuals))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, marker={"color": "#6C5CE7", "line": {"color": "#FFFFFF", "width": 0.5}}, name="Hata Dağılımı"))
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#2ECC71", annotation_text="Sıfır Hata")
    fig.add_vline(x=mean_err, line_width=1.5, line_color="#F1C40F", annotation_text=f"Ort: {mean_err:.2f}")
    fig.update_xaxes(title_text="Hata Payı (Gerçek - Tahmin) [MPa]")
    fig.update_yaxes(title_text="Numune Sayısı (Adet)")
    return apply_dark_theme(fig, f"Test Kümesi Hata Dağılımı [Ort: {mean_err:.2f}, Std: {std_err:.2f} MPa]")


def plot_tournament_leaderboard(results_df: pd.DataFrame, mode_name: str = "Mod C") -> go.Figure:
    df_plot = results_df.copy().sort_values(by="Test R²", ascending=True)

    colors = []
    for _, row in df_plot.iterrows():
        if "OVERFITTING" in str(row["Status"]):
            colors.append("#E74C3C")
        elif "DISQUALIFIED" in str(row["Status"]):
            colors.append("#7F8C8D")
        else:
            colors.append("#2ECC71")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_plot["Model Name"],
        x=df_plot["Test R²"],
        orientation="h",
        marker={"color": colors},
        text=df_plot.apply(lambda r: f"Test R²: {r['Test R²']:.3f} | Train R²: {r['Train R²']:.3f} ({r['Status']})", axis=1),
        textposition="auto",
        name="Test R²",
    ))
    fig.update_xaxes(title_text="Görülmemiş Test Kümesi R² Skoru")
    fig.update_yaxes(title_text="Algoritma")
    return apply_dark_theme(fig, f"{mode_name} Turnuvası - Test R² Başarısı")


def plot_feature_importances(imp_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top_df = imp_df.head(top_n).copy()
    total_top = top_df["Importance_Pct"].sum()
    if total_top > 0:
        top_df["Display_Pct"] = (top_df["Importance_Pct"] / total_top) * 100.0
    else:
        top_df["Display_Pct"] = 100.0 / len(top_df)

    top_df = top_df.sort_values(by="Display_Pct", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_df["Feature"],
        x=top_df["Display_Pct"],
        orientation="h",
        marker={"color": top_df["Display_Pct"], "colorscale": "Tealgrn", "showscale": False},
        text=top_df["Display_Pct"].apply(lambda v: f"%{v:.1f}"),
        textposition="outside",
    ))
    fig.update_xaxes(title_text="Göreceli Etki Payı Dağılımı (%) [Toplam: %100]")
    fig.update_yaxes(title_text="Fiziksel / Kimyasal Parametre")
    return apply_dark_theme(fig, f"Klinker Mukavemetine Göreceli Etki Payı Dağılımı (Toplam: %100)")


def plot_time_series_predictions(dates, y_actual, y_pred, target_name="28 Gn", corridor_min=52.0, corridor_max=53.0, limit_min=54.0):
    df_plot = pd.DataFrame({"Tarih": dates, "Gerçek (MPa)": y_actual, "Tahmin (MPa)": y_pred}).dropna().sort_values("Tarih")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["Tarih"], y=df_plot["Gerçek (MPa)"], mode="lines+markers", name=f"Gerçek {target_name}", line={"color": COLOR_CYAN, "width": 2}))
    fig.add_trace(go.Scatter(x=df_plot["Tarih"], y=df_plot["Tahmin (MPa)"], mode="lines", name=f"Tahmin {target_name}", line={"color": COLOR_CORAL, "width": 2, "dash": "dot"}))
    fig.update_yaxes(title_text="Basınç Dayanımı (MPa)")
    return apply_dark_theme(fig, f"{target_name} Zaman Serisi Gerçek vs Tahmin")
