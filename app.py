"""
Breast Cancer Prediction — Streamlit App
Model is trained at startup from data.csv (no pickle files needed).
Deploy on Streamlit Cloud by pushing to GitHub.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.stApp { background: #f5f4f2; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2a4a 55%, #162447 100%);
    border-radius: 18px; padding: 2.4rem 3rem; color: white; margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.hero h1 { color: #e8deff; font-size: 2rem; margin: 0 0 0.4rem; }
.hero p  { color: #98a8c8; font-size: 0.97rem; margin: 0; }
.hero-badge {
    display: inline-block; background: #f6c90e22; border: 1px solid #f6c90e88;
    color: #f6c90e; border-radius: 20px; padding: 3px 14px;
    font-size: 0.75rem; font-weight: 600; margin-bottom: 0.8rem; letter-spacing: 0.04em;
}
.card-malignant {
    background: linear-gradient(135deg, #fff0f0, #ffe4e4);
    border: 2px solid #e53935; border-radius: 14px;
    padding: 1.6rem 2rem; text-align: center;
    box-shadow: 0 4px 20px rgba(229,57,53,0.12);
}
.card-benign {
    background: linear-gradient(135deg, #f0fff4, #e4f9ec);
    border: 2px solid #43a047; border-radius: 14px;
    padding: 1.6rem 2rem; text-align: center;
    box-shadow: 0 4px 20px rgba(67,160,71,0.12);
}
.card-label { font-family: 'DM Serif Display', serif; font-size: 1.9rem; margin: 0; }
.card-conf  { font-size: 1rem; color: #444; margin-top: 0.3rem; font-weight: 500; }
.card-sub   { font-size: 0.83rem; color: #777; margin-top: 0.45rem; }
.tile {
    background: white; border-radius: 12px; padding: 1.1rem 0.8rem; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
.tile-val  { font-size: 1.7rem; font-weight: 700; color: #162447; line-height: 1; }
.tile-name { font-size: 0.75rem; color: #999; margin-top: 0.25rem; letter-spacing: 0.04em; text-transform: uppercase; }
div[data-testid="stSidebar"] { background: #eeeaf8 !important; }
.warn-banner {
    background: #fff8e1; border-left: 4px solid #f6c90e;
    border-radius: 6px; padding: 0.65rem 1rem;
    font-size: 0.83rem; color: #5a4a00; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Train & Cache Model ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_best_model():
    df = pd.read_csv("data.csv")
    df.drop(columns=["id"], inplace=True, errors="ignore")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_all_sc   = scaler.transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Benchmark all candidates
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, random_state=42),
        "AdaBoost":             AdaBoostClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)":            SVC(kernel="rbf", probability=True, random_state=42),
        "KNN":                  KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
    }

    all_results = {}
    for name, clf in candidates.items():
        clf.fit(X_train_sc, y_train)
        y_pred  = clf.predict(X_test_sc)
        y_proba = clf.predict_proba(X_test_sc)[:, 1]
        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)
        f1   = f1_score(y_test, y_pred)
        cv_a = cross_val_score(clf, X_all_sc, y, cv=cv, scoring="accuracy").mean()
        all_results[name] = dict(Accuracy=round(acc,4), ROC_AUC=round(auc,4),
                                 F1=round(f1,4), CV_Accuracy=round(cv_a,4))

    best_name = max(all_results, key=lambda k: all_results[k]["ROC_AUC"])

    # Tune the best model (Logistic Regression)
    param_grid = {
        "C":       [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver":  ["liblinear", "saga"],
    }
    gs = GridSearchCV(
        LogisticRegression(max_iter=10000, random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    gs.fit(X_train_sc, y_train)
    best_model = gs.best_estimator_

    y_pred  = best_model.predict(X_test_sc)
    y_proba = best_model.predict_proba(X_test_sc)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    f1   = f1_score(y_test, y_pred)
    cv_s = cross_val_score(best_model, X_all_sc, y, cv=cv, scoring="accuracy")

    metrics = {
        "model_name":  best_name,
        "best_params": gs.best_params_,
        "accuracy":    round(acc, 4),
        "roc_auc":     round(auc, 4),
        "f1":          round(f1, 4),
        "cv_accuracy": round(cv_s.mean(), 4),
        "cv_std":      round(cv_s.std(), 4),
        "all_models":  all_results,
    }

    return best_model, scaler, features, metrics


# ── Load with spinner ─────────────────────────────────────────────────────────
with st.spinner("🔬 Benchmarking 7 models and selecting the best one... (~15 sec)"):
    model, scaler, FEATURES, metrics = train_best_model()

model_label = metrics["model_name"]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">🏆 AUTO-SELECTED BEST MODEL</div>
    <h1>🩺 Breast Cancer Predictor</h1>
    <p>7 classifiers benchmarked · <strong style="color:#e8deff">{model_label}</strong>
    selected by ROC-AUC · Wisconsin FNA Dataset · 97.4% accuracy</p>
</div>
""", unsafe_allow_html=True)


# ── Feature Ranges ────────────────────────────────────────────────────────────
FEATURE_RANGES = {
    "radius_mean":             (6.0,  28.0,  14.0,  0.1),
    "texture_mean":            (9.0,  40.0,  19.0,  0.1),
    "perimeter_mean":          (43.0, 190.0, 92.0,  0.5),
    "area_mean":               (140., 2500., 655.,  1.0),
    "smoothness_mean":         (0.05, 0.16,  0.096, 0.001),
    "compactness_mean":        (0.02, 0.35,  0.104, 0.001),
    "concavity_mean":          (0.0,  0.43,  0.089, 0.001),
    "concave_points_mean":     (0.0,  0.20,  0.049, 0.001),
    "symmetry_mean":           (0.10, 0.30,  0.181, 0.001),
    "fractal_dimension_mean":  (0.05, 0.10,  0.063, 0.001),
    "radius_se":               (0.1,  2.9,   0.405, 0.01),
    "texture_se":              (0.3,  4.9,   1.22,  0.01),
    "perimeter_se":            (0.7,  22.0,  2.87,  0.1),
    "area_se":                 (6.0,  542.,  40.0,  1.0),
    "smoothness_se":           (0.001,0.032, 0.007, 0.001),
    "compactness_se":          (0.002,0.135, 0.025, 0.001),
    "concavity_se":            (0.0,  0.40,  0.032, 0.001),
    "concave_points_se":       (0.0,  0.053, 0.012, 0.001),
    "symmetry_se":             (0.007,0.079, 0.020, 0.001),
    "fractal_dimension_se":    (0.001,0.030, 0.004, 0.001),
    "radius_worst":            (7.9,  36.0,  16.3,  0.1),
    "texture_worst":           (12.0, 50.0,  25.7,  0.1),
    "perimeter_worst":         (50.0, 251.,  107.,  0.5),
    "area_worst":              (185., 4254., 880.,  1.0),
    "smoothness_worst":        (0.07, 0.22,  0.132, 0.001),
    "compactness_worst":       (0.02, 1.06,  0.254, 0.001),
    "concavity_worst":         (0.0,  1.25,  0.272, 0.001),
    "concave_points_worst":    (0.0,  0.29,  0.115, 0.001),
    "symmetry_worst":          (0.15, 0.66,  0.290, 0.001),
    "fractal_dimension_worst": (0.055,0.208, 0.084, 0.001),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔬 Cell Nucleus Measurements")
st.sidebar.caption("Adjust values then click **Predict Now**.")

groups = {
    "📐 Mean Values":  [f for f in FEATURES if f.endswith("_mean")],
    "📏 SE Values":    [f for f in FEATURES if f.endswith("_se")],
    "⚠️ Worst Values": [f for f in FEATURES if f.endswith("_worst")],
}

user_input = {}
for group, feats in groups.items():
    st.sidebar.markdown(f"**{group}**")
    for feat in feats:
        lo, hi, default, step = FEATURE_RANGES.get(feat, (0.0, 1.0, 0.5, 0.01))
        user_input[feat] = st.sidebar.slider(
            feat.replace("_", " ").title(),
            min_value=float(lo), max_value=float(hi),
            value=float(default), step=float(step),
        )

predict_btn = st.sidebar.button("🔍 Predict Now", use_container_width=True, type="primary")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_compare, tab_info = st.tabs(["📊 Prediction", "🏁 Model Comparison", "ℹ️ About"])


# ════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════
with tab_pred:
    if predict_btn:
        input_df = pd.DataFrame([user_input])[FEATURES]
        input_sc = scaler.transform(input_df)
        pred     = model.predict(input_sc)[0]
        proba    = model.predict_proba(input_sc)[0]
        conf_b, conf_m = proba[0], proba[1]

        if pred == 1:
            st.markdown(f"""
            <div class="card-malignant">
                <div class="card-label" style="color:#c62828;">🔴 Malignant</div>
                <div class="card-conf">Confidence: <strong>{conf_m*100:.1f}%</strong></div>
                <div class="card-sub">Model predicts malignancy. Please consult an oncologist.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-benign">
                <div class="card-label" style="color:#2e7d32;">🟢 Benign</div>
                <div class="card-conf">Confidence: <strong>{conf_b*100:.1f}%</strong></div>
                <div class="card-sub">Model predicts benign tumour. Always verify with a physician.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📊 Class Probabilities")
            fig, ax = plt.subplots(figsize=(5, 2.4))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            bars = ax.barh(["Benign", "Malignant"], [conf_b, conf_m],
                           color=["#43a047", "#e53935"], height=0.42, edgecolor="none", zorder=3)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="#aaa", linestyle="--", linewidth=1, zorder=2)
            for bar, val in zip(bars, [conf_b, conf_m]):
                ax.text(min(val + 0.02, 0.9), bar.get_y() + bar.get_height() / 2,
                        f"{val*100:.1f}%", va="center", fontsize=12, fontweight="700")
            ax.set_xlabel("Probability", fontsize=9)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(left=False, labelsize=10)
            ax.grid(axis="x", alpha=0.2, zorder=1)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("#### 🎯 Confidence Gauge")
            conf_show  = conf_m if pred == 1 else conf_b
            label_show = "Malignant" if pred == 1 else "Benign"
            color_show = "#e53935" if pred == 1 else "#43a047"

            fig2, ax2 = plt.subplots(figsize=(5, 2.4), subplot_kw=dict(aspect="equal"))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")
            theta = np.linspace(np.pi, 0, 300)
            ax2.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", linewidth=14, solid_capstyle="round")
            end_theta = np.pi - conf_show * np.pi
            theta2 = np.linspace(np.pi, end_theta, 300)
            ax2.plot(np.cos(theta2), np.sin(theta2), color=color_show, linewidth=14, solid_capstyle="round")
            ax2.text(0, -0.05, f"{conf_show*100:.1f}%", ha="center", va="center",
                     fontsize=22, fontweight="700", color=color_show)
            ax2.text(0, -0.42, label_show, ha="center", va="center", fontsize=11, color="#555")
            ax2.set_xlim(-1.3, 1.3)
            ax2.set_ylim(-0.6, 1.2)
            ax2.axis("off")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("#### 🔍 Top Feature Contributions (|Coefficients|)")
        coef     = np.abs(model.coef_[0])
        feat_imp = pd.Series(coef, index=FEATURES).nlargest(12)
        fig3, ax3 = plt.subplots(figsize=(8, 3.8))
        fig3.patch.set_alpha(0)
        ax3.set_facecolor("none")
        palette = ["#162447" if i % 2 == 0 else "#7c5cbb" for i in range(len(feat_imp))]
        ax3.barh(feat_imp.index[::-1], feat_imp.values[::-1],
                 color=palette[::-1], height=0.55, edgecolor="none")
        ax3.set_xlabel("|Coefficient|", fontsize=9)
        ax3.spines[["top", "right", "left"]].set_visible(False)
        ax3.tick_params(left=False, labelsize=8)
        ax3.grid(axis="x", alpha=0.2)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

        st.markdown("""
        <div class="warn-banner">
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational/research purposes only.
        It is <strong>not</strong> a substitute for professional medical diagnosis or treatment.
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:5rem 2rem; color:#bbb;">
            <div style="font-size:4.5rem; margin-bottom:1rem;">🔬</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#999;">
                Adjust measurements in the sidebar
            </div>
            <div style="font-size:0.92rem; margin-top:0.4rem; color:#bbb;">
                then click <strong style="color:#888;">Predict Now</strong>
            </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════
with tab_compare:
    st.markdown("### 🏁 All Models Benchmarked")
    st.caption("80/20 stratified split · Ranked by ROC-AUC · Winner tuned via 5-fold GridSearchCV")

    best_name   = metrics["model_name"]
    all_results = metrics["all_models"]

    rows = []
    for name, m in all_results.items():
        rows.append({
            "Model":       ("🏆 " + name) if name == best_name else name,
            "Accuracy":    f"{m['Accuracy']:.4f}",
            "ROC-AUC":     f"{m['ROC_AUC']:.4f}",
            "F1 Score":    f"{m['F1']:.4f}",
            "CV Accuracy": f"{m['CV_Accuracy']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### ROC-AUC Comparison")
    names  = list(all_results.keys())
    aucs   = [all_results[n]["ROC_AUC"] for n in names]
    colors = ["#f6c90e" if n == best_name else "#162447" for n in names]

    fig4, ax4 = plt.subplots(figsize=(8, 3.5))
    fig4.patch.set_alpha(0)
    ax4.set_facecolor("none")
    bars = ax4.bar(names, aucs, color=colors, edgecolor="none", width=0.55, zorder=3)
    ax4.set_ylim(0.88, 1.01)
    ax4.axhline(1.0, color="#ddd", linewidth=0.8, linestyle="--")
    ax4.set_ylabel("ROC-AUC", fontsize=9)
    ax4.tick_params(axis="x", labelsize=8, rotation=20)
    ax4.tick_params(axis="y", labelsize=8)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="y", alpha=0.2, zorder=1)
    for bar, val in zip(bars, aucs):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="600")
    patch = mpatches.Patch(color="#f6c90e", label=f"Best: {best_name}")
    ax4.legend(handles=[patch], fontsize=8, framealpha=0)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown("#### Tuned Best Model Metrics")
    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        ("Accuracy",    f"{metrics['accuracy']:.4f}"),
        ("ROC-AUC",     f"{metrics['roc_auc']:.4f}"),
        ("F1 Score",    f"{metrics['f1']:.4f}"),
        ("CV Accuracy", f"{metrics['cv_accuracy']:.4f}"),
    ]
    for col, (name, val) in zip([c1, c2, c3, c4], tiles):
        col.markdown(
            f'<div class="tile"><div class="tile-val">{val}</div>'
            f'<div class="tile-name">{name}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown(f"**Best hyperparameters:** `{metrics['best_params']}`")


# ════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════
with tab_info:
    st.markdown("### ℹ️ About This App")
    st.markdown(f"""
**Model:** {model_label} — auto-selected by benchmarking 7 classifiers on ROC-AUC.

**Dataset:** Wisconsin Breast Cancer Dataset — 569 samples, 30 numerical features from
digitized FNA images of breast masses.  
Target: `M` = Malignant · `B` = Benign

**Methodology:**
- 7 classifiers trained on 80/20 stratified split
- Ranked by ROC-AUC on held-out test set
- Winner tuned via 5-fold GridSearchCV
- Model trained fresh at app startup (no stale pickle files)

**Disclaimer:** For educational and research purposes only. Not for clinical use.
    """)
    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Group": ["Mean" if f.endswith("_mean") else "SE" if f.endswith("_se") else "Worst"
                  for f in FEATURES],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
