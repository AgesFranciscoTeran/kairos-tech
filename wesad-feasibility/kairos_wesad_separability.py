"""
KAIROS — Análisis de viabilidad sobre datos públicos (WESAD)
============================================================

Pregunta de investigación
--------------------------
La incertidumbre técnica central de Kairos es la ESPECIFICIDAD:
¿se puede separar una activación fisiológica genuina (estrés sostenido) de una
activación benigna, usando solo señales de muñeca (PPG/HRV, EDA, temperatura, IMU),
con una tasa de falsas alarmas suficientemente baja?

Este script da una PRIMERA medición directa de esa incertidumbre ANTES de recolectar
datos propios, corriendo el pipeline de extracción de características de Kairos sobre
WESAD (Schmidt et al., 2018), un dataset público con exactamente las mismas modalidades.

Por qué WESAD y por qué la muñeca (Empatica E4)
-----------------------------------------------
WESAD trae dos dispositivos: RespiBAN (pecho) y Empatica E4 (muñeca). Usamos SOLO las
señales de muñeca del E4 porque su form factor coincide con Kairos:
    BVP (64 Hz)  -> HRV            (análogo a PPG de Kairos)
    EDA (4 Hz)   -> SCL/SCR
    TEMP (4 Hz)  -> temperatura periférica
    ACC (32 Hz)  -> movimiento     (el "IMU gating" que descarta el confusor de ejercicio)

Etiquetas WESAD (a 700 Hz): 0=transitorio, 1=baseline, 2=stress, 3=amusement,
4=meditation, 5-7=ignorar. Aquí comparamos baseline (1) vs stress (2).

Las dos comparaciones que importan
----------------------------------
1) Solo fisiología  vs  fisiología + ACC   -> ¿el gating por IMU ayuda?
2) Normalización global  vs  baseline personal por sujeto
   -> la hipótesis de Kairos: la "normalidad" fisiológica es idiosincrática, así que
      normalizar contra el baseline de cada usuario debería mejorar la separación.

Se evalúa con Leave-One-Subject-Out (LOSO), reportando ROC-AUC y —lo más importante
para Kairos— la ESPECIFICIDAD (1 - tasa de falsas alarmas) a una sensibilidad fija.

Uso
---
    # 1. Descargar WESAD (~2.1 GB): https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
    # 2. Descomprimir de modo que quede: WESAD/S2/S2.pkl, WESAD/S3/S3.pkl, ...
    pip install -r requirements.txt
    python kairos_wesad_separability.py --data-dir ./WESAD --out-dir ./out

Salidas en --out-dir:
    figures/fig_distributions.png     distribuciones baseline vs stress (crudo y normalizado)
    figures/fig_roc.png               curvas ROC de las 4 condiciones (LOSO)
    figures/fig_specificity.png       especificidad @ sensibilidad fija
    results/feature_table.csv         tabla de features por ventana
    results/metrics.json              todas las métricas

NOTA DE HONESTIDAD: este script NO trae resultados incluidos. Los números salen de
correrlo sobre WESAD real. Es un laboratorio para medir, no una afirmación de desempeño.

Referencia: Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K.
(2018). Introducing WESAD, a multimodal dataset for wearable stress and affect
detection. ICMI 2018.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- #
# Configuración
# ----------------------------------------------------------------------------- #
FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}   # Hz, señales de muñeca (E4)
LABEL_FS = 700                                      # Hz, señal de etiquetas WESAD
WIN_SEC = 60                                        # ventana de análisis (diseño Kairos: 30-120 s)
STEP_SEC = 30                                       # solapamiento del 50%
LABELS = {1: "baseline", 2: "stress"}              # comparación central
PURITY = 0.90                                       # % mínimo de una sola etiqueta en la ventana
MIN_PEAKS = 6                                       # mínimo de picos BVP para calcular HRV
SENS_TARGET = 0.80                                  # sensibilidad objetivo para reportar especificidad

# Bloques de features (el orden define las columnas)
PHYS_FEATURES = [
    "hr_mean", "sdnn", "rmssd", "pnn50",            # HRV (desde BVP)
    "eda_scl_mean", "eda_scl_slope", "eda_std",     # EDA tónica
    "eda_scr_count", "eda_scr_amp",                 # EDA fásica
    "temp_mean", "temp_slope",                      # temperatura
]
ACC_FEATURES = ["acc_move_std", "acc_move_mad"]     # movimiento (IMU gating)
ALL_FEATURES = PHYS_FEATURES + ACC_FEATURES


# ----------------------------------------------------------------------------- #
# Extracción de características (una ventana = un vector de features)
# ----------------------------------------------------------------------------- #
def _butter_band(sig, fs, lo, hi, order=3):
    nyq = 0.5 * fs
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, sig)


def _slope(sig):
    """Pendiente lineal (por segundo aproximado) de una señal corta."""
    if len(sig) < 2:
        return 0.0
    x = np.arange(len(sig))
    return float(np.polyfit(x, sig, 1)[0])


def hrv_features(bvp, fs=64):
    """HRV desde BVP de muñeca: filtra, detecta picos sistólicos, calcula IBIs."""
    out = {"hr_mean": np.nan, "sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan}
    if len(bvp) < fs * 5:
        return out
    filt = _butter_band(np.asarray(bvp, float).ravel(), fs, 0.7, 3.5)
    dist = max(1, int(fs * 0.33))                  # <= 180 bpm
    prom = 0.3 * np.std(filt)
    peaks, _ = sps.find_peaks(filt, distance=dist, prominence=prom)
    if len(peaks) < MIN_PEAKS:
        return out
    ibi = np.diff(peaks) / fs * 1000.0             # ms
    ibi = ibi[(ibi > 300) & (ibi < 2000)]          # descarta artefactos (30-200 bpm)
    if len(ibi) < MIN_PEAKS - 1:
        return out
    diff = np.diff(ibi)
    out["hr_mean"] = float(60000.0 / np.mean(ibi))
    out["sdnn"] = float(np.std(ibi))
    out["rmssd"] = float(np.sqrt(np.mean(diff ** 2)))
    out["pnn50"] = float(np.mean(np.abs(diff) > 50) * 100.0)
    return out


def eda_features(eda, fs=4):
    """EDA de muñeca: nivel tónico (SCL) + respuestas fásicas (SCR)."""
    eda = np.asarray(eda, float).ravel()
    out = {
        "eda_scl_mean": float(np.mean(eda)),
        "eda_scl_slope": _slope(eda),
        "eda_std": float(np.std(eda)),
        "eda_scr_count": 0.0,
        "eda_scr_amp": 0.0,
    }
    if len(eda) >= fs * 5:
        nyq = 0.5 * fs
        b, a = sps.butter(2, 0.05 / nyq, btype="high")  # separa lo fásico
        phasic = sps.filtfilt(b, a, eda)
        pk, props = sps.find_peaks(phasic, height=0.01 * (np.std(eda) + 1e-9),
                                   distance=max(1, int(fs * 1.0)))
        out["eda_scr_count"] = float(len(pk))
        if len(pk):
            out["eda_scr_amp"] = float(np.mean(props["peak_heights"]))
    return out


def temp_features(temp, fs=4):
    temp = np.asarray(temp, float).ravel()
    return {"temp_mean": float(np.mean(temp)), "temp_slope": _slope(temp)}


def acc_features(acc, fs=32):
    """Movimiento a partir de la magnitud del acelerómetro (unidades-agnóstico).
    Esta es la señal que Kairos usa para descartar el confusor de ejercicio."""
    acc = np.asarray(acc, float)
    mag = np.linalg.norm(acc, axis=1) if acc.ndim == 2 else np.abs(acc.ravel())
    med = np.median(mag)
    return {
        "acc_move_std": float(np.std(mag)),
        "acc_move_mad": float(np.mean(np.abs(mag - med))),
    }


def window_features(wrist, i0, i1_sec):
    """Extrae todas las features de una ventana [i0, i0+WIN_SEC) segundos."""
    feats = {}

    def sl(name):
        f = FS[name]
        a = int(i0 * f)
        b = int((i0 + WIN_SEC) * f)
        return np.asarray(wrist[name])[a:b]

    feats.update(hrv_features(sl("BVP"), FS["BVP"]))
    feats.update(eda_features(sl("EDA"), FS["EDA"]))
    feats.update(temp_features(sl("TEMP"), FS["TEMP"]))
    feats.update(acc_features(sl("ACC"), FS["ACC"]))
    return feats


# ----------------------------------------------------------------------------- #
# Carga y construcción de la tabla de features
# ----------------------------------------------------------------------------- #
def load_subject(pkl_path):
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")
    return data["signal"]["wrist"], np.asarray(data["label"]).ravel(), str(data["subject"])


def window_label(labels, t0):
    """Etiqueta de la ventana por voto mayoritario, con chequeo de pureza."""
    seg = labels[int(t0 * LABEL_FS):int((t0 + WIN_SEC) * LABEL_FS)]
    if len(seg) == 0:
        return None
    vals, counts = np.unique(seg, return_counts=True)
    top = vals[np.argmax(counts)]
    purity = counts.max() / counts.sum()
    if purity < PURITY or int(top) not in LABELS:
        return None
    return int(top)


def build_feature_table(data_dir):
    rows = []
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "S*.pkl"), recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No se encontraron .pkl de WESAD en {data_dir!r}. "
            "Esperado: {data_dir}/S2/S2.pkl, etc.")
    for p in paths:
        wrist, labels, subject = load_subject(p)
        dur = len(labels) / LABEL_FS
        n_win = 0
        for t0 in np.arange(0, dur - WIN_SEC, STEP_SEC):
            lab = window_label(labels, t0)
            if lab is None:
                continue
            feats = window_features(wrist, t0, WIN_SEC)
            feats["subject"] = subject
            feats["label"] = lab
            rows.append(feats)
            n_win += 1
        print(f"  {subject}: {n_win} ventanas válidas")
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.dropna(subset=ALL_FEATURES).reset_index(drop=True)
    print(f"Ventanas totales: {before} | tras descartar HRV inválida: {len(df)}")
    return df


# ----------------------------------------------------------------------------- #
# Normalización
# ----------------------------------------------------------------------------- #
def normalize_global(train_df, test_df, cols):
    """z-score con media/desv del set de entrenamiento (normalización 'poblacional')."""
    mu = train_df[cols].mean()
    sd = train_df[cols].std().replace(0, 1.0)
    return (train_df[cols] - mu) / sd, (test_df[cols] - mu) / sd


def normalize_personal(df, cols):
    """z-score de cada sujeto contra SU PROPIO baseline (label==1).
    Esto es la 'línea base personal' de Kairos: no usa las ventanas de estrés
    para normalizar, solo el reposo de cada usuario."""
    out = df.copy()
    for subj, idx in df.groupby("subject").groups.items():
        base = df.loc[idx]
        base = base[base["label"] == 1]
        ref = df.loc[idx]
        if len(base) >= 2:
            mu, sd = base[cols].mean(), base[cols].std().replace(0, 1.0)
        else:  # fallback si el sujeto casi no tiene baseline
            mu, sd = ref[cols].mean(), ref[cols].std().replace(0, 1.0)
        out.loc[idx, cols] = (df.loc[idx, cols] - mu) / sd
    return out


# ----------------------------------------------------------------------------- #
# Evaluación LOSO
# ----------------------------------------------------------------------------- #
def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-12)


def loso_predictions(df, cols, personal):
    """Predicciones out-of-fold con Leave-One-Subject-Out.
    Devuelve (y_true, y_score) concatenados sobre todos los sujetos."""
    y_true, y_score = [], []
    work = normalize_personal(df, cols) if personal else df
    subjects = work["subject"].unique()
    for test_subj in subjects:
        tr = work[work["subject"] != test_subj]
        te = work[work["subject"] == test_subj]
        if te["label"].nunique() < 2:      # el sujeto de test necesita ambas clases
            continue
        if personal:
            Xtr, Xte = tr[cols].values, te[cols].values          # ya normalizado por sujeto
        else:
            Xtr_df, Xte_df = normalize_global(tr, te, cols)
            Xtr, Xte = Xtr_df.values, Xte_df.values
        ytr = (tr["label"].values == 2).astype(int)
        yte = (te["label"].values == 2).astype(int)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        y_true.extend(yte.tolist())
        y_score.extend(proba.tolist())
    return np.asarray(y_true), np.asarray(y_score)


def specificity_at_sensitivity(y_true, y_score, target=SENS_TARGET):
    """Especificidad (1 - tasa de falsas alarmas) en el umbral que alcanza la
    sensibilidad objetivo. Traduce directo la métrica de Kairos: a X% de detección,
    ¿qué fracción de reposo se marca como falsa alarma?"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target)
    idx = min(idx, len(tpr) - 1)
    fa = float(fpr[idx])                 # tasa de falsas alarmas
    return {"sensitivity": float(tpr[idx]), "specificity": 1 - fa, "false_alarm_rate": fa}


def evaluate(df):
    conditions = {
        "phys_global":   (PHYS_FEATURES, False),
        "phys_personal": (PHYS_FEATURES, True),
        "physacc_global":   (ALL_FEATURES, False),
        "physacc_personal": (ALL_FEATURES, True),
    }
    results = {}
    roc_data = {}
    for name, (cols, personal) in conditions.items():
        yt, ys = loso_predictions(df, cols, personal)
        if len(np.unique(yt)) < 2:
            results[name] = {"auc": None, "note": "clases insuficientes"}
            continue
        auc = float(roc_auc_score(yt, ys))
        spec = specificity_at_sensitivity(yt, ys)
        results[name] = {"auc": auc, **spec, "n_windows": int(len(yt))}
        fpr, tpr, _ = roc_curve(yt, ys)
        roc_data[name] = (fpr, tpr, auc)
        print(f"  {name:18s} AUC={auc:.3f}  "
              f"espec@{int(SENS_TARGET*100)}%sens={spec['specificity']:.3f}  "
              f"(falsas alarmas={spec['false_alarm_rate']:.3f})")
    return results, roc_data


# ----------------------------------------------------------------------------- #
# Figuras
# ----------------------------------------------------------------------------- #
JADE, GOLD, ROSE, INK = "#4E7A5B", "#C9A24B", "#B03A48", "#1c1c1c"


def fig_distributions(df, out):
    keep = ["rmssd", "eda_scl_mean", "hr_mean", "acc_move_std"]
    dfn = normalize_personal(df, ALL_FEATURES)
    fig, axes = plt.subplots(2, len(keep), figsize=(4 * len(keep), 7))
    for j, feat in enumerate(keep):
        for row, (frame, tag) in enumerate([(df, "crudo"), (dfn, "baseline personal")]):
            ax = axes[row, j]
            b = frame[frame["label"] == 1][feat]
            s = frame[frame["label"] == 2][feat]
            ax.hist(b, bins=30, alpha=0.6, color=JADE, label="baseline", density=True)
            ax.hist(s, bins=30, alpha=0.6, color=ROSE, label="stress", density=True)
            d = cohens_d(s.dropna().values, b.dropna().values)
            ax.set_title(f"{feat} ({tag})\nCohen's d = {d:.2f}", fontsize=9)
            if row == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Separabilidad por feature: baseline vs stress", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out, "fig_distributions.png"), dpi=130)
    plt.close(fig)


def fig_roc(roc_data, out):
    fig, ax = plt.subplots(figsize=(7, 6))
    styles = {
        "phys_global": (ROSE, "--"), "phys_personal": (ROSE, "-"),
        "physacc_global": (JADE, "--"), "physacc_personal": (JADE, "-"),
    }
    labels = {
        "phys_global": "solo fisiología · global",
        "phys_personal": "solo fisiología · baseline personal",
        "physacc_global": "fisiología + IMU · global",
        "physacc_personal": "fisiología + IMU · baseline personal",
    }
    for name, (fpr, tpr, auc) in roc_data.items():
        c, ls = styles[name]
        ax.plot(fpr, tpr, color=c, linestyle=ls, lw=2,
                label=f"{labels[name]} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.axhline(SENS_TARGET, color=GOLD, lw=1, linestyle="-.",
               label=f"sensibilidad objetivo = {SENS_TARGET:.0%}")
    ax.set_xlabel("Tasa de falsas alarmas (1 - especificidad)")
    ax.set_ylabel("Sensibilidad (detección de estrés real)")
    ax.set_title("ROC (Leave-One-Subject-Out)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_roc.png"), dpi=130)
    plt.close(fig)


def fig_specificity(results, out):
    order = ["phys_global", "phys_personal", "physacc_global", "physacc_personal"]
    names = ["fisio\nglobal", "fisio\npersonal", "fisio+IMU\nglobal", "fisio+IMU\npersonal"]
    vals = [results[k].get("specificity", 0) or 0 for k in order]
    colors = [ROSE, ROSE, JADE, JADE]
    alphas = [0.55, 1.0, 0.55, 1.0]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, vals, color=colors)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Especificidad @ {SENS_TARGET:.0%} sensibilidad")
    ax.set_title("Menos falsas alarmas es mejor (barra más alta = mejor)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_specificity.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Kairos — viabilidad de separabilidad sobre WESAD")
    ap.add_argument("--data-dir", default="WESAD", help="carpeta con S2/S2.pkl, S3/S3.pkl, ...")
    ap.add_argument("--out-dir", default="out", help="carpeta de salida")
    args = ap.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    print("[1/4] Construyendo tabla de features desde WESAD...")
    df = build_feature_table(args.data_dir)
    df.to_csv(os.path.join(res_dir, "feature_table.csv"), index=False)

    print("\n[2/4] Resumen del dataset")
    summary = df.groupby("label").size().rename(index=LABELS).to_dict()
    print(f"  ventanas por clase: {summary}")
    print(f"  sujetos: {df['subject'].nunique()}")

    print("\n[3/4] Evaluación LOSO (4 condiciones)")
    results, roc_data = evaluate(df)

    print("\n[4/4] Generando figuras...")
    fig_distributions(df, fig_dir)
    if roc_data:
        fig_roc(roc_data, fig_dir)
    fig_specificity(results, fig_dir)

    metrics = {
        "config": {"win_sec": WIN_SEC, "step_sec": STEP_SEC, "purity": PURITY,
                   "sens_target": SENS_TARGET, "labels": LABELS},
        "dataset": {"subjects": int(df["subject"].nunique()),
                    "windows_per_class": {LABELS[k]: int(v)
                                          for k, v in df.groupby("label").size().items()}},
        "results": results,
    }
    with open(os.path.join(res_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(f"\nListo. Revisa {fig_dir}/ y {res_dir}/metrics.json")


if __name__ == "__main__":
    main()
