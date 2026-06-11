"""
MCP Server 2 — Heart Disease Diagnosis
Strumenti per valutazione del rischio, pattern detection e report clinici
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

DATA_PATH = Path(__file__).parent.parent / "data" / "heart.csv"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ALL_FEATURES     = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset non trovato in {DATA_PATH}.\n"
            "Scarica 'heart.csv' da https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset "
            "e posizionalo nella cartella data/"
        )
    return pd.read_csv(DATA_PATH)


app = Server("heart-diagnosis")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="risk_assessment",
            description=(
                "Valuta il rischio di malattia cardiaca per un paziente dato un insieme "
                "di parametri clinici. Usa un modello Random Forest addestrato sul dataset."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "age":      {"type": "number", "description": "Età in anni"},
                    "sex":      {"type": "integer", "enum": [0, 1], "description": "0=femmina, 1=maschio"},
                    "cp":       {"type": "integer", "enum": [0,1,2,3], "description": "Tipo dolore toracico"},
                    "trestbps": {"type": "number", "description": "Pressione arteriosa (mm Hg)"},
                    "chol":     {"type": "number", "description": "Colesterolo (mg/dl)"},
                    "fbs":      {"type": "integer", "enum": [0, 1], "description": "Glicemia a digiuno > 120 mg/dl"},
                    "restecg":  {"type": "integer", "enum": [0,1,2], "description": "ECG a riposo"},
                    "thalach":  {"type": "number", "description": "FC massima raggiunta"},
                    "exang":    {"type": "integer", "enum": [0, 1], "description": "Angina da esercizio"},
                    "oldpeak":  {"type": "number", "description": "Depressione segmento ST"},
                    "slope":    {"type": "integer", "enum": [0,1,2], "description": "Pendenza ST"},
                    "ca":       {"type": "integer", "enum": [0,1,2,3,4], "description": "Vasi maggiori colorati"},
                    "thal":     {"type": "integer", "enum": [0,1,2,3], "description": "Thalassemia"},
                },
                "required": ["age"],
            },
        ),
        Tool(
            name="pattern_detection",
            description=(
                "Identifica pattern clinici nel dataset usando clustering k-means, "
                "anomaly detection (Isolation Forest) o regole di associazione."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["clustering", "anomaly_detection", "decision_rules"],
                        "description": "Algoritmo da applicare",
                        "default": "clustering",
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Numero di cluster (solo per clustering)",
                        "default": 3,
                        "minimum": 2,
                        "maximum": 6,
                    },
                    "contamination": {
                        "type": "number",
                        "description": "Frazione anomalie attesa (solo anomaly_detection)",
                        "default": 0.05,
                    },
                    "min_support": {
                        "type": "number",
                        "description": "Supporto minimo per regole (0-1)",
                        "default": 0.3,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="generate_report",
            description=(
                "Genera un report clinico strutturato del dataset Heart Disease "
                "con statistiche, feature importance e raccomandazioni."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["executive_summary", "full_analysis", "risk_focus"],
                        "description": "Tipo di report",
                        "default": "full_analysis",
                    },
                    "include_feature_importance": {
                        "type": "boolean",
                        "description": "Includi importanza feature (Random Forest)",
                        "default": True,
                    },
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    df = load_data()

    # ── risk_assessment ───────────────────────────────────────────────────────
    if name == "risk_assessment":
        # Addestra RF sul dataset completo, poi predici sul paziente
        df_clean = df.dropna(subset=ALL_FEATURES + ["target"])
        X = df_clean[ALL_FEATURES].values
        y = df_clean["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)

        # Costruisce vettore paziente con valori medi come default
        means = df_clean[ALL_FEATURES].mean()
        patient = {f: arguments.get(f, means[f]) for f in ALL_FEATURES}
        patient_arr = np.array([[patient[f] for f in ALL_FEATURES]])
        patient_scaled = scaler.transform(patient_arr)

        proba      = rf.predict_proba(patient_scaled)[0]
        risk_score = int(round(proba[1] * 100))
        prediction = int(rf.predict(patient_scaled)[0])

        # Feature importance per questo paziente
        importances = rf.feature_importances_
        top_features = sorted(
            zip(ALL_FEATURES, importances), key=lambda x: -x[1]
        )[:5]

        # Categoria rischio
        if risk_score < 30:
            category = "BASSO"
            color    = "🟢"
        elif risk_score < 60:
            category = "MODERATO"
            color    = "🟡"
        else:
            category = "ALTO"
            color    = "🔴"

        # Percentile rispetto al dataset
        all_probas = rf.predict_proba(X_scaled)[:, 1] * 100
        percentile = int(stats.percentileofscore(all_probas, risk_score))

        lines = [
            "=== RISK ASSESSMENT ===\n",
            f"Parametri paziente forniti: {list(arguments.keys())}",
            f"(Valori mancanti completati con la media del dataset)\n",
            f"── RISULTATO",
            f"   Score di rischio : {risk_score}/100  {color}",
            f"   Categoria        : {category}",
            f"   Probabilità      : malattia {proba[1]*100:.1f}%  |  sano {proba[0]*100:.1f}%",
            f"   Predizione RF    : {'malattia presente' if prediction==1 else 'malattia assente'}",
            f"   Percentile       : top {100-percentile}% dei casi più a rischio nel dataset\n",
            f"── TOP 5 FEATURE PIÙ INFLUENTI (dataset)",
        ]
        for feat, imp in top_features:
            bar = "█" * int(imp * 40)
            lines.append(f"   {feat:10s} {bar} {imp*100:.1f}%")

        lines += [
            "\n── PARAMETRI PAZIENTE UTILIZZATI",
        ]
        for feat in ALL_FEATURES:
            val   = patient[feat]
            m_val = means[feat]
            diff  = "↑" if val > m_val * 1.1 else ("↓" if val < m_val * 0.9 else "≈")
            lines.append(f"   {feat:10s}: {val:.1f}  (media dataset: {m_val:.1f})  {diff}")

        lines += [
            "\n⚠ DISCLAIMER: Questo tool è esclusivamente per scopi didattici e di ricerca.",
            "Non costituisce diagnosi medica. Consultare un medico per valutazioni cliniche.",
        ]

        return [TextContent(type="text", text="\n".join(lines))]

    # ── pattern_detection ─────────────────────────────────────────────────────
    elif name == "pattern_detection":
        algorithm     = arguments.get("algorithm", "clustering")
        df_clean      = df.dropna(subset=NUMERIC_FEATURES)
        X             = df_clean[NUMERIC_FEATURES].values
        scaler        = StandardScaler()
        X_scaled      = scaler.fit_transform(X)

        lines = [f"=== PATTERN DETECTION — {algorithm.upper()} ===\n"]

        # ── k-means clustering ────────────────────────────────────────────────
        if algorithm == "clustering":
            k = arguments.get("n_clusters", 3)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            df_c = df_clean.copy()
            df_c["cluster"] = labels

            lines.append(f"K-Means con k={k} su feature numeriche: {NUMERIC_FEATURES}\n")
            for c in range(k):
                sub   = df_c[df_c["cluster"] == c]
                n     = len(sub)
                pos   = sub["target"].mean() * 100 if "target" in sub.columns else float("nan")
                means = sub[NUMERIC_FEATURES].mean()
                lines.append(f"── Cluster {c}  (n={n}, {n/len(df_c)*100:.1f}%)")
                lines.append(f"   Prevalenza malattia: {pos:.1f}%")
                for feat in NUMERIC_FEATURES:
                    lines.append(f"   {feat:10s}: {means[feat]:.2f}")
                lines.append("")

        # ── anomaly detection ─────────────────────────────────────────────────
        elif algorithm == "anomaly_detection":
            cont = arguments.get("contamination", 0.05)
            iso  = IsolationForest(contamination=cont, random_state=42)
            preds = iso.fit_predict(X_scaled)
            df_c  = df_clean.copy()
            df_c["anomaly"] = (preds == -1).astype(int)

            n_anomalies = df_c["anomaly"].sum()
            lines.append(
                f"Isolation Forest  contamination={cont}\n"
                f"Anomalie rilevate: {n_anomalies} / {len(df_c)} "
                f"({n_anomalies/len(df_c)*100:.1f}%)\n"
            )

            sub_anom   = df_c[df_c["anomaly"] == 1]
            sub_normal = df_c[df_c["anomaly"] == 0]

            if "target" in df_c.columns:
                lines.append(f"Prevalenza malattia negli anomali  : {sub_anom['target'].mean()*100:.1f}%")
                lines.append(f"Prevalenza malattia nei normali    : {sub_normal['target'].mean()*100:.1f}%\n")

            lines.append("── Caratteristiche medie anomali vs normali")
            lines.append(f"{'Feature':12s}  {'Anomali':>10s}  {'Normali':>10s}  {'Diff%':>8s}")
            lines.append("─" * 48)
            for feat in NUMERIC_FEATURES:
                a  = sub_anom[feat].mean()
                n  = sub_normal[feat].mean()
                dp = (a - n) / n * 100 if n != 0 else 0
                lines.append(f"{feat:12s}  {a:>10.2f}  {n:>10.2f}  {dp:>+7.1f}%")

        # ── decision rules ────────────────────────────────────────────────────
        elif algorithm == "decision_rules":
            min_sup = arguments.get("min_support", 0.3)
            df_b    = df.dropna(subset=ALL_FEATURES + ["target"]).copy()

            # Binarizza le feature numeriche in alto/basso rispetto alla mediana
            rules_found = []
            for feat in ALL_FEATURES:
                median = df_b[feat].median()
                high   = df_b[df_b[feat] > median]
                low    = df_b[df_b[feat] <= median]
                for subset, direction in [(high, "alto"), (low, "basso")]:
                    if len(subset) == 0: continue
                    support    = len(subset) / len(df_b)
                    confidence = subset["target"].mean()
                    lift       = confidence / df_b["target"].mean()
                    if support >= min_sup:
                        rules_found.append((feat, direction, support, confidence, lift, len(subset)))

            rules_found.sort(key=lambda x: -x[4])  # ordina per lift
            lines.append(f"Regole con support >= {min_sup}  (ordinate per lift)\n")
            lines.append(f"{'Feature':12s} {'Dir':6s} {'Support':>8s} {'Conf':>6s} {'Lift':>6s}  Regola")
            lines.append("─" * 72)
            for feat, direction, support, conf, lift, n in rules_found[:15]:
                rule = f"{feat} {direction} → malattia {'presente' if conf>0.5 else 'assente'}"
                lines.append(f"{feat:12s} {direction:6s} {support*100:>7.1f}%  {conf*100:>5.1f}%  {lift:>5.2f}x  {rule}")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── generate_report ───────────────────────────────────────────────────────
    elif name == "generate_report":
        fmt              = arguments.get("format", "full_analysis")
        incl_importance  = arguments.get("include_feature_importance", True)
        df_clean         = df.dropna(subset=ALL_FEATURES + ["target"])
        n                = len(df_clean)
        n_pos            = int((df_clean["target"] == 1).sum())
        n_neg            = int((df_clean["target"] == 0).sum())
        prevalence       = n_pos / n * 100

        lines = [
            "=" * 60,
            "  REPORT CLINICO — HEART DISEASE DATASET (UCI)",
            "=" * 60,
            f"\nDataset  : UCI Heart Disease",
            f"Pazienti : {n}  |  Malati: {n_pos} ({prevalence:.1f}%)  |  Sani: {n_neg} ({100-prevalence:.1f}%)",
            f"Feature  : {len(ALL_FEATURES)}",
            f"Formato  : {fmt}\n",
        ]

        if fmt in ("executive_summary", "full_analysis"):
            lines += [
                "── 1. PANORAMICA EPIDEMIOLOGICA",
                f"   Prevalenza malattia cardiaca: {prevalence:.1f}%",
                f"   Età media (malati): {df_clean[df_clean['target']==1]['age'].mean():.1f} anni",
                f"   Età media (sani)  : {df_clean[df_clean['target']==0]['age'].mean():.1f} anni",
                f"   % maschi nel dataset: {df_clean['sex'].mean()*100:.1f}%\n",
            ]

        if fmt == "full_analysis":
            lines += ["── 2. STATISTICHE CHIAVE PER GRUPPO TARGET\n"]
            for col in ["age", "chol", "trestbps", "thalach", "oldpeak"]:
                m0 = df_clean[df_clean["target"]==0][col].mean()
                m1 = df_clean[df_clean["target"]==1][col].mean()
                t, p = stats.ttest_ind(
                    df_clean[df_clean["target"]==0][col].dropna(),
                    df_clean[df_clean["target"]==1][col].dropna()
                )
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                lines.append(f"   {col:10s}  sani={m0:.2f}  malati={m1:.2f}  p={p:.4f} {sig}")
            lines.append("")

        if incl_importance:
            X = df_clean[ALL_FEATURES].values
            y = df_clean["target"].values
            scaler = StandardScaler()
            X_s    = scaler.fit_transform(X)
            rf     = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_s, y)
            importances = sorted(
                zip(ALL_FEATURES, rf.feature_importances_), key=lambda x: -x[1]
            )

            lines += [f"\n── {'3' if fmt=='full_analysis' else '2'}. FEATURE IMPORTANCE (Random Forest 200 alberi)"]
            for feat, imp in importances:
                bar = "█" * int(imp * 50)
                lines.append(f"   {feat:10s}  {bar}  {imp*100:.2f}%")
            lines.append("")

        if fmt in ("risk_focus", "full_analysis"):
            high_risk = df_clean[
                (df_clean["cp"] == 0) &
                (df_clean["exang"] == 1) &
                (df_clean["oldpeak"] > 2)
            ]
            lines += [
                f"\n── PROFILO ALTO RISCHIO",
                f"   Criteri: cp=0 (tipico) + angina da esercizio + oldpeak > 2",
                f"   Pazienti: {len(high_risk)} ({len(high_risk)/n*100:.1f}% del dataset)",
                f"   Prevalenza malattia: {high_risk['target'].mean()*100:.1f}%",
                f"   Età media: {high_risk['age'].mean():.1f} anni\n",
            ]

        lines += [
            "── CONCLUSIONI",
            "   Le feature più predittive sono ca (vasi colorati), thal, cp e oldpeak.",
            "   I pazienti con angina tipica, alta depressione ST e vasi ostruiti",
            "   mostrano la maggiore prevalenza di malattia cardiaca.",
            "\n── LIMITAZIONI",
            "   Dataset di piccole dimensioni (303 pazienti), single-center.",
            "   Non rappresentativo di tutte le popolazioni.",
            "   Bias di selezione: pazienti già a rischio cardiovascolare.",
            "\n⚠ DISCLAIMER: Esclusivamente per scopi didattici e di ricerca.",
        ]

        return [TextContent(type="text", text="\n".join(lines))]

    return [TextContent(type="text", text=f"Tool '{name}' non riconosciuto.")]


async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())