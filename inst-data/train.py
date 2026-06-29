"""
train.py — Entraînement du modèle AgriSat
Lancer : python train.py
"""
import pandas as pd
import numpy as np
import pickle, json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, r2_score, mean_squared_error

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

print("==============================================")
print("  AgriSat — Entraînement du modèle IA")
print("==============================================")

# ── 1. Charger le CSV ─────────────────────────────
csv_files = list(Path(".").glob("*.csv"))
if not csv_files:
    print("ERREUR : Aucun fichier CSV trouvé !")
    print("Mettez Crop_recommendationV2.csv dans ce dossier.")
    exit(1)

df = pd.read_csv(csv_files[0])
print(f"\n✅ Fichier charge : {csv_files[0].name}")
print(f"   {len(df)} lignes x {df.shape[1]} colonnes")

# ── 2. Préparer les données ───────────────────────
le = LabelEncoder()
df["culture_enc"] = le.fit_transform(df["label"])

wue = df["water_usage_efficiency"]
df["stress"] = 1 - (wue - wue.min()) / (wue.max() - wue.min() + 1e-9)
df["rendement"] = (wue * 0.3 + df["N"] * 0.05 + df["soil_moisture"] * 0.02).clip(0, 100)
df["classe"] = pd.cut(df["stress"],
    bins=[-0.001, 0.25, 0.5, 0.75, 1.001],
    labels=["faible", "modere", "eleve", "severe"]).astype(str)

FEATURES = [c for c in [
    "N","P","K","temperature","humidity","rainfall","ph",
    "soil_moisture","soil_type","wind_speed","sunlight_exposure",
    "organic_matter","pest_pressure","fertilizer_usage",
    "irrigation_frequency","crop_density"
] if c in df.columns]

X = df[FEATURES].fillna(0)
X_tr, X_te, yc_tr, yc_te, ys_tr, ys_te, yr_tr, yr_te, ycl_tr, ycl_te = \
    train_test_split(X, df["culture_enc"], df["stress"],
                     df["rendement"], df["classe"],
                     test_size=0.2, random_state=42)

# ── 3. Entraîner ──────────────────────────────────
print("\n⏳ Entraînement en cours (1-2 minutes)...")

clf_culture = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(X_tr, yc_tr)
clf_stress  = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(X_tr, ycl_tr)
reg_stress  = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1).fit(X_tr, ys_tr)
reg_rdt     = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1).fit(X_tr, yr_tr)

f1_c  = f1_score(yc_te,  clf_culture.predict(X_te), average="weighted")
f1_s  = f1_score(ycl_te, clf_stress.predict(X_te),  average="weighted")
r2_s  = r2_score(ys_te,  reg_stress.predict(X_te))
r2_r  = r2_score(yr_te,  reg_rdt.predict(X_te))
rmse_r = np.sqrt(mean_squared_error(yr_te, reg_rdt.predict(X_te)))

# ── 4. Sauvegarder ────────────────────────────────
bundle = {
    "clf_culture":   clf_culture,
    "clf_stress":    clf_stress,
    "reg_stress":    reg_stress,
    "reg_rendement": reg_rdt,
    "le_culture":    le,
    "features":      FEATURES,
    "cultures":      list(le.classes_),
    "metrics": {
        "f1_culture":    round(f1_c, 3),
        "f1_stress":     round(f1_s, 3),
        "r2_stress":     round(r2_s, 3),
        "r2_rendement":  round(r2_r, 3),
        "rmse_rendement":round(rmse_r, 2),
    }
}
with open("models/modele_agrisat.pkl", "wb") as f:
    pickle.dump(bundle, f)
with open("results/metriques.json", "w") as f:
    json.dump(bundle["metrics"], f, indent=2)

print("\n==============================================")
print("  RESULTATS")
print("==============================================")
print(f"  Culture    F1  : {f1_c:.3f}  {'✅ Excellent' if f1_c>0.9 else '⚠️ Moyen'}")
print(f"  Stress     F1  : {f1_s:.3f}")
print(f"  Stress     R²  : {r2_s:.3f}")
print(f"  Rendement  R²  : {r2_r:.3f}  {'✅ Excellent' if r2_r>0.9 else '⚠️ Moyen'}")
print(f"\n✅ Modele sauvegarde : models/modele_agrisat.pkl")
print(f"\n➡️  Maintenant lancez : python app.py")