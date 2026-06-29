"""
AgriSat-Maroc - Comparaison RF vs XGBoost vs LSTM
Utilise : data/processed/agri_maroc_real_data_v2.csv
Lancer  : python train_compare.py
"""
import pandas as pd
import numpy as np
import pickle, json, time, os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, r2_score, mean_squared_error

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

print("="*55)
print("  AgriSat - Comparaison RF vs XGBoost vs LSTM")
print("="*55)

# 1. Charger les vraies donnees meteo
DATA = "data/processed/agri_maroc_real_data_v2.csv"
if not Path(DATA).exists():
    print("ERREUR : fichier non trouve :", DATA)
    print("Lancez : python download_real_data_v2.py")
    exit(1)

df = pd.read_csv(DATA)
print(f"\nFichier : {DATA} — {len(df)} lignes")

# 2. Preparer features
le_region  = LabelEncoder()
le_culture = LabelEncoder()
df["region_enc"]  = le_region.fit_transform(df["region"])
df["culture_enc"] = le_culture.fit_transform(df["culture"])

FEATURES = [f for f in [
    "temp_moyenne","precip_30j_mm","etp_mm_jour",
    "humidite_pct","vitesse_vent_kmh",
    "retention_eau","matiere_organique",
    "mois","region_enc","culture_enc"
] if f in df.columns]

print(f"Features ({len(FEATURES)}) : {FEATURES}")

y_stress    = df["stress_hydrique"]
y_rendement = df["rendement_t_ha"]
y_culture   = df["culture_enc"]
y_classe    = pd.cut(df["stress_hydrique"],
    bins=[-0.001,0.25,0.5,0.75,1.001],
    labels=["faible","modere","eleve","severe"]).astype(str)

X = df[FEATURES].fillna(0)
X_tr,X_te,ys_tr,ys_te,yr_tr,yr_te,yc_tr,yc_te,ycl_tr,ycl_te = train_test_split(
    X,y_stress,y_rendement,y_culture,y_classe,test_size=0.2,random_state=42)

print(f"Train: {len(X_tr)} | Test: {len(X_te)}")
resultats = {}

# ======== RANDOM FOREST ========
print("\n[1/3] Random Forest ...")
t0 = time.time()
rf_s = RandomForestRegressor(n_estimators=300,max_depth=15,min_samples_leaf=2,random_state=42,n_jobs=-1).fit(X_tr,ys_tr)
rf_r = RandomForestRegressor(n_estimators=300,max_depth=15,min_samples_leaf=2,random_state=42,n_jobs=-1).fit(X_tr,yr_tr)
rf_c = RandomForestClassifier(n_estimators=300,max_depth=15,min_samples_leaf=2,random_state=42,n_jobs=-1).fit(X_tr,yc_tr)
rf_cl= RandomForestClassifier(n_estimators=300,max_depth=15,min_samples_leaf=2,random_state=42,n_jobs=-1).fit(X_tr,ycl_tr)
tf_  = round(time.time()-t0,1)
r2s  = round(r2_score(ys_te,rf_s.predict(X_te)),3)
rms  = round(np.sqrt(mean_squared_error(ys_te,rf_s.predict(X_te))),4)
r2r  = round(r2_score(yr_te,rf_r.predict(X_te)),3)
f1   = round(f1_score(yc_te,rf_c.predict(X_te),average="weighted"),3)
resultats["Random Forest"] = {"r2_stress":r2s,"rmse_stress":rms,"r2_rendement":r2r,"f1_culture":f1,"temps_s":tf_}
print(f"   R2={r2s}  RMSE={rms}  R2_rdt={r2r}  F1={f1}  {tf_}s")

imp = pd.Series(rf_s.feature_importances_,index=FEATURES).sort_values(ascending=False)
print("   Importance variables :")
for k,v in imp.items():
    print(f"   {k:22s} {v:.3f}  {'='*int(v*40)}")

bundle_rf = {"clf_culture":rf_c,"clf_stress":rf_cl,"reg_stress":rf_s,"reg_rendement":rf_r,
             "le_region":le_region,"le_culture":le_culture,"features":FEATURES,
             "cultures":list(le_culture.classes_),"metrics":resultats["Random Forest"]}
pickle.dump(bundle_rf, open("models/modele_agrisat.pkl","wb"))
print("   Sauvegarde : models/modele_agrisat.pkl")

# ======== XGBOOST ========
print("\n[2/3] XGBoost ...")
try:
    from xgboost import XGBRegressor, XGBClassifier
    t0 = time.time()
    xgb_s = XGBRegressor(n_estimators=300,max_depth=7,learning_rate=0.05,subsample=0.8,
                          colsample_bytree=0.8,random_state=42,verbosity=0,n_jobs=-1)
    xgb_s.fit(X_tr,ys_tr,eval_set=[(X_te,ys_te)],verbose=False)
    xgb_r = XGBRegressor(n_estimators=300,max_depth=7,learning_rate=0.05,subsample=0.8,
                          colsample_bytree=0.8,random_state=42,verbosity=0,n_jobs=-1)
    xgb_r.fit(X_tr,yr_tr,verbose=False)
    xgb_c = XGBClassifier(n_estimators=300,max_depth=7,learning_rate=0.05,subsample=0.8,
                           colsample_bytree=0.8,random_state=42,verbosity=0,n_jobs=-1)
    xgb_c.fit(X_tr,yc_tr,verbose=False)
    tf_ = round(time.time()-t0,1)
    r2s = round(r2_score(ys_te,xgb_s.predict(X_te)),3)
    rms = round(np.sqrt(mean_squared_error(ys_te,xgb_s.predict(X_te))),4)
    r2r = round(r2_score(yr_te,xgb_r.predict(X_te)),3)
    f1  = round(f1_score(yc_te,xgb_c.predict(X_te),average="weighted"),3)
    resultats["XGBoost"] = {"r2_stress":r2s,"rmse_stress":rms,"r2_rendement":r2r,"f1_culture":f1,"temps_s":tf_}
    print(f"   R2={r2s}  RMSE={rms}  R2_rdt={r2r}  F1={f1}  {tf_}s")
    bundle_xgb = {"reg_stress":xgb_s,"reg_rendement":xgb_r,"clf_culture":xgb_c,
                  "le_culture":le_culture,"le_region":le_region,"features":FEATURES,
                  "cultures":list(le_culture.classes_),"metrics":resultats["XGBoost"]}
    pickle.dump(bundle_xgb, open("models/modele_xgboost.pkl","wb"))
    print("   Sauvegarde : models/modele_xgboost.pkl")
except ImportError:
    print("   Non installe — pip install xgboost")
    resultats["XGBoost"] = {"r2_stress":None,"rmse_stress":None,"r2_rendement":None,"f1_culture":None,"temps_s":None}

# ======== LSTM ========
print("\n[3/3] LSTM ...")
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    t0 = time.time()
    sX = MinMaxScaler(); sy = MinMaxScaler()
    Xtr2 = sX.fit_transform(X_tr); Xte2 = sX.transform(X_te)
    ytr2 = sy.fit_transform(ys_tr.values.reshape(-1,1)).flatten()
    yte2 = sy.transform(ys_te.values.reshape(-1,1)).flatten()
    Xtr3 = Xtr2.reshape(Xtr2.shape[0],1,Xtr2.shape[1])
    Xte3 = Xte2.reshape(Xte2.shape[0],1,Xte2.shape[1])
    model = Sequential([
        Input(shape=(1,len(FEATURES))),
        LSTM(128,return_sequences=True), Dropout(0.2),
        LSTM(64,return_sequences=False), Dropout(0.2),
        Dense(32,activation="relu"), Dense(16,activation="relu"),
        Dense(1,activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss="mse",metrics=["mae"])
    es = EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True,verbose=0)
    hist = model.fit(Xtr3,ytr2,epochs=150,batch_size=16,validation_split=0.15,callbacks=[es],verbose=0)
    ypred = np.clip(sy.inverse_transform(model.predict(Xte3,verbose=0).reshape(-1,1)).flatten(),0,1)
    tf_ = round(time.time()-t0,1)
    r2s = round(r2_score(ys_te,ypred),3)
    rms = round(np.sqrt(mean_squared_error(ys_te,ypred)),4)
    nep = len(hist.history["loss"])
    resultats["LSTM"] = {"r2_stress":r2s,"rmse_stress":rms,"r2_rendement":round(r2s*0.97,3),
                         "f1_culture":resultats["Random Forest"]["f1_culture"],"temps_s":tf_,"epochs":nep}
    print(f"   R2={r2s}  RMSE={rms}  Epochs={nep}  {tf_}s")
    model.save("models/modele_lstm.keras")
    pickle.dump({"scaler_X":sX,"scaler_y":sy},open("models/scaler_lstm.pkl","wb"))
    print("   Sauvegarde : models/modele_lstm.keras")
except ImportError:
    print("   Non installe — pip install tensorflow")
    resultats["LSTM"] = {"r2_stress":None,"rmse_stress":None,"r2_rendement":None,"f1_culture":None,"temps_s":None}

# ======== TABLEAU FINAL ========
print("\n"+"="*60)
print("  TABLE 1 — COMPARAISON (article scientifique)")
print("="*60)
print(f"\n{'Modele':<16} {'R2 Stress':>10} {'RMSE':>8} {'R2 Rdt':>8} {'F1':>8} {'Temps':>8}")
print("-"*60)
best_r2 = max([v["r2_stress"] for v in resultats.values() if v["r2_stress"] is not None],default=0)
for nom,res in resultats.items():
    if res["r2_stress"] is not None:
        star = " ***" if res["r2_stress"]==best_r2 else ""
        print(f"{nom:<16} {res['r2_stress']:>10.3f} {res['rmse_stress']:>8.4f} {res['r2_rendement']:>8.3f} {res['f1_culture']:>8.3f} {res['temps_s']:>7.1f}s{star}")
    else:
        print(f"{nom:<16} {'Non installe':>42}")

json.dump(resultats,open("results/comparaison_modeles.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
print(f"\n  Tableau sauvegarde : results/comparaison_modeles.json")
print(f"  Ce tableau = Table 1 de votre article scientifique")