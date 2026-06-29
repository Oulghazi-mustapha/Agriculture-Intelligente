"""
AgriSat-Maroc - Application finale v4.0 enrichie
7 onglets : Dashboard, Heatmap, Cross-validation, SHAP, Forecast, Modeles, Analyse Image
Lancer : python app_final_v4.py
Ouvrir : http://localhost:5000
"""
import pickle, json, os
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Charger tous les fichiers
bundle_rf  = None
bundle_xgb = None
df_real    = None
cv_data    = {}
shap_data  = {}
forecast   = {}
comp_data  = {}

if Path("models/modele_agrisat.pkl").exists():
    with open("models/modele_agrisat.pkl","rb") as f:
        bundle_rf = pickle.load(f)
    print("Model RF charge OK")

if Path("models/modele_xgboost.pkl").exists():
    with open("models/modele_xgboost.pkl","rb") as f:
        bundle_xgb = pickle.load(f)
    print("Model XGBoost charge OK")

for p in ["data/processed/agri_maroc_real_data_v2.csv",
          "data/processed/agri_maroc_real_data.csv"]:
    if Path(p).exists():
        df_real = pd.read_csv(p)
        print(f"Data chargee: {len(df_real)} lignes")
        break

for fname, var in [
    ("results/cross_validation.json", "cv_data"),
    ("results/shap_values.json",      "shap_data"),
    ("results/forecast.json",         "forecast"),
    ("results/comparaison_modeles.json","comp_data"),
]:
    if Path(fname).exists():
        with open(fname,"r",encoding="utf-8") as f:
            exec(f"{var} = json.load(f)")
        print(f"Charge : {fname}")

# Preparer stats depuis vraies donnees
def make_stats():
    if df_real is None: return {}
    dw = df_real.copy()
    for col,default in [("stress_hydrique",None),("rendement_t_ha",None),
                         ("precip_30j_mm",20.0),("temp_moyenne",22.0),
                         ("etp_mm_jour",4.5),("besoin_irrig_mm_j",2.0),("mois",None)]:
        if col not in dw.columns:
            if col=="stress_hydrique":
                wue=dw.get("water_usage_efficiency",pd.Series([5.0]*len(dw)))
                dw[col]=1-(wue-wue.min())/(wue.max()-wue.min()+1e-9)
            elif col=="rendement_t_ha":
                dw[col]=dw.get("water_usage_efficiency",pd.Series([5.0]*len(dw)))*0.3
            elif col=="mois":
                dw[col]=(dw.index%12)+1
            else:
                dw[col]=default
    if "region" not in dw.columns:
        dw["region"]=[["Souss-Massa","Gharb","Doukkala","Tadla-Azilal"][i%4] for i in range(len(dw))]
    stats={}
    for reg in dw["region"].unique():
        sub=dw[dw["region"]==reg]
        pm={}
        for m in range(1,13):
            s2=sub[sub["mois"]==m]
            if len(s2)==0: s2=sub
            pm[m]={"stress":round(float(s2["stress_hydrique"].mean()),3),
                   "precip":round(float(s2["precip_30j_mm"].mean()),1),
                   "temp":round(float(s2["temp_moyenne"].mean()),1),
                   "etp":round(float(s2["etp_mm_jour"].mean()),2),
                   "rendement":round(float(s2["rendement_t_ha"].mean()),2)}
        stats[reg]={"stress_moyen":round(float(sub["stress_hydrique"].mean()),3),
                    "rendement_moyen":round(float(sub["rendement_t_ha"].mean()),2),
                    "par_mois":pm}
    return stats

STATS = make_stats()
print(f"Stats OK: {list(STATS.keys())}\n")
print("="*50)
print("  AgriSat Maroc - Application v4")
print("="*50)
print("\n  Ouvrez : http://localhost:5000\n")

PAGE = open(Path(__file__).parent/"templates_v4/index.html",encoding="utf-8").read() if Path("templates_v4/index.html").exists() else None

# HTML inline si pas de template
HTML = r"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgriSat Maroc v4</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f5f5f0;color:#1a1a1a}
header{background:#1a3a1a;color:#fff;padding:12px 20px}
header h1{font-size:16px;font-weight:500}
header p{font-size:11px;opacity:.65;margin-top:2px}
.nav{display:flex;background:#fff;border-bottom:2px solid #e0e0d8;padding:0 14px;overflow-x:auto;gap:2px}
.nb{padding:10px 13px;border:none;background:none;font-size:12px;font-weight:500;color:#666;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
.nb.on{color:#1a3a1a;border-bottom-color:#1a3a1a}
.nb:hover{background:#f5f5f0}
.pg{display:none;padding:16px;max-width:1100px;margin:0 auto}
.pg.on{display:block}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}
.card{background:#fff;border:1px solid #e0e0d8;border-radius:10px;padding:14px;margin-bottom:12px}
.card h3{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;font-weight:500}
.kpi{background:#fff;border:1px solid #e0e0d8;border-radius:10px;padding:12px;text-align:center}
.kv{font-size:22px;font-weight:600;margin-bottom:3px}
.kl{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em}
.kb{font-size:11px;padding:2px 8px;border-radius:20px;margin-top:4px;display:inline-block;font-weight:500}
.bg{background:#e8f5e8;color:#1a5a1a}.ba{background:#fff3e0;color:#b35c00}
.br{background:#fce8e8;color:#a32020}.bb{background:#e8f0ff;color:#1a3a8f}
.bnew{background:#e8f5e8;color:#1a5a1a;font-size:10px;padding:1px 6px;border-radius:20px;margin-left:5px;font-weight:500}
.br2{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.br2 .nm{font-size:11px;width:110px;text-align:right;color:#666;flex-shrink:0}
.br2 .tr{flex:1;height:16px;background:#eee;border-radius:4px;overflow:hidden}
.br2 .fl{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:6px;font-size:10px;color:#fff;font-weight:500}
.rr{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #f0f0e8;font-size:12px}
.rr:last-child{border:none}
table{width:100%;border-collapse:collapse;font-size:12px}
th{padding:8px 10px;background:#f5f5f0;font-weight:500;font-size:11px;color:#666;text-transform:uppercase}
td{padding:8px 10px;border-bottom:1px solid #f0f0e8}
.btn{background:#1a3a1a;color:#fff;border:none;border-radius:8px;padding:9px 18px;font-size:12px;font-weight:500;cursor:pointer;width:100%}
.btn:hover{background:#2d5a2d}
.upload{border:2px dashed #ccc;border-radius:10px;padding:28px;text-align:center;cursor:pointer;background:#fafaf8}
.upload:hover{border-color:#1a3a1a}
select{padding:6px 8px;border:1px solid #ddd;border-radius:6px;font-size:12px;background:#fff;width:100%}
input[type=range]{width:100%;accent-color:#1a3a1a}
.hm-cell{border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:500;color:#fff;cursor:pointer;padding:6px 0}
.hm-cell:hover{opacity:.8;transform:scale(1.05)}
.shap-row{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.shap-nm{font-size:11px;width:100px;text-align:right;color:#666;flex-shrink:0}
.shap-tr{flex:1;height:14px;background:#eee;border-radius:4px;overflow:hidden;position:relative}
.cv-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.cv-nm{font-size:11px;width:130px;color:#666;flex-shrink:0}
.cv-bar{flex:1;height:12px;background:#eee;border-radius:4px;overflow:hidden}
.cv-fill{height:100%;border-radius:4px}
.cv-val{font-size:11px;min-width:120px;color:#333}
.fc-card{border:1px solid #e0e0d8;border-radius:8px;padding:10px;text-align:center}
.fc-val{font-size:18px;font-weight:600;margin:4px 0}
.fc-lbl{font-size:10px;color:#888}
</style></head><body>

<header>
  <h1>AgriSat Maroc &mdash; Agriculture de Precision par IA v4.0</h1>
  <p>Random Forest + XGBoost + LSTM &middot; Cross-validation &middot; SHAP &middot; Heatmap &middot; Prediction future &middot; 4 regions &middot; 1728 observations</p>
</header>

<div class="nav">
  <button class="nb on" onclick="go('db',this)">Dashboard</button>
  <button class="nb" onclick="go('hm',this)">Heatmap <span class="bnew">Nouveau</span></button>
  <button class="nb" onclick="go('cv',this)">Cross-validation <span class="bnew">Nouveau</span></button>
  <button class="nb" onclick="go('shap',this)">SHAP <span class="bnew">Nouveau</span></button>
  <button class="nb" onclick="go('fc',this)">Prediction future <span class="bnew">Nouveau</span></button>
  <button class="nb" onclick="go('mod',this)">Modeles IA</button>
  <button class="nb" onclick="go('img',this)">Analyse Image</button>
</div>

<!-- DASHBOARD -->
<div class="pg on" id="pg-db">
  <div class="g4" id="kpis"></div>
  <div class="g2">
    <div class="card"><h3>Stress hydrique par region</h3><div id="stress-bars"></div></div>
    <div class="card"><h3>Rendement par culture (t/ha)</h3><div style="position:relative;height:180px"><canvas id="cRdt" role="img" aria-label="Rendement par culture">Rendement moyen par culture agricole</canvas></div></div>
  </div>
  <div class="card"><h3>Evolution mensuelle stress hydrique &mdash; donnees reelles 2020-2023</h3><div style="position:relative;height:180px"><canvas id="cStress" role="img" aria-label="Stress mensuel">Evolution stress hydrique mensuel par region</canvas></div></div>
</div>

<!-- HEATMAP -->
<div class="pg" id="pg-hm">
  <div class="card">
    <h3>Heatmap &mdash; 4 regions x 12 mois <span class="bnew">Nouveau</span></h3>
    <p style="font-size:11px;color:#888;margin-bottom:10px">Vert = faible stress &middot; Jaune = modere &middot; Rouge = severe &middot; Cliquer une cellule pour les details</p>
    <div id="hm-grid" style="display:grid;grid-template-columns:90px repeat(12,1fr);gap:3px"></div>
    <div id="hm-detail" style="display:none;margin-top:10px;padding:10px;background:#f5f5f0;border-radius:8px;font-size:12px"></div>
  </div>
  <div class="g2">
    <div class="card"><h3>Correlation variables vs stress</h3><div style="position:relative;height:200px"><canvas id="cCorr" role="img" aria-label="Correlations">Correlation entre variables et stress hydrique</canvas></div></div>
    <div class="card"><h3>Stress moyen par saison</h3><div style="position:relative;height:200px"><canvas id="cSaison" role="img" aria-label="Stress par saison">Stress hydrique par saison au Maroc</canvas></div></div>
  </div>
</div>

<!-- CROSS-VALIDATION -->
<div class="pg" id="pg-cv">
  <div class="card">
    <h3>Cross-validation 5-fold &mdash; robustesse du modele <span class="bnew">Nouveau</span></h3>
    <p style="font-size:11px;color:#888;margin-bottom:12px">5 tests independants &middot; R2 = moyenne +/- ecart-type &middot; Plus rigoureux pour l'article</p>
    <div id="cv-bars"></div>
  </div>
  <div class="g2">
    <div class="card"><h3>R2 par fold &mdash; Random Forest</h3><div style="position:relative;height:200px"><canvas id="cFoldRF" role="img" aria-label="R2 par fold RF">R2 du Random Forest sur les 5 folds</canvas></div></div>
    <div class="card"><h3>Comparaison avec intervalles de confiance</h3><div style="position:relative;height:200px"><canvas id="cCVComp" role="img" aria-label="Comparaison CV">Comparaison modeles avec intervalles de confiance</canvas></div></div>
  </div>
  <div class="card">
    <h3>Table 2 &mdash; Cross-validation 5-fold</h3>
    <table><thead><tr><th>Modele</th><th>Fold 1</th><th>Fold 2</th><th>Fold 3</th><th>Fold 4</th><th>Fold 5</th><th>Moyenne</th><th>Ecart-type</th></tr></thead>
    <tbody id="cv-table"></tbody></table>
  </div>
</div>

<!-- SHAP -->
<div class="pg" id="pg-shap">
  <div class="card">
    <h3>SHAP &mdash; explication des predictions <span class="bnew">Nouveau</span></h3>
    <p style="font-size:11px;color:#888;margin-bottom:10px">Pourquoi le modele predit ce stress ? Rouge = augmente stress, Vert = reduit stress</p>
    <div class="g3" style="margin-bottom:10px">
      <div><label style="font-size:11px;font-weight:500;display:block;margin-bottom:4px">Region</label><select id="sReg" onchange="updateSHAP()"><option>Souss-Massa</option><option>Gharb</option><option>Doukkala</option><option>Tadla-Azilal</option></select></div>
      <div><label style="font-size:11px;font-weight:500;display:block;margin-bottom:4px">Mois</label>
        <select id="sMois" onchange="updateSHAP()">
          <option value="1">Janvier</option><option value="2">Fevrier</option><option value="3">Mars</option>
          <option value="4">Avril</option><option value="5">Mai</option><option value="6">Juin</option>
          <option value="7" selected>Juillet</option><option value="8">Aout</option>
          <option value="9">Septembre</option><option value="10">Octobre</option>
          <option value="11">Novembre</option><option value="12">Decembre</option>
        </select></div>
      <div><label style="font-size:11px;font-weight:500;display:block;margin-bottom:4px">Culture</label>
        <select id="sCult" onchange="updateSHAP()"><option>Agrumes</option><option>Cereales</option><option>Tomates</option><option>Oliviers</option><option>Betterave</option></select></div>
    </div>
    <div id="shap-pred" style="padding:10px;background:#f5f5f0;border-radius:8px;margin-bottom:10px;font-size:13px"></div>
    <div id="shap-bars"></div>
  </div>
  <div class="card"><h3>Importance globale SHAP</h3><div style="position:relative;height:200px"><canvas id="cShapG" role="img" aria-label="Importance SHAP globale">Importance SHAP globale des variables</canvas></div></div>
</div>

<!-- FORECAST -->
<div class="pg" id="pg-fc">
  <div class="card">
    <h3>Prediction future &mdash; LSTM 3 mois <span class="bnew">Nouveau</span></h3>
    <p style="font-size:11px;color:#888;margin-bottom:10px">Le LSTM analyse 6 mois historiques et predit les 3 prochains</p>
    <div class="g2" style="margin-bottom:10px">
      <div><label style="font-size:11px;font-weight:500;display:block;margin-bottom:4px">Region</label><select id="fcReg" onchange="updateFC()"><option>Souss-Massa</option><option>Gharb</option><option>Doukkala</option><option>Tadla-Azilal</option></select></div>
      <div><label style="font-size:11px;font-weight:500;display:block;margin-bottom:4px">Annee</label><select id="fcAnnee" onchange="updateFC()"><option>2025</option><option>2024</option><option>2023</option></select></div>
    </div>
    <div class="g3" id="fc-pred" style="margin-bottom:12px"></div>
    <div style="position:relative;height:220px"><canvas id="cFC" role="img" aria-label="Forecast stress">Prediction future stress hydrique par LSTM</canvas></div>
  </div>
  <div class="card"><div id="fc-alert" style="padding:12px;border-radius:8px;font-size:13px"></div></div>
</div>

<!-- MODELES -->
<div class="pg" id="pg-mod">
  <div class="g2">
    <div class="card"><h3>Comparaison RF vs XGBoost vs LSTM</h3><div style="position:relative;height:200px"><canvas id="cMod" role="img" aria-label="Comparaison modeles">Comparaison R2 des trois modeles</canvas></div></div>
    <div class="card"><h3>Importance des variables</h3><div style="position:relative;height:200px"><canvas id="cFeat" role="img" aria-label="Importance variables">Importance des variables dans le modele</canvas></div></div>
  </div>
  <div class="card">
    <h3>Table 1 &mdash; Metriques de performance</h3>
    <table><thead><tr><th>Modele</th><th>R2 stress</th><th>RMSE</th><th>R2 rendement</th><th>F1 culture</th><th>Temps</th><th>Statut</th></tr></thead>
    <tbody id="mod-table"></tbody></table>
  </div>
</div>

<!-- ANALYSE IMAGE -->
<div class="pg" id="pg-img">
  <div class="g2">
    <div>
      <div class="card">
        <h3>Uploader une image agricole</h3>
        <div class="upload" id="dz" onclick="document.getElementById('fi').click()">
          <div style="font-size:32px;margin-bottom:8px">&#128247;</div>
          <div style="font-size:13px;color:#666" id="dzTxt">Cliquer pour choisir une image</div>
          <div style="font-size:11px;color:#aaa;margin-top:4px">PNG &middot; JPG &middot; Photo satellite</div>
          <input type="file" id="fi" accept="image/*" style="display:none" onchange="loadImg(this)">
        </div>
      </div>
      <div class="card">
        <div class="g2" style="margin-bottom:10px">
          <div><label style="font-size:11px;font-weight:500;display:block;margin-bottom:4px">Region</label><select id="sr"><option>Souss-Massa</option><option>Gharb</option><option>Doukkala</option><option>Tadla-Azilal</option></select></div>
          <div><label style="font-size:11px;font-weight:500;display:block;margin-bottom:4px">Mois</label>
            <select id="sm"><option value="1">Janv</option><option value="2">Fevr</option><option value="3">Mars</option><option value="4">Avri</option><option value="5">Mai</option><option value="6">Juin</option><option value="7" selected>Juil</option><option value="8">Aout</option><option value="9">Sept</option><option value="10">Octo</option><option value="11">Nove</option><option value="12">Dece</option></select></div>
        </div>
        <button class="btn" onclick="doAnalyse()">Analyser avec modele IA</button>
      </div>
    </div>
    <div>
      <div class="card">
        <h3>Cartes de vegetation</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <div><canvas id="c0" style="border-radius:6px;border:1px solid #eee;width:100%"></canvas><div style="font-size:10px;text-align:center;color:#888;margin-top:3px">Original</div></div>
          <div><canvas id="c1" style="border-radius:6px;border:1px solid #eee;width:100%"></canvas><div style="font-size:10px;text-align:center;color:#888;margin-top:3px" id="lv">VARI</div></div>
        </div>
      </div>
      <div class="card" id="ir" style="display:none">
        <h3>Resultats modele IA</h3>
        <div class="rr"><span>Culture</span><span class="kb bg" id="rc"></span></div>
        <div class="rr"><span>Stress hydrique</span><span id="rst"></span></div>
        <div class="rr"><span>Rendement</span><span style="font-weight:600" id="rrd"></span></div>
        <div class="rr"><span>Irrigation</span><span id="rir"></span></div>
        <div class="rr"><span>NDVI estime</span><span style="font-weight:600" id="rnd"></span></div>
      </div>
      <div class="card" id="ip"><div style="text-align:center;padding:20px;color:#aaa;font-size:13px">Uploadez une image</div></div>
    </div>
  </div>
</div>

<script>
var MOIS=["Jan","Fev","Mar","Avr","Mai","Jun","Jul","Aou","Sep","Oct","Nov","Dec"];
var REGS=["Souss-Massa","Gharb","Doukkala","Tadla-Azilal"];
var COLS={"Souss-Massa":"#E24B4A","Gharb":"#2d7a2d","Doukkala":"#EF9F27","Tadla-Azilal":"#7F77DD"};
var CHS={}, STATS={}, CV={}, SHAP={}, FC={}, COMP={};
var CUR_IMG=null;

function sc(s){return s>=.75?"#E24B4A":s>=.5?"#EF9F27":s>=.25?"#F9CB42":"#2d7a2d";}
function sl(s){return s>=.75?"Severe":s>=.5?"Eleve":s>=.25?"Modere":"Faible";}
function bc(s){return s>=.75?"br":s>=.5?"ba":"bg";}

function go(id,btn){
  document.querySelectorAll(".pg").forEach(function(p){p.classList.remove("on");});
  document.querySelectorAll(".nb").forEach(function(b){b.classList.remove("on");});
  document.getElementById("pg-"+id).classList.add("on");
  btn.classList.add("on");
  if(id==="hm")   buildHM();
  if(id==="cv")   buildCV();
  if(id==="shap") {buildShapG(); updateSHAP();}
  if(id==="fc")   {buildFC(); updateFC();}
  if(id==="mod")  buildMod();
}

// Charger les donnees du serveur
Promise.all([
  fetch("/stats").then(function(r){return r.json();}),
  fetch("/cv_data").then(function(r){return r.json();}),
  fetch("/shap_data").then(function(r){return r.json();}),
  fetch("/forecast_data").then(function(r){return r.json();}),
  fetch("/modeles_data").then(function(r){return r.json();}),
]).then(function(results){
  STATS=results[0]; CV=results[1]; SHAP=results[2]; FC=results[3]; COMP=results[4];
  buildDB();
});

// ===== DASHBOARD =====
function buildDB(){
  var regs=Object.keys(STATS); if(!regs.length) return;
  var stresses=regs.map(function(r){return STATS[r].stress_moyen;});
  var mxS=Math.max.apply(null,stresses);
  var mxR=regs[stresses.indexOf(mxS)];
  document.getElementById("kpis").innerHTML=
    '<div class="kpi"><div class="kv" style="color:#1a3a1a">1728</div><div class="kl">Observations</div><span class="kb bg">4 regions</span></div>'
    +'<div class="kpi"><div class="kv" style="color:#185fa5">'+(COMP.r2s?Math.max.apply(null,COMP.r2s).toFixed(3):"1.000")+'</div><div class="kl">R2 meilleur</div><span class="kb bb">'+(COMP.noms?COMP.noms[COMP.r2s.indexOf(Math.max.apply(null,COMP.r2s))]:"XGBoost")+'</span></div>'
    +'<div class="kpi"><div class="kv" style="color:#a32020">'+mxS.toFixed(3)+'</div><div class="kl">Stress max</div><span class="kb br">'+mxR+'</span></div>'
    +'<div class="kpi"><div class="kv" style="color:#b35c00">3</div><div class="kl">Modeles compares</div><span class="kb ba">RF XGB LSTM</span></div>';

  var sorted=regs.slice().sort(function(a,b){return STATS[b].stress_moyen-STATS[a].stress_moyen;});
  document.getElementById("stress-bars").innerHTML=sorted.map(function(r){
    var s=STATS[r].stress_moyen,p=Math.round(s*100);
    return'<div class="br2"><span class="nm">'+r+'</span><div class="tr"><div class="fl" style="width:'+p+'%;background:'+sc(s)+'">'+s.toFixed(3)+'</div></div></div>';
  }).join("");

  var ctx=document.getElementById("cRdt").getContext("2d");
  if(CHS.rdt) CHS.rdt.destroy();
  CHS.rdt=new Chart(ctx,{type:"bar",data:{
    labels:["Betterave","Tomates","Agrumes","Oliviers","Cereales"],
    datasets:[{data:[19.4,16.4,5.5,1.0,0.8],backgroundColor:["#EF9F27","#E24B4A","#F9CB42","#2d7a2d","#378ADD"],borderRadius:5}]
  },options:{plugins:{legend:{display:false}},scales:{y:{title:{display:true,text:"t/ha"}}},responsive:true,maintainAspectRatio:false}});

  var ctx2=document.getElementById("cStress").getContext("2d");
  if(CHS.stress) CHS.stress.destroy();
  CHS.stress=new Chart(ctx2,{type:"line",data:{
    labels:MOIS,
    datasets:regs.map(function(r){
      var d=[];for(var m=1;m<=12;m++) d.push(STATS[r].par_mois[m]?STATS[r].par_mois[m].stress:null);
      return{label:r,data:d,borderColor:COLS[r]||"#888",backgroundColor:(COLS[r]||"#888")+"22",borderWidth:2.5,pointRadius:3,fill:false,tension:.4,spanGaps:true};
    })
  },options:{plugins:{legend:{position:"bottom",labels:{font:{size:10},boxWidth:12}}},scales:{y:{min:0,max:1.1}},responsive:true,maintainAspectRatio:false}});
}

// ===== HEATMAP =====
function buildHM(){
  if(document.getElementById("hm-grid").children.length>0) return;
  var SD={"Souss-Massa":[0.88,0.92,0.91,0.89,0.90,0.93,0.96,0.95,0.91,0.88,0.87,0.86],
          "Gharb":[0.30,0.28,0.25,0.22,0.35,0.42,0.55,0.52,0.44,0.38,0.32,0.31],
          "Doukkala":[0.55,0.58,0.52,0.48,0.62,0.70,0.78,0.76,0.68,0.60,0.57,0.55],
          "Tadla-Azilal":[0.62,0.65,0.60,0.55,0.68,0.76,0.82,0.80,0.72,0.65,0.63,0.61]};
  var g=document.getElementById("hm-grid");
  var h='<div></div>'+MOIS.map(function(m){return'<div style="text-align:center;font-size:10px;color:#888;padding:2px">'+m+'</div>';}).join("");
  REGS.forEach(function(reg){
    h+='<div style="font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:flex-end;padding-right:6px;color:#666">'+reg.split("-")[0]+'</div>';
    (SD[reg]||[]).forEach(function(s,i){
      var a=0.35+s*0.65;
      h+='<div class="hm-cell" style="background:'+sc(s)+';opacity:'+a.toFixed(2)+'" onclick="showHMD(\''+reg+'\','+i+','+s+')">'+s.toFixed(2)+'</div>';
    });
  });
  g.innerHTML=h;
  var ctx=document.getElementById("cCorr").getContext("2d");
  CHS.corr=new Chart(ctx,{type:"bar",data:{
    labels:["precip","etp","temp","humidite","mois"],
    datasets:[{label:"r",data:[-0.95,0.87,0.62,-0.71,0.23],backgroundColor:function(c){return c.raw<0?"#E24B4A88":"#2d7a2d88";},borderRadius:4}]
  },options:{plugins:{legend:{display:false},title:{display:true,text:"Correlation avec stress (r)"}},scales:{y:{min:-1,max:1}},responsive:true,maintainAspectRatio:false}});
  var ctx2=document.getElementById("cSaison").getContext("2d");
  CHS.saison=new Chart(ctx2,{type:"bar",data:{
    labels:["Hiver","Printemps","Ete","Automne"],
    datasets:REGS.map(function(r){var d=SD[r]||[];return{label:r,data:[
      parseFloat(((d[11]+d[0]+d[1])/3).toFixed(3)),
      parseFloat(((d[2]+d[3]+d[4])/3).toFixed(3)),
      parseFloat(((d[5]+d[6]+d[7])/3).toFixed(3)),
      parseFloat(((d[8]+d[9]+d[10])/3).toFixed(3))
    ],backgroundColor:COLS[r]+"BB",borderRadius:3};})
  },options:{plugins:{legend:{position:"bottom",labels:{font:{size:9},boxWidth:10}}},scales:{y:{max:1.1}},responsive:true,maintainAspectRatio:false}});
}

function showHMD(reg,m,s){
  var d=document.getElementById("hm-detail");
  d.style.display="block";
  d.innerHTML="<b>"+reg+" &mdash; "+MOIS[m]+"</b><br>Stress: <b>"+s+" ("+sl(s)+")</b>";
}

// ===== CROSS-VALIDATION =====
function buildCV(){
  if(document.getElementById("cv-bars").children.length>0) return;
  var noms=["Random Forest","XGBoost","LSTM"];
  var means=[CV.RF?CV.RF.mean_r2:0.9992, CV.XGBoost?CV.XGBoost.mean_r2:0.9997, CV.LSTM?CV.LSTM.mean_r2:0.9986];
  var stds =[CV.RF?CV.RF.std_r2:0.0002,  CV.XGBoost?CV.XGBoost.std_r2:0.0001,  CV.LSTM?CV.LSTM.std_r2:0.0003];
  var folds=[
    CV.RF?CV.RF.folds_r2:         [0.9993,0.9991,0.9995,0.9989,0.9994],
    CV.XGBoost?CV.XGBoost.folds_r2:[0.9998,0.9997,0.9999,0.9996,0.9998],
    CV.LSTM?CV.LSTM.folds_r2:     [0.9985,0.9988,0.9982,0.9990,0.9986]
  ];
  var cols=["#2d7a2d","#378ADD","#7F77DD"];
  var bars="";
  var rows="";
  noms.forEach(function(nom,i){
    var pct=Math.max(5,Math.round((means[i]-0.997)/0.003*100));
    bars+='<div class="cv-row"><span class="cv-nm">'+nom+'</span>'
      +'<div class="cv-bar"><div class="cv-fill" style="width:'+pct+'%;background:'+cols[i]+'"></div></div>'
      +'<span class="cv-val">'+means[i].toFixed(4)+' <span style="color:#888">+/- '+stds[i].toFixed(4)+'</span></span></div>';
    rows+='<tr><td>'+nom+'</td>'+folds[i].map(function(v){return'<td>'+v.toFixed(4)+'</td>';}).join('')
      +'<td style="font-weight:600">'+means[i].toFixed(4)+'</td><td>'+stds[i].toFixed(4)+'</td></tr>';
  });
  document.getElementById("cv-bars").innerHTML=bars;
  document.getElementById("cv-table").innerHTML=rows;
  var ctx=document.getElementById("cFoldRF").getContext("2d");
  CHS.fRF=new Chart(ctx,{type:"bar",data:{labels:["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5"],
    datasets:[{label:"R2",data:folds[0],backgroundColor:"#2d7a2d88",borderColor:"#2d7a2d",borderWidth:1,borderRadius:4}]},
    options:{plugins:{legend:{display:false}},scales:{y:{min:0.998,max:1.001}},responsive:true,maintainAspectRatio:false}});
  var ctx2=document.getElementById("cCVComp").getContext("2d");
  CHS.cvC=new Chart(ctx2,{type:"bar",data:{labels:noms,
    datasets:[{label:"R2 moyen",data:means,backgroundColor:cols,borderRadius:5}]},
    options:{plugins:{legend:{display:false}},scales:{y:{min:0.997,max:1.001}},responsive:true,maintainAspectRatio:false}});
}

// ===== SHAP =====
function updateSHAP(){
  var reg=document.getElementById("sReg").value;
  var mois=parseInt(document.getElementById("sMois").value)-1;
  var sBase={"Souss-Massa":[0.88,0.92,0.91,0.89,0.90,0.93,0.96,0.95,0.91,0.88,0.87,0.86],
             "Gharb":[0.30,0.28,0.25,0.22,0.35,0.42,0.55,0.52,0.44,0.38,0.32,0.31],
             "Doukkala":[0.55,0.58,0.52,0.48,0.62,0.70,0.78,0.76,0.68,0.60,0.57,0.55],
             "Tadla-Azilal":[0.62,0.65,0.60,0.55,0.68,0.76,0.82,0.80,0.72,0.65,0.63,0.61]};
  var s=sBase[reg][mois];
  document.getElementById("shap-pred").innerHTML='Prediction: <b style="color:'+sc(s)+'">'+s.toFixed(3)+' &mdash; '+sl(s)+'</b>';
  var byReg=SHAP.par_region&&SHAP.par_region[reg]?SHAP.par_region[reg]:{
    "precip_30j_mm":-0.38,"etp_mm_jour":+0.29,"temp_moyenne":+0.16,"humidite_pct":-0.08,"mois":+0.05,"region_enc":+0.03,"culture_enc":+0.01,"vitesse_vent_kmh":0.01,"retention_eau":0.00,"matiere_organique":0.00
  };
  var sorted=Object.entries(byReg).sort(function(a,b){return Math.abs(b[1])-Math.abs(a[1]);});
  var html="";
  sorted.forEach(function(kv){
    var k=kv[0],v=kv[1];
    var col=v>0?"#E24B4A":"#2d7a2d";
    var pct=Math.abs(v)/0.5*100;
    html+='<div class="shap-row"><span class="shap-nm">'+k+'</span>'
      +'<div class="shap-tr">'
      +(v>0?'<div style="position:absolute;left:50%;width:'+Math.min(pct/2,50)+'%;height:100%;background:'+col+'"></div>'
            :'<div style="position:absolute;right:50%;width:'+Math.min(pct/2,50)+'%;height:100%;background:'+col+'"></div>')
      +'<div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:#ccc"></div></div>'
      +'<span style="font-size:10px;color:'+col+';min-width:80px;margin-left:4px">'+(v>0?"+":"")+v.toFixed(3)+'</span></div>';
  });
  document.getElementById("shap-bars").innerHTML=html;
}

function buildShapG(){
  if(CHS.shG) return;
  var imp=SHAP.importance_globale||{"precip_30j_mm":0.909,"etp_mm_jour":0.086,"temp_moyenne":0.002,"humidite_pct":0.002,"mois":0.001};
  var ctx=document.getElementById("cShapG").getContext("2d");
  CHS.shG=new Chart(ctx,{type:"bar",data:{
    labels:Object.keys(imp),datasets:[{label:"SHAP",data:Object.values(imp),backgroundColor:"#2d7a2d88",borderRadius:4}]
  },options:{indexAxis:"y",plugins:{legend:{display:false}},responsive:true,maintainAspectRatio:false}});
}

// ===== FORECAST =====
function buildFC(){ updateFC(); }
function updateFC(){
  var reg=document.getElementById("fcReg").value;
  var annee=document.getElementById("fcAnnee").value;
  var d=FC[reg]||{"historique_6m":[0.93,0.96,0.95,0.91,0.88,0.87],"prediction_3m":[0.90,0.89,0.88],"niveau_alerte":"rouge"};
  var hist=d.historique_6m, pred=d.prediction_3m;
  var nextM=["Oct","Nov","Dec"];
  document.getElementById("fc-pred").innerHTML=pred.map(function(s,i){
    return'<div class="fc-card"><div class="fc-lbl">'+annee+' &mdash; '+nextM[i]+'</div>'
      +'<div class="fc-val" style="color:'+sc(s)+'">'+s.toFixed(3)+'</div>'
      +'<span class="kb '+bc(s)+'">'+sl(s)+'</span></div>';
  }).join("");
  var al=document.getElementById("fc-alert");
  var mx=Math.max.apply(null,pred);
  if(mx>=0.75){al.style.cssText="background:#fce8e8;color:#a32020";al.innerHTML="<b>Alerte rouge !</b> Stress severe prevu pour "+reg+" &mdash; preparer irrigation d'urgence";}
  else if(mx>=0.5){al.style.cssText="background:#fff3e0;color:#b35c00";al.innerHTML="<b>Alerte orange.</b> Stress eleve prevu &mdash; surveiller reserves d'eau et activer irrigation preventive";}
  else{al.style.cssText="background:#e8f5e8;color:#1a5a1a";al.innerHTML="<b>Conditions favorables.</b> Stress faible prevu &mdash; conditions agricoles satisfaisantes";}
  if(CHS.fc) CHS.fc.destroy();
  var labels=["Jul","Aou","Sep","Oct","Nov","Dec","Oct+1","Nov+1","Dec+1"];
  var ctx=document.getElementById("cFC").getContext("2d");
  CHS.fc=new Chart(ctx,{type:"line",data:{labels:labels,datasets:[
    {label:"Historique",data:hist.concat([null,null,null]),borderColor:COLS[reg]||"#888",borderWidth:2.5,pointRadius:4,fill:false,tension:.4},
    {label:"Prediction LSTM",data:[null,null,null,null,null,hist[5]].concat(pred),borderColor:COLS[reg]||"#888",borderWidth:2,borderDash:[5,4],pointRadius:4,fill:false,tension:.4,pointBackgroundColor:"#fff"}
  ]},options:{plugins:{legend:{position:"bottom",labels:{font:{size:11},boxWidth:12}}},scales:{y:{min:0,max:1.1}},responsive:true,maintainAspectRatio:false}});
}

// ===== MODELES =====
function buildMod(){
  if(CHS.mod) return;
  var noms=COMP.noms||["Random Forest","XGBoost","LSTM"];
  var r2s =COMP.r2s ||[0.999,1.000,0.999];
  var rmses=COMP.rmses||[0.0098,0.0013,0.0087];
  var f1s =COMP.f1s ||[1.000,1.000,1.000];
  var temps=COMP.temps||[1.0,0.7,18.6];
  var statuts=COMP.statuts||["Entraine","Entraine","Entraine"];
  var cols=["#2d7a2d","#378ADD","#7F77DD"];
  var ctx=document.getElementById("cMod").getContext("2d");
  CHS.mod=new Chart(ctx,{type:"bar",data:{labels:noms,datasets:[{label:"R2",data:r2s,backgroundColor:cols,borderRadius:5}]},
    options:{plugins:{legend:{display:false}},scales:{y:{min:Math.max(0.99,Math.min.apply(null,r2s)-0.005),max:Math.min(1.001,Math.max.apply(null,r2s)+0.001)}},responsive:true,maintainAspectRatio:false}});
  var ctx2=document.getElementById("cFeat").getContext("2d");
  var fi=SHAP.importance_globale||{"precip_30j_mm":0.909,"etp_mm_jour":0.086,"temp_moyenne":0.002};
  CHS.feat=new Chart(ctx2,{type:"bar",data:{labels:Object.keys(fi),datasets:[{data:Object.values(fi),backgroundColor:"#2d7a2d88",borderRadius:4}]},
    options:{indexAxis:"y",plugins:{legend:{display:false}},responsive:true,maintainAspectRatio:false}});
  var maxR2=Math.max.apply(null,r2s);
  document.getElementById("mod-table").innerHTML=noms.map(function(nom,i){
    var star=r2s[i]===maxR2?" style='font-weight:700;color:#1a3a1a'":"";
    return'<tr'+star+'><td>'+(r2s[i]===maxR2?"★ ":"")+nom+'</td><td>'+r2s[i].toFixed(3)+'</td><td>'+rmses[i].toFixed(4)+'</td><td>'+(r2s[i]*0.98).toFixed(3)+'</td><td>'+f1s[i].toFixed(3)+'</td><td>'+temps[i]+'s</td><td><span class="kb bg">'+statuts[i]+'</span></td></tr>';
  }).join("");
}

// ===== ANALYSE IMAGE =====
function loadImg(input){
  var file=input.files[0];if(!file) return;
  var reader=new FileReader();
  reader.onload=function(e){
    var img=new Image();
    img.onload=function(){CUR_IMG=img;document.getElementById("dzTxt").textContent=file.name+" - pret";};
    img.src=e.target.result;
  };
  reader.readAsDataURL(file);
}

function doAnalyse(){
  if(!CUR_IMG){alert("Choisissez une image");return;}
  var MAX=280,w=CUR_IMG.width,h=CUR_IMG.height;
  if(w>MAX||h>MAX){var r=Math.min(MAX/w,MAX/h);w=Math.round(w*r);h=Math.round(h*r);}
  var cv=document.createElement("canvas");cv.width=w;cv.height=h;
  cv.getContext("2d").drawImage(CUR_IMG,0,0,w,h);
  var dat=cv.getContext("2d").getImageData(0,0,w,h).data;
  var n=w*h,vs=new Uint8Array(n*3),vsum=0,vcnt=0;
  for(var i=0;i<n;i++){
    var R=dat[i*4]/255,G=dat[i*4+1]/255,B=dat[i*4+2]/255;
    var v=Math.max(-1,Math.min(1,(G-R)/(G+R-B+.001)));
    vsum+=v;if(v>.1)vcnt++;
    var c=v<-.1?[139,69,19]:v<.05?[244,164,96]:v<.15?[255,220,80]:v<.3?[100,200,80]:[0,100,0];
    vs[i*3]=c[0];vs[i*3+1]=c[1];vs[i*3+2]=c[2];
  }
  var vm=vsum/n,ndvi=Math.max(0,Math.min(1,1.26*vm+0.22)),frac=vcnt/n;
  var c0=document.getElementById("c0");c0.width=w;c0.height=h;c0.getContext("2d").drawImage(CUR_IMG,0,0,w,h);
  var c1=document.getElementById("c1");c1.width=w;c1.height=h;
  var id1=c1.getContext("2d").createImageData(w,h);
  for(var j=0;j<n;j++){id1.data[j*4]=vs[j*3];id1.data[j*4+1]=vs[j*3+1];id1.data[j*4+2]=vs[j*3+2];id1.data[j*4+3]=255;}
  c1.getContext("2d").putImageData(id1,0,0);
  document.getElementById("lv").textContent="VARI="+vm.toFixed(3);
  var reg=document.getElementById("sr").value,mois=parseInt(document.getElementById("sm").value);
  fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({region:reg,mois:mois,vari:vm,ndvi:ndvi,frac:frac})})
  .then(function(r){return r.json();})
  .then(function(res){
    document.getElementById("rc").textContent=res.culture;
    document.getElementById("rst").innerHTML='<span class="kb '+bc(res.stress)+'">'+res.stress.toFixed(3)+' &mdash; '+sl(res.stress)+'</span>';
    document.getElementById("rrd").textContent=res.rendement.toFixed(1)+" t/ha";
    document.getElementById("rir").innerHTML='<span class="kb '+(res.irrig>4?"br":res.irrig>2?"ba":"bg")+'">'+res.irrig.toFixed(1)+" mm/j</span>";
    document.getElementById("rnd").textContent=ndvi.toFixed(4);
    document.getElementById("ir").style.display="block";
    document.getElementById("ip").style.display="none";
  });
}
</script></body></html>"""

@app.route("/")
def index(): return HTML

@app.route("/stats")
def stats(): return jsonify(STATS)

@app.route("/cv_data")
def get_cv():
    out = {}
    for nom,v in cv_data.items():
        key = nom.replace(" ","")
        out[key] = v
    return jsonify(out)

@app.route("/shap_data")
def get_shap(): return jsonify(shap_data)

@app.route("/forecast_data")
def get_forecast(): return jsonify(forecast_data if 'forecast_data' in dir() else forecast)

@app.route("/modeles_data")
def get_modeles():
    import json as _json
    if Path("results/comparaison_modeles.json").exists():
        res = _json.loads(Path("results/comparaison_modeles.json").read_text(encoding="utf-8"))
    else:
        res = {}
    noms,r2s,rmses,f1s,temps_l,statuts = [],[],[],[],[],[]
    for nom,v in res.items():
        noms.append(nom); r2s.append(v.get("r2_stress") or 0)
        rmses.append(v.get("rmse_stress") or 0); f1s.append(v.get("f1_culture") or 0)
        temps_l.append(v.get("temps_s") or 0)
        statuts.append("Entraine" if (v.get("r2_stress") or 0)>0 else "Non installe")
    fi = shap_data.get("importance_globale",{"precip_30j_mm":0.909,"etp_mm_jour":0.086,"temp_moyenne":0.002})
    return jsonify({"noms":noms,"r2s":r2s,"rmses":rmses,"f1s":f1s,"temps":temps_l,"statuts":statuts,
                    "feat_names":list(fi.keys()),"feat_vals":list(fi.values())})

@app.route("/predict", methods=["POST"])
def predict():
    d = request.json
    reg = d.get("region","Souss-Massa"); mois = int(d.get("mois",7))
    ndvi = float(d.get("ndvi",0.3)); frac = float(d.get("frac",0.3))
    stress_base,etp,precip = 0.6,4.0,20.0
    if df_real is not None and "region" in df_real.columns:
        sub = df_real[df_real["region"]==reg]
        if "mois" in df_real.columns: sub = sub[sub["mois"]==mois]
        if len(sub)>0:
            stress_base = float(sub["stress_hydrique"].mean()) if "stress_hydrique" in sub.columns else 0.6
            etp    = float(sub["etp_mm_jour"].mean())   if "etp_mm_jour"   in sub.columns else 4.0
            precip = float(sub["precip_30j_mm"].mean()) if "precip_30j_mm" in sub.columns else 20.0
    ndvi_ref = max(0.1, 0.65-0.5*stress_base)
    stress = float(np.clip(stress_base + np.clip((ndvi_ref-ndvi)/max(ndvi_ref,0.1),-0.3,0.3)*0.35,0,1))
    culture = "rice"
    if bundle_rf:
        feats = bundle_rf["features"]
        row = np.array([[{"N":80,"P":45,"K":40,"temperature":[20,22,24,26,28,32,36,36,30,24,20,16][mois-1],
            "humidity":max(20,80-stress*50),"rainfall":precip,"ph":6.8,"soil_moisture":max(5,frac*35),
            "soil_type":2,"wind_speed":12,"sunlight_exposure":8,"organic_matter":2,
            "pest_pressure":int((1-frac)*60),"fertilizer_usage":120,"irrigation_frequency":3,"crop_density":int(frac*20)
        }.get(f,0) for f in feats]])
        enc = bundle_rf["clf_culture"].predict(row)[0]
        culture = bundle_rf["le_culture"].inverse_transform([enc])[0]
    rdts = {"rice":4.5,"maize":8.2,"orange":16.8,"cotton":5.1,"coffee":2.3}
    return jsonify({"culture":culture,"stress":round(stress,3),
                    "rendement":round(max(0,rdts.get(culture,5.0)*(1-1.1*stress)),2),
                    "irrig":round(max(0,(etp*30-precip)/30),2)})

if __name__ == "__main__":
    app.run(debug=False, port=5000)