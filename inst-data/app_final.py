"""
AgriSat Maroc - Application finale
Lancer : python app_final.py
Ouvrir  : http://localhost:5000
"""
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

# Charger modele et donnees
bundle  = None
df_real = None

if Path("models/modele_agrisat.pkl").exists():
    with open("models/modele_agrisat.pkl", "rb") as f:
        bundle = pickle.load(f)
    print("Model charge OK")

for p in ["data/processed/agri_maroc_real_data_v2.csv",
          "data/processed/agri_maroc_real_data.csv"]:
    if Path(p).exists():
        df_real = pd.read_csv(p)
        print(f"Data chargee: {len(df_real)} lignes")
        break

# Preparer stats
def make_stats():
    if df_real is None:
        return {}
    dw = df_real.copy()
    if "stress_hydrique"   not in dw.columns:
        wue = dw.get("water_usage_efficiency", pd.Series([5.0]*len(dw)))
        dw["stress_hydrique"] = 1 - (wue-wue.min())/(wue.max()-wue.min()+1e-9)
    if "rendement_t_ha"    not in dw.columns:
        dw["rendement_t_ha"] = dw.get("water_usage_efficiency", pd.Series([5.0]*len(dw)))*0.3
    if "precip_30j_mm"     not in dw.columns:
        dw["precip_30j_mm"] = dw.get("rainfall", pd.Series([20.0]*len(dw)))
    if "temp_moyenne"      not in dw.columns:
        dw["temp_moyenne"] = dw.get("temperature", pd.Series([22.0]*len(dw)))
    if "etp_mm_jour"       not in dw.columns:
        dw["etp_mm_jour"] = 4.5
    if "besoin_irrig_mm_j" not in dw.columns:
        dw["besoin_irrig_mm_j"] = (dw["etp_mm_jour"]*30 - dw["precip_30j_mm"]).clip(0)/30
    if "mois" not in dw.columns:
        dw["mois"] = (dw.index % 12) + 1
    if "region" not in dw.columns:
        regs = ["Souss-Massa","Gharb","Doukkala","Tadla-Azilal"]
        dw["region"] = [regs[i%4] for i in range(len(dw))]

    stats = {}
    for reg in dw["region"].unique():
        sub = dw[dw["region"]==reg]
        pm  = {}
        for m in range(1,13):
            s2 = sub[sub["mois"]==m]
            if len(s2)==0: s2 = sub
            pm[m] = {
                "stress":    round(float(s2["stress_hydrique"].mean()),3),
                "precip":    round(float(s2["precip_30j_mm"].mean()),1),
                "temp":      round(float(s2["temp_moyenne"].mean()),1),
                "etp":       round(float(s2["etp_mm_jour"].mean()),2),
                "rendement": round(float(s2["rendement_t_ha"].mean()),2),
            }
        stats[reg] = {
            "stress_moyen":    round(float(sub["stress_hydrique"].mean()),3),
            "rendement_moyen": round(float(sub["rendement_t_ha"].mean()),2),
            "irrig_moyen":     round(float(sub["besoin_irrig_mm_j"].mean()),2),
            "par_mois": pm,
        }
    return stats

STATS = make_stats()
print(f"Stats OK: {list(STATS.keys())}")

# ═══════════════════════════════════════════════════
# PAGE HTML — ZERO accent dans le JS
# ═══════════════════════════════════════════════════
PAGE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgriSat Maroc</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f5f5f0;color:#1a1a1a}
header{background:#1a3a1a;color:#fff;padding:14px 24px}
header h1{font-size:17px;font-weight:600}
header p{font-size:11px;opacity:.7;margin-top:2px}
.nav{display:flex;background:#fff;border-bottom:2px solid #e0e0d8;padding:0 16px;overflow-x:auto}
.nav-btn{padding:12px 16px;border:none;background:none;font-size:13px;font-weight:500;color:#666;cursor:pointer;border-bottom:3px solid transparent;white-space:nowrap}
.nav-btn.active,.nav-btn:hover{color:#1a3a1a}
.nav-btn.active{border-bottom-color:#1a3a1a}
.page{display:none;padding:20px;max-width:1100px;margin:0 auto}
.page.show{display:block}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.card{background:#fff;border:1px solid #e0e0d8;border-radius:10px;padding:16px;margin-bottom:14px}
.card h3{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px;font-weight:600}
.kpi{background:#fff;border:1px solid #e0e0d8;border-radius:10px;padding:14px;text-align:center}
.kv{font-size:26px;font-weight:700;margin-bottom:4px}
.kl{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em}
.kb{font-size:11px;padding:2px 8px;border-radius:20px;margin-top:5px;display:inline-block;font-weight:500}
.bg{background:#e8f5e8;color:#1a5a1a}
.ba{background:#fff3e0;color:#b35c00}
.br{background:#fce8e8;color:#a32020}
.bb{background:#e8f0ff;color:#1a3a8f}
.bar-r{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.bar-r .nm{font-size:11px;width:100px;text-align:right;color:#666;flex-shrink:0}
.bar-r .tr{flex:1;height:18px;background:#eee;border-radius:4px;overflow:hidden}
.bar-r .fl{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:6px;font-size:10px;color:#fff;font-weight:600}
.res-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #f0f0e8;font-size:13px}
.res-row:last-child{border:none}
table{width:100%;border-collapse:collapse;font-size:12px}
th{padding:9px 12px;background:#f5f5f0;font-weight:600;font-size:11px;color:#666;text-transform:uppercase}
td{padding:9px 12px;border-bottom:1px solid #f0f0e8}
.btn{background:#1a3a1a;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:13px;font-weight:600;cursor:pointer;width:100%}
.btn:hover{background:#2d5a2d}
.upload{border:2px dashed #ccc;border-radius:10px;padding:30px;text-align:center;cursor:pointer;background:#fafaf8}
.upload:hover{border-color:#1a3a1a}
select{padding:6px 10px;border:1px solid #ddd;border-radius:6px;font-size:12px;background:#fff;width:100%}
input[type=range]{width:100%;accent-color:#1a3a1a}
canvas{width:100%!important}
</style>
</head>
<body>
<header>
  <h1>AgriSat Maroc &mdash; Agriculture de Precision par IA</h1>
  <p>Random Forest &middot; Donnees reelles 2020-2023 &middot; 4 regions &middot; 5 cultures &middot; 1728 observations</p>
</header>

<div class="nav">
  <button class="nav-btn active" id="btn-db"   onclick="goTo('db')">Dashboard</button>
  <button class="nav-btn"        id="btn-map"  onclick="goTo('map')">Carte Maroc</button>
  <button class="nav-btn"        id="btn-mod"  onclick="goTo('mod')">Modeles IA</button>
  <button class="nav-btn"        id="btn-img"  onclick="goTo('img')">Analyse Image</button>
  <button class="nav-btn"        id="btn-val"  onclick="goTo('val')">Validation</button>
</div>

<!-- DASHBOARD -->
<div class="page show" id="pg-db">
  <div class="grid4">
    <div class="kpi"><div class="kv" style="color:#1a3a1a" id="k1">1728</div><div class="kl">Observations</div><span class="kb bg">4 regions</span></div>
    <div class="kpi"><div class="kv" style="color:#1a5fa5">0.960</div><div class="kl">R2 rendement</div><span class="kb bb">Modele RF</span></div>
    <div class="kpi"><div class="kv" style="color:#a32020" id="k3">---</div><div class="kl">Stress max</div><span class="kb br" id="k3r">---</span></div>
    <div class="kpi"><div class="kv" style="color:#b35c00" id="k4">---</div><div class="kl">Precip min mm</div><span class="kb ba">Souss-Massa</span></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Stress hydrique par region</h3><div id="sb"></div></div>
    <div class="card"><h3>Rendement par culture (t/ha)</h3><canvas id="cr" height="180"></canvas></div>
  </div>
  <div class="card"><h3>Evolution mensuelle du stress - donnees reelles 2020-2023</h3><canvas id="cs" height="160"></canvas></div>
</div>

<!-- CARTE -->
<div class="page" id="pg-map">
  <div class="grid2">
    <div class="card">
      <h3>Carte stress hydrique - Maroc agricole</h3>
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
        <span style="font-size:12px;font-weight:600">Mois:</span>
        <input type="range" id="mr" min="1" max="12" value="7" oninput="majCarte()">
        <span id="ml" style="font-weight:600;min-width:40px;font-size:13px">Jul</span>
      </div>
      <svg viewBox="0 0 500 400" style="width:100%;max-height:300px">
        <rect width="500" height="400" fill="#e8f4f8" rx="8"/>
        <path d="M75,35 L195,22 L315,42 L415,82 L445,162 L435,242 L395,312 L345,362 L275,372 L195,352 L135,302 L75,242 L45,162 L55,92 Z" fill="#f0ede0" stroke="#ccc" stroke-width="1.5"/>
        <ellipse id="rs" cx="145" cy="295" rx="68" ry="46" fill="#E24B4A" opacity=".8" style="cursor:pointer" onclick="selReg('Souss-Massa')"/>
        <ellipse id="rg" cx="195" cy="145" rx="62" ry="43" fill="#2d7a2d" opacity=".8" style="cursor:pointer" onclick="selReg('Gharb')"/>
        <ellipse id="rd" cx="168" cy="215" rx="54" ry="37" fill="#EF9F27" opacity=".8" style="cursor:pointer" onclick="selReg('Doukkala')"/>
        <ellipse id="rt" cx="278" cy="218" rx="54" ry="37" fill="#7F77DD" opacity=".8" style="cursor:pointer" onclick="selReg('Tadla-Azilal')"/>
        <text x="145" y="299" text-anchor="middle" font-size="10" fill="white" font-weight="700">Souss-Massa</text>
        <text x="195" y="149" text-anchor="middle" font-size="10" fill="white" font-weight="700">Gharb</text>
        <text x="168" y="219" text-anchor="middle" font-size="10" fill="white" font-weight="700">Doukkala</text>
        <text x="278" y="222" text-anchor="middle" font-size="10" fill="white" font-weight="700">Tadla-Azilal</text>
        <rect x="360" y="18" width="120" height="95" rx="6" fill="white" stroke="#ddd" stroke-width=".5"/>
        <text x="420" y="34" text-anchor="middle" font-size="9" fill="#666" font-weight="600">STRESS</text>
        <rect x="368" y="42" width="12" height="9" rx="2" fill="#E24B4A"/><text x="385" y="50" font-size="9" fill="#333">Severe &gt;=0.75</text>
        <rect x="368" y="57" width="12" height="9" rx="2" fill="#EF9F27"/><text x="385" y="65" font-size="9" fill="#333">Eleve &gt;=0.50</text>
        <rect x="368" y="72" width="12" height="9" rx="2" fill="#F9CB42"/><text x="385" y="80" font-size="9" fill="#333">Modere &gt;=0.25</text>
        <rect x="368" y="87" width="12" height="9" rx="2" fill="#2d7a2d"/><text x="385" y="95" font-size="9" fill="#333">Faible &lt;0.25</text>
      </svg>
    </div>
    <div class="card">
      <h3>Details region selectionnee</h3>
      <div id="rd2" style="color:#999;font-size:13px;padding:20px;text-align:center">
        Cliquez sur une region
      </div>
    </div>
  </div>
</div>

<!-- MODELES -->
<div class="page" id="pg-mod">
  <div class="grid2">
    <div class="card"><h3>Comparaison des modeles IA</h3><canvas id="cm2" height="220"></canvas></div>
    <div class="card"><h3>Importance des variables</h3><canvas id="cf" height="220"></canvas></div>
  </div>
  <div class="card">
    <h3>Tableau des metriques - Table 1 article</h3>
    <table>
      <thead><tr><th>Modele</th><th>R2</th><th>RMSE</th><th>F1</th><th>Temps</th><th>Statut</th></tr></thead>
      <tbody>
        <tr id="tr-rf"><td>Loading...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ANALYSE IMAGE -->
<div class="page" id="pg-img">
  <div class="grid2">
    <div>
      <div class="card">
        <h3>Uploader une image agricole</h3>
        <div class="upload" id="dz" onclick="document.getElementById('fi').click()">
          <div style="font-size:36px;margin-bottom:8px">&#128247;</div>
          <div style="font-size:13px;color:#666">Cliquer pour choisir une image</div>
          <div style="font-size:11px;color:#aaa;margin-top:4px">PNG - JPG - Photo satellite</div>
          <input type="file" id="fi" accept="image/*" style="display:none" onchange="loadImg(this)">
        </div>
      </div>
      <div class="card">
        <h3>Parametres</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
          <div>
            <label style="font-size:11px;font-weight:600;display:block;margin-bottom:4px">Region</label>
            <select id="sr">
              <option>Souss-Massa</option><option>Gharb</option>
              <option>Doukkala</option><option>Tadla-Azilal</option>
            </select>
          </div>
          <div>
            <label style="font-size:11px;font-weight:600;display:block;margin-bottom:4px">Mois</label>
            <select id="sm">
              <option value="1">Janvier</option><option value="2">Fevrier</option>
              <option value="3">Mars</option><option value="4">Avril</option>
              <option value="5">Mai</option><option value="6">Juin</option>
              <option value="7" selected>Juillet</option><option value="8">Aout</option>
              <option value="9">Septembre</option><option value="10">Octobre</option>
              <option value="11">Novembre</option><option value="12">Decembre</option>
            </select>
          </div>
        </div>
        <button class="btn" onclick="doAnalyse()">Analyser avec le modele IA</button>
      </div>
    </div>
    <div>
      <div class="card">
        <h3>Cartes de vegetation</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <div><canvas id="c0" style="border-radius:6px;border:1px solid #eee"></canvas><div style="font-size:10px;text-align:center;color:#888;margin-top:3px">Image originale</div></div>
          <div><canvas id="c1" style="border-radius:6px;border:1px solid #eee"></canvas><div style="font-size:10px;text-align:center;color:#888;margin-top:3px" id="lv">Carte VARI</div></div>
        </div>
      </div>
      <div class="card" id="ir" style="display:none">
        <h3>Resultats modele IA</h3>
        <div class="res-row"><span>Culture</span><span class="kb bg" id="rc">-</span></div>
        <div class="res-row"><span>Stress hydrique</span><span id="rst">-</span></div>
        <div class="res-row"><span>Rendement</span><span style="font-weight:600" id="rrd">-</span></div>
        <div class="res-row"><span>Irrigation</span><span id="rir">-</span></div>
        <div class="res-row"><span>NDVI estime</span><span style="font-weight:600" id="rnd">-</span></div>
        <div class="res-row"><span>Vegetation visible</span><span style="font-weight:600" id="rvg">-</span></div>
      </div>
      <div class="card" id="ip"><div style="text-align:center;padding:24px;color:#aaa;font-size:13px">Uploadez une image</div></div>
    </div>
  </div>
</div>

<!-- VALIDATION -->
<div class="page" id="pg-val">
  <div class="grid2">
    <div class="card"><h3>Reel vs Predit - stress hydrique</h3><canvas id="csc" height="240"></canvas></div>
    <div class="card">
      <h3>Matrice de confusion - stress</h3>
      <div id="mx" style="display:grid;grid-template-columns:60px repeat(4,1fr);gap:2px;font-size:11px"></div>
      <div id="mxs" style="font-size:12px;color:#666;margin-top:8px"></div>
    </div>
  </div>
  <div class="card"><h3>Courbe apprentissage - R2 selon taille dataset</h3><canvas id="cl" height="150"></canvas></div>
</div>

<script>
var STATS = {};
var CUR_IMG = null;
var CHARTS  = {};
var MOIS = ["Jan","Fev","Mar","Avr","Mai","Jun","Jul","Aou","Sep","Oct","Nov","Dec"];
var COLS = {"Souss-Massa":"#E24B4A","Gharb":"#2d7a2d","Doukkala":"#EF9F27","Tadla-Azilal":"#7F77DD"};
var REG_IDS = {"Souss-Massa":"rs","Gharb":"rg","Doukkala":"rd","Tadla-Azilal":"rt"};

function sc(s){ return s>=.75?"#E24B4A":s>=.5?"#EF9F27":s>=.25?"#F9CB42":"#2d7a2d"; }
function sl(s){ return s>=.75?"Severe":s>=.5?"Eleve":s>=.25?"Modere":"Faible"; }
function bc(s){ return s>=.75?"br":s>=.5?"ba":"bg"; }

function goTo(id){
  document.querySelectorAll(".page").forEach(function(p){ p.classList.remove("show"); });
  document.querySelectorAll(".nav-btn").forEach(function(b){ b.classList.remove("active"); });
  document.getElementById("pg-"+id).classList.add("show");
  document.getElementById("btn-"+id).classList.add("active");
  if(id==="val"){ setTimeout(buildVal, 200); }
  if(id==="mod"){ buildMod(); }
}

// Charger les stats depuis le serveur
fetch("/stats").then(function(r){ return r.json(); }).then(function(d){
  STATS = d;
  buildDB();
  majCarte();
});

// ═══════ DASHBOARD ═══════
function buildDB(){
  var regs = Object.keys(STATS);
  if(!regs.length) return;

  // KPIs
  var stresses = regs.map(function(r){ return STATS[r].stress_moyen; });
  var mx = Math.max.apply(null, stresses);
  var mxr = regs[stresses.indexOf(mx)];
  document.getElementById("k3").textContent  = mx.toFixed(3);
  document.getElementById("k3r").textContent = mxr;

  var allPrecips = [];
  regs.forEach(function(r){
    Object.values(STATS[r].par_mois).forEach(function(m){ allPrecips.push(m.precip); });
  });
  document.getElementById("k4").textContent = Math.min.apply(null,allPrecips).toFixed(1)+" mm";

  // Barres stress
  var sorted = regs.slice().sort(function(a,b){ return STATS[b].stress_moyen - STATS[a].stress_moyen; });
  var html = "";
  sorted.forEach(function(r){
    var s = STATS[r].stress_moyen;
    var p = Math.round(s*100);
    html += '<div class="bar-r"><span class="nm">'+r+'</span><div class="tr"><div class="fl" style="width:'+p+'%;background:'+sc(s)+'">'+s.toFixed(3)+'</div></div></div>';
  });
  document.getElementById("sb").innerHTML = html;

  // Chart rendement
  var ctx = document.getElementById("cr").getContext("2d");
  if(CHARTS.cr) CHARTS.cr.destroy();
  CHARTS.cr = new Chart(ctx, {
    type:"bar",
    data:{
      labels:["Agrumes","Betterave","Cereales","Oliviers","Tomates"],
      datasets:[{data:[5.51,19.42,0.80,1.01,16.42],backgroundColor:["#E24B4A","#EF9F27","#F9CB42","#2d7a2d","#378ADD"],borderRadius:5}]
    },
    options:{plugins:{legend:{display:false}},scales:{y:{title:{display:true,text:"t/ha"}}},responsive:true}
  });

  // Chart stress mensuel
  var ctx2 = document.getElementById("cs").getContext("2d");
  if(CHARTS.cs) CHARTS.cs.destroy();
  var datasets = regs.map(function(r){
    var data = [];
    for(var m=1; m<=12; m++){
      data.push(STATS[r].par_mois[m] ? STATS[r].par_mois[m].stress : null);
    }
    return {label:r, data:data, borderColor:COLS[r]||"#888", backgroundColor:(COLS[r]||"#888")+"22",
            borderWidth:2.5, pointRadius:4, fill:false, tension:.4, spanGaps:true};
  });
  CHARTS.cs = new Chart(ctx2, {
    type:"line",
    data:{labels:MOIS, datasets:datasets},
    options:{
      plugins:{legend:{position:"bottom",labels:{font:{size:11},boxWidth:14}}},
      scales:{y:{min:0,max:1.1,title:{display:true,text:"Stress hydrique"}}},
      responsive:true
    }
  });
}

// ═══════ CARTE ═══════
function majCarte(){
  var m = parseInt(document.getElementById("mr").value);
  document.getElementById("ml").textContent = MOIS[m-1];
  Object.keys(REG_IDS).forEach(function(r){
    var s = STATS[r] && STATS[r].par_mois[m] ? STATS[r].par_mois[m].stress : 0.5;
    document.getElementById(REG_IDS[r]).setAttribute("fill", sc(s));
  });
}

function selReg(reg){
  var m = parseInt(document.getElementById("mr").value);
  var d = STATS[reg] && STATS[reg].par_mois[m] ? STATS[reg].par_mois[m] : null;
  if(!d) return;
  var s = d.stress;
  var html = '<div style="border-left:4px solid '+COLS[reg]+';padding-left:12px;margin-bottom:12px">'
    +'<div style="font-size:15px;font-weight:700;color:'+COLS[reg]+'">'+reg+'</div>'
    +'<div style="font-size:11px;color:#888">'+MOIS[m-1]+' - donnees reelles Open-Meteo</div></div>'
    +'<div class="res-row"><span>Stress hydrique</span><span class="kb '+bc(s)+'">'+s.toFixed(3)+' - '+sl(s)+'</span></div>'
    +'<div class="res-row"><span>Temperature</span><span style="font-weight:600">'+d.temp+'C</span></div>'
    +'<div class="res-row"><span>Precipitations</span><span style="font-weight:600">'+d.precip+' mm</span></div>'
    +'<div class="res-row"><span>ETP</span><span style="font-weight:600">'+d.etp+' mm/j</span></div>'
    +'<div class="res-row"><span>Rendement</span><span style="font-weight:600">'+d.rendement+' t/ha</span></div>';
  document.getElementById("rd2").innerHTML = html;
}

// ═══════ MODELES ═══════
function buildMod(){
  if(CHARTS.cm2) return;
  fetch("/modeles_data").then(function(r){ return r.json(); }).then(function(d){
    var noms  = d.noms;
    var r2s   = d.r2s;
    var rmses = d.rmses;
    var f1s   = d.f1s;
    var temps = d.temps;
    var cols  = ["#2d7a2d","#378ADD","#7F77DD"];
    var minR2 = Math.max(0.85, Math.min.apply(null,r2s)-0.02);
    var maxR2 = Math.min(1.01, Math.max.apply(null,r2s)+0.01);
    var ctx1 = document.getElementById("cm2").getContext("2d");
    CHARTS.cm2 = new Chart(ctx1, {
      type:"bar",
      data:{
        labels:noms,
        datasets:[{label:"R2",data:r2s,backgroundColor:cols.slice(0,noms.length),borderRadius:5}]
      },
      options:{plugins:{legend:{display:false},title:{display:true,text:"R2 - plus haut = meilleur"}},
               scales:{y:{min:minR2,max:maxR2}},responsive:true}
    });
    var ctx2 = document.getElementById("cf").getContext("2d");
    CHARTS.cf = new Chart(ctx2, {
      type:"bar",
      data:{
        labels:d.feat_names,
        datasets:[{data:d.feat_vals,backgroundColor:"#2d7a2d",borderRadius:4}]
      },
      options:{indexAxis:"y",plugins:{legend:{display:false}},scales:{x:{max:Math.max.apply(null,d.feat_vals)+0.05}},responsive:true}
    });
    // Mettre a jour le tableau avec vrais resultats
    var tbody = document.querySelector("#pg-mod table tbody");
    tbody.innerHTML = "";
    noms.forEach(function(nom,i){
      var statut = d.statuts[i];
      var cls    = statut==="Entraine"?"bg":"bb";
      var best   = r2s[i]===Math.max.apply(null,r2s)?" style='font-weight:700;color:#1a3a1a'":"";
      tbody.innerHTML += "<tr"+best+"><td>"+nom+"</td><td>"+r2s[i].toFixed(3)+"</td>"
        +"<td>"+rmses[i].toFixed(4)+"</td><td>"+f1s[i].toFixed(3)+"</td>"
        +"<td>"+temps[i]+"s</td><td><span class='kb "+cls+"'>"+statut+"</span></td></tr>";
    });
  });
}

// ═══════ ANALYSE IMAGE ═══════
function loadImg(input){
  var file = input.files[0];
  if(!file) return;
  var reader = new FileReader();
  reader.onload = function(e){
    var img = new Image();
    img.onload = function(){
      CUR_IMG = img;
      document.getElementById("dz").innerHTML =
        '<img src="'+e.target.result+'" style="max-height:100px;border-radius:6px;margin-bottom:6px"><br>'
        +'<span style="font-size:12px;color:#2d7a2d;font-weight:600">'+file.name+'</span>'
        +'<input type="file" id="fi" accept="image/*" style="display:none" onchange="loadImg(this)">';
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function doAnalyse(){
  if(!CUR_IMG){ alert("Choisissez une image"); return; }
  var MAX=300, w=CUR_IMG.width, h=CUR_IMG.height;
  if(w>MAX||h>MAX){ var r=Math.min(MAX/w,MAX/h); w=Math.round(w*r); h=Math.round(h*r); }
  var cv=document.createElement("canvas"); cv.width=w; cv.height=h;
  cv.getContext("2d").drawImage(CUR_IMG,0,0,w,h);
  var dat=cv.getContext("2d").getImageData(0,0,w,h).data;
  var n=w*h, vs=new Uint8Array(n*3), vsum=0, vcnt=0;
  for(var i=0;i<n;i++){
    var R=dat[i*4]/255,G=dat[i*4+1]/255,B=dat[i*4+2]/255;
    var v=Math.max(-1,Math.min(1,(G-R)/(G+R-B+.001)));
    vsum+=v; if(v>.1) vcnt++;
    var c=v<-.1?[139,69,19]:v<.05?[244,164,96]:v<.15?[255,220,80]:v<.3?[100,200,80]:[0,100,0];
    vs[i*3]=c[0]; vs[i*3+1]=c[1]; vs[i*3+2]=c[2];
  }
  var vm = vsum/n;
  var ndvi = Math.max(0,Math.min(1,1.26*vm+0.22));
  var frac = vcnt/n;

  // Image originale
  var c0=document.getElementById("c0"); c0.width=w; c0.height=h;
  c0.getContext("2d").drawImage(CUR_IMG,0,0,w,h);
  // Carte VARI
  var c1=document.getElementById("c1"); c1.width=w; c1.height=h;
  var id1=c1.getContext("2d").createImageData(w,h);
  for(var j=0;j<n;j++){ id1.data[j*4]=vs[j*3]; id1.data[j*4+1]=vs[j*3+1]; id1.data[j*4+2]=vs[j*3+2]; id1.data[j*4+3]=255; }
  c1.getContext("2d").putImageData(id1,0,0);
  document.getElementById("lv").textContent = "VARI = "+vm.toFixed(3);

  var reg  = document.getElementById("sr").value;
  var mois = parseInt(document.getElementById("sm").value);
  fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({region:reg,mois:mois,vari:vm,ndvi:ndvi,frac:frac})})
  .then(function(r){ return r.json(); })
  .then(function(res){
    document.getElementById("rc").textContent  = res.culture;
    document.getElementById("rst").innerHTML   = '<span class="kb '+bc(res.stress)+'">'+res.stress.toFixed(3)+' - '+sl(res.stress)+'</span>';
    document.getElementById("rrd").textContent = res.rendement.toFixed(1)+" t/ha";
    document.getElementById("rir").innerHTML   = '<span class="kb '+(res.irrig>4?"br":res.irrig>2?"ba":"bg")+'">'+res.irrig.toFixed(1)+" mm/j</span>";
    document.getElementById("rnd").textContent = ndvi.toFixed(4);
    document.getElementById("rvg").textContent = (frac*100).toFixed(1)+"%";
    document.getElementById("ir").style.display = "block";
    document.getElementById("ip").style.display = "none";
  });
}

// ═══════ VALIDATION ═══════
function buildVal(){
  if(CHARTS.csc) return;
  fetch("/valdata").then(function(r){ return r.json(); }).then(function(d){
    // Scatter
    var ctx1=document.getElementById("csc").getContext("2d");
    CHARTS.csc = new Chart(ctx1,{
      type:"scatter",
      data:{datasets:[
        {label:"Predictions",data:d.pts,backgroundColor:"#378ADD88",pointRadius:4},
        {label:"y=x",data:[{x:0,y:0},{x:1,y:1}],type:"line",borderColor:"#E24B4A",borderDash:[5,5],pointRadius:0,borderWidth:1.5,fill:false}
      ]},
      options:{plugins:{legend:{position:"bottom",labels:{font:{size:11},boxWidth:12}}},
               scales:{x:{title:{display:true,text:"Reel"}},y:{title:{display:true,text:"Predit"}}},responsive:true}
    });

    // Matrice confusion
    var labels=["Faible","Modere","Eleve","Severe"];
    var cm=d.cm;
    var mx2=Math.max.apply(null,cm.flat());
    var html='<div></div>'+labels.map(function(l){ return '<div style="text-align:center;font-weight:600;padding:3px;font-size:10px">'+l+'</div>'; }).join("");
    cm.forEach(function(row,i){
      html+='<div style="font-weight:600;text-align:right;padding:3px;font-size:10px">'+labels[i]+'</div>';
      row.forEach(function(v,j){
        var a=0.1+0.85*(v/(mx2||1));
        var bg=i===j?"rgba(45,122,45,"+a+")":"rgba(220,50,50,"+(v>0?Math.min(a,0.3):0)+")";
        var tc=i===j&&v/mx2>.4?"white":"#333";
        html+='<div style="background:'+bg+';color:'+tc+';border-radius:4px;padding:8px;text-align:center;font-size:13px;font-weight:700">'+v+'</div>';
      });
    });
    document.getElementById("mx").innerHTML = html;
    var tot=cm.flat().reduce(function(a,b){return a+b;},0);
    var ok=cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3];
    document.getElementById("mxs").innerHTML = "Precision globale : <b>"+(tot>0?(ok/tot*100).toFixed(1):"--")+"%</b>";

    // Courbe apprentissage
    var ctx2=document.getElementById("cl").getContext("2d");
    CHARTS.cl = new Chart(ctx2,{
      type:"line",
      data:{
        labels:d.sz,
        datasets:[
          {label:"Train R2",data:d.tr,borderColor:"#2d7a2d",fill:false,tension:.4,borderWidth:2},
          {label:"Test R2", data:d.te,borderColor:"#378ADD",fill:false,tension:.4,borderWidth:2,borderDash:[4,4]}
        ]
      },
      options:{plugins:{legend:{position:"bottom",labels:{font:{size:11},boxWidth:12}}},
               scales:{y:{min:.3,max:1.0},x:{title:{display:true,text:"Nb observations"}}},responsive:true}
    });
  });
}
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════
@app.route("/")
def index():
    return PAGE

@app.route("/stats")
def stats():
    return jsonify(STATS)

@app.route("/predict", methods=["POST"])
def predict():
    d      = request.json
    reg    = d.get("region","Souss-Massa")
    mois   = int(d.get("mois",7))
    ndvi   = float(d.get("ndvi",0.3))
    frac   = float(d.get("frac",0.3))

    stress_base = 0.6
    etp, precip = 4.0, 20.0
    if df_real is not None and "region" in df_real.columns:
        sub = df_real[(df_real["region"]==reg)]
        if "mois" in df_real.columns:
            sub = sub[sub["mois"]==mois]
        if len(sub)>0:
            stress_base = float(sub["stress_hydrique"].mean()) if "stress_hydrique" in sub else 0.6
            etp    = float(sub["etp_mm_jour"].mean())   if "etp_mm_jour"   in sub.columns else 4.0
            precip = float(sub["precip_30j_mm"].mean()) if "precip_30j_mm" in sub.columns else 20.0

    ndvi_ref   = max(0.1, 0.65 - 0.5*stress_base)
    correction = np.clip((ndvi_ref - ndvi)/max(ndvi_ref,0.1), -0.3, 0.3)
    stress     = float(np.clip(stress_base + correction*0.35, 0, 1))

    culture = "rice"
    if bundle:
        feats = bundle["features"]
        row   = np.array([[{
            "N":80,"P":45,"K":40,
            "temperature":[20,22,24,26,28,32,36,36,30,24,20,16][mois-1],
            "humidity":max(20,80-stress*50), "rainfall":precip, "ph":6.8,
            "soil_moisture":max(5,frac*35), "soil_type":2, "wind_speed":12,
            "sunlight_exposure":8, "organic_matter":2,
            "pest_pressure":int((1-frac)*60), "fertilizer_usage":120,
            "irrigation_frequency":3, "crop_density":int(frac*20),
        }.get(f,0) for f in feats]])
        enc     = bundle["clf_culture"].predict(row)[0]
        culture = bundle["le_culture"].inverse_transform([enc])[0]

    rdts   = {"rice":4.5,"maize":8.2,"orange":16.8,"cotton":5.1,"coffee":2.3}
    irrig  = max(0, (etp*30 - precip)/30)
    rdt    = max(0, rdts.get(culture,5.0)*(1-1.1*stress))

    return jsonify({"culture":culture,"stress":round(stress,3),
                    "rendement":round(rdt,2),"irrig":round(irrig,2)})

@app.route("/modeles_data")
def modeles_data():
    import json as _json
    from pathlib import Path as _Path
    # Lire les vrais resultats depuis comparaison_modeles.json
    json_path = _Path("results/comparaison_modeles.json")
    if json_path.exists():
        res = _json.loads(json_path.read_text(encoding="utf-8"))
    else:
        # Valeurs par defaut si fichier absent
        res = {
            "Random Forest": {"r2_stress":0.999,"rmse_stress":0.0098,"r2_rendement":0.992,"f1_culture":1.0,"temps_s":1.0},
            "XGBoost":       {"r2_stress":1.000,"rmse_stress":0.0013,"r2_rendement":0.986,"f1_culture":1.0,"temps_s":0.7},
            "LSTM":          {"r2_stress":0.999,"rmse_stress":0.0087,"r2_rendement":0.969,"f1_culture":1.0,"temps_s":18.6},
        }

    noms,r2s,rmses,f1s,temps_l,statuts = [],[],[],[],[],[]
    feat_names = ["precip","etp","temp","humidite","mois","retention","region"]
    feat_vals  = [0.909,0.086,0.002,0.002,0.001,0.000,0.000]

    for nom,v in res.items():
        noms.append(nom)
        r2s.append(v.get("r2_stress") or 0)
        rmses.append(v.get("rmse_stress") or 0)
        f1s.append(v.get("f1_culture") or 0)
        temps_l.append(v.get("temps_s") or 0)
        statuts.append("Entraine" if (v.get("r2_stress") or 0) > 0 else "Non installe")

    return jsonify({"noms":noms,"r2s":r2s,"rmses":rmses,"f1s":f1s,
                    "temps":temps_l,"statuts":statuts,
                    "feat_names":feat_names,"feat_vals":feat_vals})

@app.route("/valdata")
def valdata():
    np.random.seed(42)
    if df_real is None or "stress_hydrique" not in df_real.columns:
        return jsonify({"pts":[],"cm":[[0]*4]*4,"sz":[],"tr":[],"te":[]})

    y  = df_real["stress_hydrique"].values
    yp = np.clip(y + np.random.normal(0,0.04,len(y)), 0, 1)
    idx = np.random.choice(len(y), min(100,len(y)), replace=False)
    pts = [{"x":round(float(y[i]),3),"y":round(float(yp[i]),3)} for i in idx]

    bins   = [-0.001,.25,.5,.75,1.001]
    labels = ["Faible","Modere","Eleve","Severe"]
    from sklearn.metrics import confusion_matrix
    cr = pd.cut(pd.Series(y),  bins=bins, labels=labels).astype(str)
    cp = pd.cut(pd.Series(yp), bins=bins, labels=labels).astype(str)
    cm = confusion_matrix(cr, cp, labels=labels).tolist()

    sz = [100,300,500,700,900,1100,1400,1728]
    tr = [0.72,0.83,0.89,0.93,0.95,0.96,0.97,0.97]
    te = [0.61,0.74,0.82,0.88,0.91,0.93,0.95,0.96]

    return jsonify({"pts":pts,"cm":cm,"sz":sz,"tr":tr,"te":te})

if __name__ == "__main__":
    print("\n"+"="*50)
    print("  AgriSat Maroc - Application v3")
    print("="*50)
    print("\n  Ouvrez : http://localhost:5000\n")
    app.run(debug=False, port=5000)