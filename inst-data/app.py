"""
app.py — Application AgriSat Maroc
Lancer : python app.py
Puis ouvrir : http://localhost:5000
"""
import pickle
import numpy as np
import base64
import io
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Charger le modèle ─────────────────────────────
MODEL_PATH = Path("models/modele_agrisat.pkl")
bundle = None

if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    print(f"✅ Modèle chargé — cultures : {bundle['cultures']}")
else:
    print("⚠️  Modèle non trouvé. Lancez d'abord : python train.py")

# ── Page HTML complète ────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgriSat Maroc</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f5f5f0; color: #1a1a1a; }
  header { background: #1a3a1a; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; }
  header span { font-size: 12px; opacity: 0.7; }
  .container { max-width: 900px; margin: 24px auto; padding: 0 16px; }
  .card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px; border: 1px solid #e0e0d8; }
  .card h2 { font-size: 14px; font-weight: 600; color: #555; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.05em; }
  .upload-zone { border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.2s; background: #fafaf8; }
  .upload-zone:hover, .upload-zone.drag { border-color: #2d7a2d; background: #f0f7f0; }
  .upload-zone input { display: none; }
  .upload-icon { font-size: 40px; margin-bottom: 10px; }
  .upload-text { font-size: 15px; color: #555; }
  .upload-hint { font-size: 12px; color: #999; margin-top: 6px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 14px; }
  label { font-size: 12px; font-weight: 500; color: #666; display: block; margin-bottom: 4px; }
  select { width: 100%; padding: 8px 10px; border: 1px solid #ddd; border-radius: 8px; font-size: 13px; background: white; }
  select:focus { outline: none; border-color: #2d7a2d; }
  .btn { width: 100%; padding: 12px; background: #2d7a2d; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; }
  .btn:hover { background: #1f5c1f; }
  .btn:disabled { background: #ccc; cursor: not-allowed; }
  .results { display: none; }
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }
  .kpi { background: #f8f8f5; border: 1px solid #e8e8e0; border-radius: 10px; padding: 12px; text-align: center; }
  .kpi-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .kpi-val { font-size: 22px; font-weight: 700; }
  .kpi-badge { font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 20px; margin-top: 4px; display: inline-block; }
  .canvas-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px; }
  .canvas-wrap { border-radius: 8px; overflow: hidden; border: 1px solid #e0e0d8; }
  canvas { width: 100%; display: block; }
  .canvas-label { font-size: 11px; text-align: center; padding: 5px; background: #f5f5f0; color: #666; font-weight: 500; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .bar-name { font-size: 11px; color: #666; width: 130px; text-align: right; flex-shrink: 0; }
  .bar-track { flex: 1; height: 14px; background: #eee; border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; font-size: 10px; color: white; display: flex; align-items: center; padding-left: 6px; transition: width 0.6s ease; }
  .rec { border-left: 3px solid; padding: 8px 10px; border-radius: 0 6px 6px 0; margin-bottom: 6px; font-size: 12px; }
  .badge-green { background: #e8f5e8; color: #2d6a2d; }
  .badge-amber { background: #fff3e0; color: #b35c00; }
  .badge-red   { background: #fce8e8; color: #a32020; }
  .model-info { font-size: 11px; color: #999; text-align: center; margin-top: 8px; }
  #preview-img { max-height: 120px; border-radius: 8px; margin-bottom: 8px; }
</style>
</head>
<body>

<header>
  <div class="upload-icon" style="font-size:24px">🛰</div>
  <div>
    <h1>AgriSat Maroc — Analyse d'image agricole</h1>
    <span>Modèle IA entraîné sur données réelles · Random Forest</span>
  </div>
</header>

<div class="container">

  <!-- Upload -->
  <div class="card">
    <h2>1 — Uploader une image de champ agricole</h2>
    <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div id="upload-content">
        <div class="upload-icon">📷</div>
        <div class="upload-text">Cliquer ou glisser une image ici</div>
        <div class="upload-hint">PNG · JPG · Photo satellite · Google Maps · Photo terrain</div>
      </div>
      <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
    </div>
  </div>

  <!-- Paramètres -->
  <div class="card">
    <h2>2 — Paramètres de la parcelle</h2>
    <div class="grid3">
      <div>
        <label>Région</label>
        <select id="region">
          <option value="souss">Souss-Massa</option>
          <option value="gharb">Gharb</option>
          <option value="doukkala">Doukkala</option>
          <option value="tadla">Tadla-Azilal</option>
        </select>
      </div>
      <div>
        <label>Mois</label>
        <select id="mois">
          <option value="1">Janvier</option><option value="2">Février</option>
          <option value="3">Mars</option><option value="4">Avril</option>
          <option value="5">Mai</option><option value="6">Juin</option>
          <option value="7" selected>Juillet</option><option value="8">Août</option>
          <option value="9">Septembre</option><option value="10">Octobre</option>
          <option value="11">Novembre</option><option value="12">Décembre</option>
        </select>
      </div>
      <div>
        <label>Culture</label>
        <select id="culture">
          <option value="rice">Riz / Céréales</option>
          <option value="maize">Maïs</option>
          <option value="orange">Agrumes</option>
          <option value="cotton">Coton / Betterave</option>
          <option value="coffee">Oliviers / Café</option>
        </select>
      </div>
    </div>
    <button class="btn" id="btnAnalyse" onclick="analyser()" disabled>
      🔬 Analyser avec le modèle IA
    </button>
    <div class="model-info" id="model-info">Modèle : F1 culture = 0.993 · R² rendement = 0.960</div>
  </div>

  <!-- Résultats -->
  <div class="results" id="results">

    <!-- Cartes colorées -->
    <div class="card">
      <h2>Analyse visuelle de l'image</h2>
      <div class="canvas-grid">
        <div class="canvas-wrap">
          <canvas id="cvOrig"></canvas>
          <div class="canvas-label">Image originale</div>
        </div>
        <div class="canvas-wrap">
          <canvas id="cvVARI"></canvas>
          <div class="canvas-label" id="lblVARI">Carte VARI</div>
        </div>
        <div class="canvas-wrap">
          <canvas id="cvNDVI"></canvas>
          <div class="canvas-label" id="lblNDVI">NDVI estimé</div>
        </div>
      </div>
    </div>

    <!-- KPIs -->
    <div class="card">
      <h2>Prédictions du modèle IA</h2>
      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-label">Culture recommandée</div>
          <div class="kpi-val" id="kCulture" style="font-size:16px">—</div>
          <span class="kpi-badge" id="bCulture" style="background:#e8f5e8;color:#2d6a2d"></span>
        </div>
        <div class="kpi">
          <div class="kpi-label">Stress hydrique</div>
          <div class="kpi-val" id="kStress">—</div>
          <span class="kpi-badge" id="bStress"></span>
        </div>
        <div class="kpi">
          <div class="kpi-label">Rendement estimé</div>
          <div class="kpi-val" id="kRdt">—</div>
          <span class="kpi-badge" style="background:#e8f0ff;color:#1a3a8f">t/ha estimé</span>
        </div>
        <div class="kpi">
          <div class="kpi-label">Irrigation requise</div>
          <div class="kpi-val" id="kIrrig">—</div>
          <span class="kpi-badge" id="bIrrig"></span>
        </div>
      </div>

      <!-- Feature importance -->
      <h2 style="margin-bottom:10px">Variables les plus importantes</h2>
      <div id="features"></div>

      <!-- Recommandations -->
      <h2 style="margin-top:14px;margin-bottom:8px">Recommandations agronomiques</h2>
      <div id="recos"></div>
    </div>

  </div>
</div>

<script>
// Données climatiques réelles par région
const REGIONS = {
  souss:{etp:[75,96,123,156,195,234,247,237,183,132,90,72],precip:[28,22,18,10,4,1,0,1,8,18,28,30],sol:"Sableux-limoneux",ret:0.20},
  gharb:{etp:[54,72,105,144,180,216,234,219,156,102,60,48],precip:[60,55,48,38,22,8,2,4,22,45,62,68],sol:"Argileux",ret:0.38},
  doukkala:{etp:[60,84,114,150,186,222,237,225,168,114,69,54],precip:[45,40,32,22,12,3,1,2,14,32,48,50],sol:"Argilo-limoneux",ret:0.32},
  tadla:{etp:[66,90,126,165,204,237,255,243,180,123,78,60],precip:[38,32,28,20,14,5,2,3,18,32,40,42],sol:"Limoneux",ret:0.28}
};

let currentImg = null;

const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e=>{e.preventDefault();dz.classList.add('drag')});
dz.addEventListener('dragleave', ()=>dz.classList.remove('drag'));
dz.addEventListener('drop', e=>{e.preventDefault();dz.classList.remove('drag');handleFile(e.dataTransfer.files[0])});

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      currentImg = img;
      document.getElementById('upload-content').innerHTML =
        `<img src="${e.target.result}" id="preview-img"><br>
         <div class="upload-text" style="color:#2d7a2d">✅ ${file.name} — ${img.width}×${img.height}px</div>
         <div class="upload-hint">Cliquer pour changer l'image</div>`;
      document.getElementById('btnAnalyse').disabled = false;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function getPixelData(img) {
  const MAX = 350;
  let w = img.width, h = img.height;
  if (w > MAX || h > MAX) { const r = Math.min(MAX/w, MAX/h); w = Math.round(w*r); h = Math.round(h*r); }
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  cv.getContext('2d').drawImage(img, 0, 0, w, h);
  return { data: cv.getContext('2d').getImageData(0,0,w,h).data, w, h };
}

function analyser() {
  if (!currentImg) return;
  document.getElementById('btnAnalyse').textContent = '⏳ Analyse en cours...';
  document.getElementById('btnAnalyse').disabled = true;

  const regKey = document.getElementById('region').value;
  const mois   = parseInt(document.getElementById('mois').value);
  const reg    = REGIONS[regKey];
  const m      = mois - 1;

  // Extraire pixels
  const {data, w, h} = getPixelData(currentImg);
  const n = w * h;
  const vari_arr = new Float32Array(n);
  const ndvi_arr = new Float32Array(n);
  let vari_sum = 0, veg_count = 0;
  let r_sum=0, g_sum=0, b_sum=0;

  for (let i = 0; i < n; i++) {
    const R=data[i*4]/255, G=data[i*4+1]/255, B=data[i*4+2]/255;
    const dv = G+R-B+0.001;
    const vari = Math.max(-1, Math.min(1, (G-R)/dv));
    const ndvi = Math.max(0, Math.min(1, 1.26*vari+0.22));
    vari_arr[i] = vari; ndvi_arr[i] = ndvi;
    vari_sum += vari; r_sum+=R; g_sum+=G; b_sum+=B;
    if (vari > 0.1) veg_count++;
  }

  const vari_mean = vari_sum / n;
  const ndvi_mean = parseFloat((1.26*vari_mean+0.22).toFixed(4));
  const frac_veg  = veg_count / n;

  // Données climatiques région
  const etp    = reg.etp[m];
  const precip = reg.precip[m];
  const deficit = Math.max(0, etp - precip);

  // Paramètres à envoyer au modèle IA
  const params = {
    N: 80, P: 45, K: 40,
    temperature: [20,22,24,26,28,32,36,36,30,24,20,16][m],
    humidity: Math.round(40 + frac_veg * 40),
    rainfall: precip,
    ph: 6.8,
    soil_moisture: Math.round(10 + frac_veg * 30),
    soil_type: {souss:2,gharb:1,doukkala:2,tadla:3}[regKey] || 2,
    wind_speed: 12,
    sunlight_exposure: 8,
    organic_matter: reg.ret * 10,
    pest_pressure: Math.round((1 - frac_veg) * 60),
    fertilizer_usage: 120,
    irrigation_frequency: deficit > 100 ? 5 : 2,
    crop_density: Math.round(frac_veg * 20),
    vari: vari_mean,
    ndvi: ndvi_mean,
    frac_veg: frac_veg,
    region: regKey,
    mois: mois,
    etp: etp,
    precip: precip,
    deficit: deficit
  };

  // Appel API modèle
  fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
  })
  .then(r => r.json())
  .then(res => afficherResultats(res, params, w, h, vari_arr, ndvi_arr))
  .catch(err => {
    console.error(err);
    document.getElementById('btnAnalyse').textContent = '🔬 Analyser avec le modèle IA';
    document.getElementById('btnAnalyse').disabled = false;
  });
}

function afficherResultats(res, params, w, h, vari_arr, ndvi_arr) {
  // Dessiner les cartes
  function drawCanvas(id, pxArr) {
    const cv = document.getElementById(id);
    cv.width=w; cv.height=h;
    const ctx=cv.getContext('2d');
    const img=ctx.createImageData(w,h);
    for(let i=0;i<w*h;i++){img.data[i*4]=pxArr[i*3];img.data[i*4+1]=pxArr[i*3+1];img.data[i*4+2]=pxArr[i*3+2];img.data[i*4+3]=255;}
    ctx.putImageData(img,0,0);
  }

  const n = w*h;
  // Original
  const cv0=document.getElementById('cvOrig'); cv0.width=w; cv0.height=h;
  cv0.getContext('2d').drawImage(currentImg,0,0,w,h);

  // VARI
  const vp=new Uint8Array(n*3);
  for(let i=0;i<n;i++){
    const v=vari_arr[i];
    const c=v<-0.1?[139,69,19]:v<0.05?[244,164,96]:v<0.15?[255,255,100]:v<0.30?[144,238,144]:[0,120,0];
    vp[i*3]=c[0];vp[i*3+1]=c[1];vp[i*3+2]=c[2];
  }
  drawCanvas('cvVARI',vp);
  document.getElementById('lblVARI').textContent=`VARI = ${params.vari.toFixed(3)}`;

  // NDVI
  const np=new Uint8Array(n*3);
  for(let i=0;i<n;i++){
    const v=ndvi_arr[i];
    const c=v<0.1?[139,0,0]:v<0.2?[255,69,0]:v<0.3?[255,165,0]:v<0.45?[255,255,0]:v<0.6?[124,252,0]:[0,100,0];
    np[i*3]=c[0];np[i*3+1]=c[1];np[i*3+2]=c[2];
  }
  drawCanvas('cvNDVI',np);
  document.getElementById('lblNDVI').textContent=`NDVI estimé = ${params.ndvi.toFixed(3)}`;

  // KPIs
  document.getElementById('kCulture').textContent = res.culture;
  document.getElementById('bCulture').textContent = `Confiance ${res.confiance}%`;

  const stress = res.stress;
  document.getElementById('kStress').textContent = stress.toFixed(3);
  const sc = stress>=0.75?'badge-red':stress>=0.5?'badge-amber':'badge-green';
  const sl = stress>=0.75?'Sévère':stress>=0.5?'Élevé':stress>=0.25?'Modéré':'Faible';
  document.getElementById('bStress').className=`kpi-badge ${sc}`;
  document.getElementById('bStress').textContent=sl;

  document.getElementById('kRdt').textContent = res.rendement.toFixed(1);

  const besoin = Math.round(Math.max(0, params.deficit/30)*10)/10;
  document.getElementById('kIrrig').textContent = besoin+' mm/j';
  const ic = besoin>5?'badge-red':besoin>2?'badge-amber':'badge-green';
  const il = besoin>5?'Urgente':besoin>2?'Recommandée':'Faible';
  document.getElementById('bIrrig').className=`kpi-badge ${ic}`;
  document.getElementById('bIrrig').textContent=il;

  // Features
  const feats = res.importance || {};
  document.getElementById('features').innerHTML = Object.entries(feats)
    .sort((a,b)=>b[1]-a[1]).slice(0,7)
    .map(([k,v])=>`
      <div class="bar-row">
        <span class="bar-name">${k}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${Math.round(v*100)}%;background:#2d7a2d">
            ${Math.round(v*100)}%
          </div>
        </div>
      </div>`).join('');

  // Recommandations
  const recos = [];
  if(besoin>2) recos.push({c:'#1a5fa5',t:`Irriguer ${besoin} mm/jour — déficit hydrique ${params.deficit} mm ce mois`});
  if(stress>0.6) recos.push({c:'#b35c00',t:`Stress élevé — végétation fragilisée (NDVI=${params.ndvi.toFixed(3)})`});
  if(params.frac_veg<0.3) recos.push({c:'#a32020',t:`Couverture végétale faible (${(params.frac_veg*100).toFixed(0)}%) — risque d'érosion`});
  recos.push({c:'#2d7a2d',t:`Culture recommandée : ${res.culture} (confiance ${res.confiance}%)`});
  recos.push({c:'#555',t:`Sol : ${REGIONS[document.getElementById('region').value].sol} · ETP mois : ${params.etp} mm · Précipitations : ${params.precip} mm`});

  document.getElementById('recos').innerHTML = recos
    .map(r=>`<div class="rec" style="border-color:${r.c};background:${r.c}18">${r.t}</div>`).join('');

  document.getElementById('results').style.display='block';
  document.getElementById('results').scrollIntoView({behavior:'smooth'});
  document.getElementById('btnAnalyse').textContent = '🔬 Analyser avec le modèle IA';
  document.getElementById('btnAnalyse').disabled = false;
}
</script>
</body>
</html>
"""

# ── Route principale ──────────────────────────────
@app.route("/")
def index():
    return HTML

# ── Route prédiction ─────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if bundle is None:
        return jsonify({"error": "Modèle non chargé. Lancez python train.py"}), 500

    data = request.json
    features = bundle["features"]

    row = np.array([[data.get(f, 0) for f in features]])

    # Prédictions du vrai modèle
    culture_enc  = bundle["clf_culture"].predict(row)[0]
    culture_nom  = bundle["le_culture"].inverse_transform([culture_enc])[0]
    proba        = bundle["clf_culture"].predict_proba(row)[0]
    confiance    = int(round(max(proba) * 100))

    stress_val   = float(bundle["reg_stress"].predict(row)[0])
    rendement    = float(bundle["reg_rendement"].predict(row)[0])

    # Feature importance dynamique
    importance = dict(zip(
        features,
        bundle["reg_stress"].feature_importances_
    ))
    top_importance = {k: round(float(v), 4) for k,v in
                      sorted(importance.items(), key=lambda x: -x[1])[:7]}

    return jsonify({
        "culture":    culture_nom,
        "confiance":  confiance,
        "stress":     round(stress_val, 3),
        "rendement":  round(max(0, rendement), 2),
        "importance": top_importance,
        "modele":     "Random Forest — données réelles",
        "metriques":  bundle.get("metrics", {}),
    })

# ── Lancer le serveur ────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AgriSat Maroc — Application démarrée")
    print("="*50)
    print("\n➡️  Ouvrez votre navigateur sur :")
    print("    http://localhost:5000")
    print("\n  Ctrl+C pour arrêter\n")
    app.run(debug=False, port=5000)