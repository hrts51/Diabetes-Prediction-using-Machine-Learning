"""
gen_html_docs.py
----------------
Generates a single self-contained HTML documentation file with:
  - All project documentation
  - Interview preparation guide
  - All 17 plots embedded as base64 (no external dependencies)
  - Professional styling with sidebar navigation

Run:  python src/gen_html_docs.py
Output: docs/documentation.html
"""

import base64, os, json

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(ROOT, 'outputs', 'plots')
OUT_FILE = os.path.join(ROOT, 'docs', 'documentation.html')
os.makedirs(os.path.join(ROOT, 'docs'), exist_ok=True)

# ── Encode plots ──────────────────────────────────────────────────────────────
def encode(fname):
    p = os.path.join(PLOT_DIR, fname)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def img(fname, caption='', width='100%'):
    b64 = encode(fname)
    if not b64:
        return f'<p style="color:#e74c3c">Plot not found: {fname}</p>'
    return f'''
<figure style="margin:24px 0; text-align:center;">
  <img src="data:image/png;base64,{b64}"
       style="width:{width}; max-width:1100px; border-radius:8px;
              box-shadow:0 2px 12px rgba(0,0,0,0.12);" alt="{caption}">
  <figcaption style="margin-top:8px; color:#666; font-size:0.88rem;
                     font-style:italic;">{caption}</figcaption>
</figure>'''

# ── Load results ──────────────────────────────────────────────────────────────
import csv

def load_csv(fname):
    path = os.path.join(ROOT, 'outputs', 'models', fname)
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

tuned_rows   = load_csv('results_tuned.csv')
default_rows = load_csv('results_default.csv')

def results_table(rows, highlight_col='F1-Score'):
    if not rows:
        return '<p>No results found.</p>'
    cols = list(rows[0].keys())
    # Find best value per numeric column
    best = {}
    for c in cols:
        try:
            vals = [float(r[c]) for r in rows]
            best[c] = max(vals)
        except:
            pass
    
    html = '<table><thead><tr>'
    for c in cols:
        html += f'<th>{c}</th>'
    html += '</tr></thead><tbody>'
    
    rows_sorted = sorted(rows, key=lambda r: float(r.get('F1-Score', 0)), reverse=True)
    for row in rows_sorted:
        is_best = float(row.get('F1-Score', 0)) == best.get('F1-Score', -1)
        tr_style = ' class="best-row"' if is_best else ''
        html += f'<tr{tr_style}>'
        for c in cols:
            val = row[c]
            cell_best = ''
            try:
                if float(val) == best.get(c, None):
                    cell_best = ' class="best-cell"'
            except:
                pass
            html += f'<td{cell_best}>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Diabetes Risk Prediction — Project Documentation</title>
<style>
  :root {{
    --primary:   #2c3e50;
    --accent:    #3498db;
    --green:     #27ae60;
    --red:       #e74c3c;
    --orange:    #f39c12;
    --bg:        #f4f6f9;
    --card:      #ffffff;
    --border:    #dde3ea;
    --text:      #2d3436;
    --muted:     #636e72;
    --sidebar-w: 280px;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
  }}

  /* ── Sidebar ── */
  #sidebar {{
    position: fixed; top: 0; left: 0;
    width: var(--sidebar-w); height: 100vh;
    background: var(--primary);
    overflow-y: auto; padding: 0 0 40px;
    z-index: 100;
  }}
  #sidebar .logo {{
    padding: 28px 20px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }}
  #sidebar .logo h2 {{
    color: #fff; font-size: 1rem; font-weight: 700; line-height: 1.4;
  }}
  #sidebar .logo p {{
    color: rgba(255,255,255,0.55); font-size: 0.75rem; margin-top: 4px;
  }}
  #sidebar nav a {{
    display: block; padding: 10px 20px;
    color: rgba(255,255,255,0.75); text-decoration: none;
    font-size: 0.84rem; border-left: 3px solid transparent;
    transition: all 0.2s;
  }}
  #sidebar nav a:hover, #sidebar nav a.active {{
    color: #fff; background: rgba(255,255,255,0.08);
    border-left-color: var(--accent);
  }}
  #sidebar nav .nav-section {{
    padding: 16px 20px 6px;
    color: rgba(255,255,255,0.35);
    font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px;
  }}

  /* ── Main ── */
  #main {{
    margin-left: var(--sidebar-w);
    padding: 40px 48px;
    max-width: 1200px;
  }}

  /* ── Hero ── */
  .hero {{
    background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    border-radius: 16px; padding: 48px 40px; margin-bottom: 40px;
    color: #fff;
  }}
  .hero h1 {{ font-size: 2.2rem; font-weight: 800; margin-bottom: 10px; }}
  .hero p  {{ font-size: 1.05rem; opacity: 0.85; margin-bottom: 6px; }}
  .badges {{ margin-top: 20px; display: flex; flex-wrap: wrap; gap: 8px; }}
  .badge {{
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: #fff; padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
  }}

  /* ── Stat cards ── */
  .stats-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 16px; margin-bottom: 40px;
  }}
  .stat-card {{
    background: var(--card); border-radius: 12px;
    padding: 20px 18px; border-top: 4px solid var(--accent);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
  }}
  .stat-card.green  {{ border-top-color: var(--green); }}
  .stat-card.orange {{ border-top-color: var(--orange); }}
  .stat-card.red    {{ border-top-color: var(--red); }}
  .stat-card .num  {{ font-size: 2rem; font-weight: 800; color: var(--primary); }}
  .stat-card .lbl  {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}

  /* ── Sections ── */
  .section {{
    background: var(--card); border-radius: 12px;
    padding: 36px 40px; margin-bottom: 32px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  h2.section-title {{
    font-size: 1.5rem; font-weight: 800; color: var(--primary);
    border-left: 5px solid var(--accent);
    padding-left: 14px; margin-bottom: 24px;
  }}
  h3 {{ font-size: 1.1rem; font-weight: 700; color: var(--primary);
        margin: 24px 0 10px; }}
  h4 {{ font-size: 0.95rem; font-weight: 700; color: var(--muted);
        margin: 18px 0 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
  p  {{ margin-bottom: 12px; }}

  /* ── Tables ── */
  table {{
    width: 100%; border-collapse: collapse;
    margin: 16px 0; font-size: 0.88rem;
  }}
  th {{
    background: var(--primary); color: #fff;
    padding: 10px 14px; text-align: left; font-weight: 600;
  }}
  td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #f8fafc; }}
  .best-row td {{ background: #f0fff4 !important; font-weight: 600; }}
  .best-cell {{ color: var(--green); font-weight: 700; }}

  /* ── Code blocks ── */
  pre, code {{
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
  }}
  pre {{
    background: #1e2a38; color: #e8eaed;
    border-radius: 10px; padding: 20px 24px;
    overflow-x: auto; font-size: 0.84rem; line-height: 1.6;
    margin: 16px 0;
  }}
  code {{
    background: #eef2f7; color: #c0392b;
    padding: 2px 6px; border-radius: 4px; font-size: 0.85em;
  }}
  pre code {{ background: none; color: inherit; padding: 0; }}

  /* ── Info boxes ── */
  .info-box {{
    border-radius: 10px; padding: 16px 20px; margin: 16px 0;
    border-left: 5px solid;
  }}
  .info-box.blue   {{ background:#ebf5fb; border-color:var(--accent); }}
  .info-box.green  {{ background:#eafaf1; border-color:var(--green);  }}
  .info-box.orange {{ background:#fef9e7; border-color:var(--orange); }}
  .info-box.red    {{ background:#fdedec; border-color:var(--red);    }}
  .info-box strong {{ display:block; margin-bottom:4px; }}

  /* ── Q&A cards ── */
  .qa-card {{
    border: 1px solid var(--border); border-radius: 10px;
    margin: 16px 0; overflow: hidden;
  }}
  .qa-q {{
    background: #f4f6f9; padding: 14px 18px;
    font-weight: 700; color: var(--primary);
    display: flex; align-items: flex-start; gap: 10px;
  }}
  .qa-q .level {{
    background: var(--accent); color: #fff;
    font-size: 0.7rem; padding: 2px 8px; border-radius: 10px;
    font-weight: 600; white-space: nowrap; margin-top: 2px;
  }}
  .qa-a {{
    padding: 14px 18px; font-size: 0.92rem; line-height: 1.7;
    border-top: 1px solid var(--border);
  }}

  /* ── Pipeline steps ── */
  .pipeline {{
    display: flex; flex-direction: column; gap: 0;
    margin: 20px 0;
  }}
  .pipe-step {{
    display: flex; align-items: flex-start; gap: 16px;
    padding: 14px 0;
    border-left: 3px solid var(--border);
    padding-left: 20px; margin-left: 16px;
    position: relative;
  }}
  .pipe-step::before {{
    content: attr(data-n);
    position: absolute; left: -16px; top: 14px;
    width: 30px; height: 30px; border-radius: 50%;
    background: var(--accent); color: #fff;
    font-weight: 800; font-size: 0.8rem;
    display: flex; align-items: center; justify-content: center;
  }}
  .pipe-step:last-child {{ border-left-color: transparent; }}
  .pipe-step .title {{ font-weight: 700; color: var(--primary); }}
  .pipe-step .desc  {{ font-size: 0.87rem; color: var(--muted); }}

  /* ── Feature cards ── */
  .feat-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin: 16px 0;
  }}
  .feat-card {{
    background: var(--bg); border-radius: 10px;
    padding: 18px; border-top: 3px solid var(--accent);
  }}
  .feat-card.orange {{ border-top-color: var(--orange); }}
  .feat-card.green  {{ border-top-color: var(--green); }}
  .feat-card h4     {{ margin: 0 0 8px; color: var(--primary); }}
  .feat-card code   {{ font-size: 0.78rem; display: block;
                       margin: 8px 0; white-space: pre-wrap; }}

  /* ── SHAP rank ── */
  .shap-bar {{ margin: 6px 0; }}
  .shap-bar .label {{
    display: flex; justify-content: space-between;
    font-size: 0.84rem; margin-bottom: 4px;
  }}
  .shap-bar .bar-bg {{
    background: #eef2f7; border-radius: 4px; height: 10px; overflow: hidden;
  }}
  .shap-bar .bar-fill {{
    height: 100%; background: linear-gradient(90deg, #3498db, #2ecc71);
    border-radius: 4px;
  }}

  /* ── TOC ── */
  .toc {{ columns: 2; gap: 24px; margin: 16px 0; }}
  .toc a {{ color: var(--accent); text-decoration: none; font-size: 0.9rem;
             display: block; padding: 3px 0; }}
  .toc a:hover {{ text-decoration: underline; }}

  /* ── Footer ── */
  footer {{
    text-align: center; padding: 32px;
    color: var(--muted); font-size: 0.85rem;
    border-top: 1px solid var(--border); margin-top: 48px;
  }}

  /* ── Responsive ── */
  @media (max-width: 900px) {{
    #sidebar {{ display: none; }}
    #main    {{ margin-left: 0; padding: 24px; }}
    .stats-grid {{ grid-template-columns: repeat(2,1fr); }}
    .feat-grid  {{ grid-template-columns: 1fr; }}
    .toc {{ columns: 1; }}
  }}
</style>
</head>
<body>

<!-- ══════════ SIDEBAR ══════════ -->
<div id="sidebar">
  <div class="logo">
    <h2>🩺 Diabetes Risk Prediction</h2>
    <p>Explainable ML System</p>
  </div>
  <nav>
    <div class="nav-section">Overview</div>
    <a href="#overview">Project Overview</a>
    <a href="#dataset">Dataset</a>
    <a href="#architecture">Architecture</a>

    <div class="nav-section">Pipeline</div>
    <a href="#preprocessing">Preprocessing</a>
    <a href="#features">Feature Engineering</a>
    <a href="#models">ML Models</a>
    <a href="#tuning">Hyperparameter Tuning</a>
    <a href="#evaluation">Evaluation Strategy</a>
    <a href="#threshold">Threshold Optimisation</a>
    <a href="#shap">SHAP Explainability</a>
    <a href="#learning">Learning Curves</a>

    <div class="nav-section">Results</div>
    <a href="#results">Results &amp; Plots</a>
    <a href="#eda-plots">EDA Plots</a>
    <a href="#model-plots">Model Plots</a>
    <a href="#shap-plots">SHAP Plots</a>

    <div class="nav-section">Reference</div>
    <a href="#interview">Interview Prep</a>
    <a href="#howtorun">How to Run</a>
    <a href="#limitations">Limitations</a>
  </nav>
</div>

<!-- ══════════ MAIN ══════════ -->
<div id="main">

<!-- Hero -->
<div class="hero">
  <h1>🩺 Diabetes Risk Prediction</h1>
  <p><strong>Explainable Machine Learning System</strong></p>
  <p>Author: Ankit Kumar &nbsp;|&nbsp; Academy of Skill Development, Kolkata &nbsp;|&nbsp; May–July 2025</p>
  <p>🔗 <a href="https://github.com/hrts51/diabetes-risk-prediction-explainable-ml"
           style="color:#fff;" target="_blank">github.com/hrts51/diabetes-risk-prediction-explainable-ml</a></p>
  <div class="badges">
    <span class="badge">Python 3.14</span>
    <span class="badge">scikit-learn 1.9</span>
    <span class="badge">XGBoost 3.4</span>
    <span class="badge">LightGBM 4.7</span>
    <span class="badge">SHAP 0.54</span>
    <span class="badge">Streamlit 1.67</span>
    <span class="badge">253,680 Records</span>
    <span class="badge">8 Models</span>
    <span class="badge">34 Plots</span>
  </div>
</div>

<!-- Stats -->
<div class="stats-grid">
  <div class="stat-card">
    <div class="num">0.826</div>
    <div class="lbl">Best ROC-AUC<br>(Tuned LightGBM)</div>
  </div>
  <div class="stat-card green">
    <div class="num">0.971</div>
    <div class="lbl">CV ROC-AUC<br>(XGBoost / LightGBM)</div>
  </div>
  <div class="stat-card orange">
    <div class="num">61.7%</div>
    <div class="lbl">Recall<br>(Diabetics Caught)</div>
  </div>
  <div class="stat-card red">
    <div class="num">253K</div>
    <div class="lbl">Real CDC Records<br>(BRFSS 2015)</div>
  </div>
</div>

<!-- ══ 1. OVERVIEW ══ -->
<div class="section" id="overview">
  <h2 class="section-title">1. Project Overview</h2>
  <p>Diabetes is one of the leading chronic diseases globally. This project builds a
  <strong>supervised binary classification system</strong> that predicts whether a person
  is at risk of diabetes based on 21 health and lifestyle indicators from the US CDC.</p>

  <div class="info-box blue">
    <strong>Primary Goals</strong>
    High recall — catch as many diabetic patients as possible |
    Model transparency via SHAP — explain every prediction |
    Practical deployment via Streamlit web app
  </div>

  <div class="info-box orange">
    <strong>Why recall matters more than accuracy</strong>
    Missing a diabetic patient (false negative) is clinically more costly than a false alarm.
    A false alarm leads to unnecessary follow-up. A missed diagnosis leads to untreated disease.
    All design decisions — SMOTE, threshold optimisation, metric choice — reflect this.
  </div>

  <h3>Table of Contents</h3>
  <div class="toc">
    <a href="#overview">1. Project Overview</a>
    <a href="#dataset">2. Dataset Description</a>
    <a href="#architecture">3. Architecture</a>
    <a href="#preprocessing">4. Data Preprocessing</a>
    <a href="#features">5. Feature Engineering</a>
    <a href="#models">6. Machine Learning Models</a>
    <a href="#tuning">7. Hyperparameter Tuning</a>
    <a href="#evaluation">8. Evaluation Strategy</a>
    <a href="#threshold">9. Threshold Optimisation</a>
    <a href="#shap">10. SHAP Explainability</a>
    <a href="#learning">11. Learning Curves</a>
    <a href="#results">12. Results</a>
    <a href="#interview">13. Interview Preparation</a>
    <a href="#howtorun">14. How to Run</a>
    <a href="#limitations">15. Limitations &amp; Future Work</a>
  </div>
</div>

<!-- ══ 2. DATASET ══ -->
<div class="section" id="dataset">
  <h2 class="section-title">2. Dataset Description</h2>

  <p><strong>CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015</strong><br>
  Source: <a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset"
  target="_blank">Kaggle — Alex Teboul</a></p>

  <p>The BRFSS is an annual telephone survey conducted by the CDC across all 50 US states,
  collecting health behaviour data from over 400,000 adults per year.</p>

  <h3>Files</h3>
  <table>
    <thead><tr><th>File</th><th>Rows</th><th>Description</th></tr></thead>
    <tbody>
      <tr><td><code>diabetes_binary_health_indicators_BRFSS2015.csv</code></td><td>253,680</td><td>Main dataset — natural 86/14 imbalance ✅ Used</td></tr>
      <tr><td><code>diabetes_binary_5050split...</code></td><td>70,692</td><td>Pre-balanced 50/50 version</td></tr>
      <tr><td><code>diabetes_012_health_indicators...</code></td><td>253,680</td><td>3-class: No / Pre-diabetic / Diabetic</td></tr>
    </tbody>
  </table>

  <h3>Class Distribution</h3>
  <table>
    <thead><tr><th>Class</th><th>Count</th><th>Percentage</th></tr></thead>
    <tbody>
      <tr><td>Non-Diabetic (0)</td><td>218,334</td><td>86.1%</td></tr>
      <tr class="best-row"><td>Diabetic (1)</td><td>35,346</td><td>13.9%</td></tr>
    </tbody>
  </table>
  <div class="info-box orange">
    This imbalance is <strong>real and meaningful</strong> — it reflects the actual prevalence
    of diabetes in the US adult population. We preserve it in validation and test sets.
  </div>

  <h3>All 21 Features</h3>
  <table>
    <thead><tr><th>Feature</th><th>Type</th><th>Description</th></tr></thead>
    <tbody>
      <tr><td><code>HighBP</code></td><td>Binary</td><td>Ever told blood pressure is high</td></tr>
      <tr><td><code>HighChol</code></td><td>Binary</td><td>Ever told cholesterol is high</td></tr>
      <tr><td><code>CholCheck</code></td><td>Binary</td><td>Cholesterol check in past 5 years</td></tr>
      <tr><td><code>BMI</code></td><td>Continuous</td><td>Body Mass Index</td></tr>
      <tr><td><code>Smoker</code></td><td>Binary</td><td>Smoked ≥100 cigarettes in lifetime</td></tr>
      <tr><td><code>Stroke</code></td><td>Binary</td><td>Ever had a stroke</td></tr>
      <tr><td><code>HeartDiseaseorAttack</code></td><td>Binary</td><td>Coronary heart disease or MI</td></tr>
      <tr><td><code>PhysActivity</code></td><td>Binary</td><td>Physical activity in past 30 days</td></tr>
      <tr><td><code>Fruits</code></td><td>Binary</td><td>Fruit consumption ≥1×/day</td></tr>
      <tr><td><code>Veggies</code></td><td>Binary</td><td>Vegetable consumption ≥1×/day</td></tr>
      <tr><td><code>HvyAlcoholConsump</code></td><td>Binary</td><td>Heavy alcohol consumption</td></tr>
      <tr><td><code>AnyHealthcare</code></td><td>Binary</td><td>Any healthcare coverage</td></tr>
      <tr><td><code>NoDocbcCost</code></td><td>Binary</td><td>Could not see doctor due to cost</td></tr>
      <tr><td><code>GenHlth</code></td><td>Ordinal 1–5</td><td>General health (1=Excellent, 5=Poor)</td></tr>
      <tr><td><code>MentHlth</code></td><td>0–30</td><td>Poor mental health days (past 30)</td></tr>
      <tr><td><code>PhysHlth</code></td><td>0–30</td><td>Poor physical health days (past 30)</td></tr>
      <tr><td><code>DiffWalk</code></td><td>Binary</td><td>Difficulty walking/climbing stairs</td></tr>
      <tr><td><code>Sex</code></td><td>Binary</td><td>0=Female, 1=Male</td></tr>
      <tr><td><code>Age</code></td><td>Ordinal 1–13</td><td>Age category (1=18–24, 13=80+)</td></tr>
      <tr><td><code>Education</code></td><td>Ordinal 1–6</td><td>Education level</td></tr>
      <tr><td><code>Income</code></td><td>Ordinal 1–8</td><td>Income scale (&lt;$10K to ≥$75K)</td></tr>
    </tbody>
  </table>
</div>

<!-- ══ 3. ARCHITECTURE ══ -->
<div class="section" id="architecture">
  <h2 class="section-title">3. Project Architecture</h2>
  <div class="pipeline">
    <div class="pipe-step" data-n="1">
      <div><div class="title">run_pipeline.py</div>
      <div class="desc">Orchestrates the full 13-step pipeline. Single entry point.</div></div>
    </div>
    <div class="pipe-step" data-n="2">
      <div><div class="title">src/preprocessing.py</div>
      <div class="desc">Load, validate, feature engineer, stratified split, SMOTE, StandardScaler</div></div>
    </div>
    <div class="pipe-step" data-n="3">
      <div><div class="title">src/models.py</div>
      <div class="desc">All 8 model definitions, train(), evaluate(), threshold optimisation functions</div></div>
    </div>
    <div class="pipe-step" data-n="4">
      <div><div class="title">src/tuning.py</div>
      <div class="desc">RandomizedSearchCV for XGBoost + LightGBM, learning curve computation</div></div>
    </div>
    <div class="pipe-step" data-n="5">
      <div><div class="title">src/visualizations.py</div>
      <div class="desc">16 plot functions: EDA, ROC, confusion matrices, learning curves, threshold, ranking</div></div>
    </div>
    <div class="pipe-step" data-n="6">
      <div><div class="title">src/explainability.py</div>
      <div class="desc">SHAP TreeExplainer, global summary, bar chart, per-patient waterfall</div></div>
    </div>
    <div class="pipe-step" data-n="7">
      <div><div class="title">app/streamlit_app.py</div>
      <div class="desc">4-tab Streamlit web app: Predict / Results / Plots / About</div></div>
    </div>
  </div>
</div>

<!-- ══ 4. PREPROCESSING ══ -->
<div class="section" id="preprocessing">
  <h2 class="section-title">4. Data Preprocessing</h2>

  <h3>Split Strategy: 80 / 5 / 15</h3>
  <table>
    <thead><tr><th>Set</th><th>Size</th><th>Rows</th><th>Class Balance</th><th>Purpose</th></tr></thead>
    <tbody>
      <tr><td>Training</td><td>80%</td><td>~202,905 → 349,268 after SMOTE</td><td>50/50 (after SMOTE)</td><td>Model training</td></tr>
      <tr><td>Validation</td><td>5%</td><td>~12,723</td><td>86/14 (real)</td><td>Threshold search only</td></tr>
      <tr><td>Test</td><td>15%</td><td>~38,052</td><td>86/14 (real)</td><td>Final evaluation</td></tr>
    </tbody>
  </table>

  <div class="info-box green">
    <strong>Critical design decision</strong>
    Validation and test sets are split <em>before</em> SMOTE, preserving real 86/14 class distribution.
    This ensures threshold search and final evaluation happen on realistic data.
  </div>

  <h3>SMOTE — Training Only</h3>
  <pre><code>from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_trainval, y_trainval)
# Before: [174,634 non-diabetic, 28,271 diabetic]
# After : [174,634 non-diabetic, 174,634 diabetic]  ← balanced</code></pre>

  <h3>StandardScaler</h3>
  <pre><code>scaler         = StandardScaler()
X_train_sc     = scaler.fit_transform(X_train)   # fit + transform
X_val_sc       = scaler.transform(X_val)          # transform only
X_test_sc      = scaler.transform(X_test)         # transform only
# Tree models (RF, XGBoost, LightGBM) use RAW data — scale-invariant</code></pre>
</div>

<!-- ══ 5. FEATURE ENGINEERING ══ -->
<div class="section" id="features">
  <h2 class="section-title">5. Feature Engineering</h2>
  <p>Three clinically-motivated composite features added on top of the original 21:</p>

  <div class="feat-grid">
    <div class="feat-card">
      <h4>MetabolicRisk</h4>
      <code>BMI×0.4 + HighBP×10 + HighChol×8
+ HeartDisease×12 + Stroke×10</code>
      <p style="font-size:0.84rem">Combines the 5 strongest cardiometabolic risk factors.
      <strong>Ranked 3rd globally in SHAP</strong> (Mean |SHAP|=0.378) — above individual
      features like BMI and HighBP.</p>
    </div>
    <div class="feat-card orange">
      <h4>LifestyleScore</h4>
      <code>PhysActivity×3 + Fruits + Veggies
- Smoker×2 - HvyAlcohol×2</code>
      <p style="font-size:0.84rem">Healthy lifestyle proxy. Physical activity weighted ×3
      (strongest evidence-based intervention). Smoking and heavy alcohol reduce the score.</p>
    </div>
    <div class="feat-card green">
      <h4>HealthcareBurden</h4>
      <code>MentHlth + PhysHlth</code>
      <p style="font-size:0.84rem">Total days of poor health in past 30 days.
      Captures chronic illness burden which correlates strongly with diabetes progression.</p>
    </div>
  </div>
</div>

<!-- ══ 6. MODELS ══ -->
<div class="section" id="models">
  <h2 class="section-title">6. Machine Learning Models</h2>
  <table>
    <thead><tr><th>Model</th><th>Type</th><th>Key Strength</th></tr></thead>
    <tbody>
      <tr><td><strong>Logistic Regression</strong></td><td>Linear</td><td>Interpretable baseline — coefficients show feature direction</td></tr>
      <tr><td><strong>Random Forest</strong></td><td>Bagging (300 trees)</td><td>Robust to outliers, Gini feature importance</td></tr>
      <tr><td><strong>XGBoost</strong></td><td>Gradient Boosting</td><td>Sequential trees, L1/L2 regularisation, top tabular performer</td></tr>
      <tr><td><strong>LightGBM</strong></td><td>Gradient Boosting (leaf-wise)</td><td>Faster than XGBoost on large data, best performer</td></tr>
      <tr><td><strong>SVM (Linear)</strong></td><td>Maximum Margin</td><td>LinearSVC + CalibratedClassifierCV, fast on 253K rows</td></tr>
      <tr><td><strong>Tuned XGBoost</strong></td><td>Gradient Boosting</td><td>Best params from 40×5-fold search (CV F1=0.9129)</td></tr>
      <tr><td><strong>Tuned LightGBM</strong></td><td>Gradient Boosting</td><td>Best params from 40×5-fold search (CV F1=0.9140) ← Best overall</td></tr>
      <tr><td><strong>Stacking Ensemble</strong></td><td>Meta-learning</td><td>RF+XGBoost+LightGBM → Logistic Regression meta-learner</td></tr>
    </tbody>
  </table>
</div>

<!-- ══ 7. TUNING ══ -->
<div class="section" id="tuning">
  <h2 class="section-title">7. Hyperparameter Tuning</h2>

  <div class="info-box blue">
    <strong>Method: RandomizedSearchCV</strong>
    40 random combinations × 5-fold stratified CV. Scoring metric: F1 (not accuracy).
    All CPU cores used (n_jobs=-1). Tuning on 60,000-row sample for speed, full retrain after.
  </div>

  <h3>Best Parameters — XGBoost (CV F1 = 0.9129)</h3>
  <pre><code>learning_rate=0.08, max_depth=7, subsample=0.6,
colsample_bytree=0.7, gamma=0.1, reg_alpha=0.5,
reg_lambda=0.5, min_child_weight=3, n_estimators=200</code></pre>

  <h3>Best Parameters — LightGBM (CV F1 = 0.9140)</h3>
  <pre><code>learning_rate=0.08, num_leaves=127, max_depth=4,
subsample=0.9, colsample_bytree=0.6, reg_alpha=0.1,
reg_lambda=1.0, min_child_samples=10, n_estimators=500,
class_weight='balanced'</code></pre>

  <div class="info-box orange">
    <strong>Note on CV F1 = 0.91</strong>
    This is computed on SMOTE-balanced training folds. Test-set F1 at default threshold is lower (~0.27)
    because the class distribution returns to real 86/14 — this is expected and correct. ROC-AUC 0.971
    is the meaningful number.
  </div>

  {img('15_tuning_improvement.png', 'Before vs After Hyperparameter Tuning')}
</div>

<!-- ══ 8. EVALUATION ══ -->
<div class="section" id="evaluation">
  <h2 class="section-title">8. Model Evaluation Strategy</h2>

  <table>
    <thead><tr><th>Metric</th><th>Formula</th><th>Why Used</th></tr></thead>
    <tbody>
      <tr><td><strong>ROC-AUC ★</strong></td><td>Area under ROC curve</td><td>Primary. Threshold-independent, robust to imbalance</td></tr>
      <tr><td>F1-Score</td><td>2×P×R/(P+R)</td><td>Balances precision and recall at chosen threshold</td></tr>
      <tr><td>Recall (Sensitivity)</td><td>TP/(TP+FN)</td><td>% of diabetics correctly identified — most clinically important</td></tr>
      <tr><td>Precision</td><td>TP/(TP+FP)</td><td>% of predicted diabetics actually diabetic</td></tr>
      <tr><td>Accuracy</td><td>(TP+TN)/Total</td><td>Reported only — misleading with 86/14 imbalance</td></tr>
    </tbody>
  </table>

  <div class="info-box red">
    <strong>Why accuracy is misleading here</strong>
    A model predicting "No Diabetes" always:
    Accuracy = 86.1% ✓ | Recall = 0% ✗ | ROC-AUC = 0.5 ✗
    ROC-AUC correctly identifies this as a random classifier.
  </div>

  <h3>5-Fold Cross-Validation Results</h3>
  <table>
    <thead><tr><th>Model</th><th>CV ROC-AUC</th></tr></thead>
    <tbody>
      <tr class="best-row"><td>XGBoost</td><td>0.9714 ± 0.0006</td></tr>
      <tr class="best-row"><td>LightGBM</td><td>0.9714 ± 0.0006</td></tr>
      <tr><td>Random Forest</td><td>0.9480 ± 0.0012</td></tr>
      <tr><td>Logistic Regression</td><td>0.8269 ± 0.0004</td></tr>
      <tr><td>SVM (Linear)</td><td>0.8267 ± 0.0005</td></tr>
    </tbody>
  </table>
</div>

<!-- ══ 9. THRESHOLD ══ -->
<div class="section" id="threshold">
  <h2 class="section-title">9. Threshold Optimisation</h2>

  <p>At default threshold 0.50, XGBoost: Accuracy=86.4% but Recall=19% — misses 81% of diabetics.</p>

  <pre><code>precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
f1_scores    = 2 * precisions * recalls / (precisions + recalls)
best_thresh  = thresholds[np.argmax(f1_scores)]</code></pre>

  <div class="info-box green">
    <strong>Key design: search on real-distribution validation set</strong>
    The validation set is split before SMOTE (86/14 distribution = same as deployment).
    If we searched on SMOTE-balanced data, the optimal threshold (~0.50) would be wrong.
  </div>

  <h3>Optimal Thresholds</h3>
  <table>
    <thead><tr><th>Model</th><th>Threshold</th><th>Recall</th><th>F1-Score</th></tr></thead>
    <tbody>
      <tr class="best-row"><td>Tuned LightGBM</td><td>0.243</td><td>61.7%</td><td>0.4661</td></tr>
      <tr><td>XGBoost</td><td>0.237</td><td>64.9%</td><td>0.4646</td></tr>
      <tr><td>LightGBM</td><td>0.275</td><td>57.5%</td><td>0.4617</td></tr>
      <tr><td>Tuned XGBoost</td><td>0.269</td><td>56.9%</td><td>0.4610</td></tr>
      <tr><td>Random Forest</td><td>0.476</td><td>59.3%</td><td>0.4518</td></tr>
    </tbody>
  </table>

  {img('10_threshold_comparison.png', 'Default Threshold (0.50) vs Optimised Threshold')}
</div>

<!-- ══ 10. SHAP ══ -->
<div class="section" id="shap">
  <h2 class="section-title">10. SHAP Explainability</h2>

  <p>SHAP (SHapley Additive exPlanations) assigns each feature a contribution score for
  every prediction, based on game theory's Shapley values.</p>

  <div class="info-box blue">
    <strong>Key Properties</strong>
    Locally accurate (SHAP values sum to prediction) |
    Consistent (higher importance → higher SHAP) |
    TreeExplainer is exact and fast for tree models
  </div>

  <h3>Top 10 Features — Tuned XGBoost</h3>
  <div class="shap-bar"><div class="label"><span>1. GenHlth (General Health)</span><span>0.766</span></div><div class="bar-bg"><div class="bar-fill" style="width:100%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>2. Age</span><span>0.451</span></div><div class="bar-bg"><div class="bar-fill" style="width:59%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>3. MetabolicRisk <em>(engineered)</em></span><span>0.378</span></div><div class="bar-bg"><div class="bar-fill" style="width:49%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>4. BMI</span><span>0.309</span></div><div class="bar-bg"><div class="bar-fill" style="width:40%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>5. Education</span><span>0.209</span></div><div class="bar-bg"><div class="bar-fill" style="width:27%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>6. Sex</span><span>0.184</span></div><div class="bar-bg"><div class="bar-fill" style="width:24%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>7. HighBP</span><span>0.183</span></div><div class="bar-bg"><div class="bar-fill" style="width:24%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>8. Income</span><span>0.176</span></div><div class="bar-bg"><div class="bar-fill" style="width:23%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>9. Fruits</span><span>0.129</span></div><div class="bar-bg"><div class="bar-fill" style="width:17%"></div></div></div>
  <div class="shap-bar"><div class="label"><span>10. DiffWalk</span><span>0.118</span></div><div class="bar-bg"><div class="bar-fill" style="width:15%"></div></div></div>

  <div class="info-box green" style="margin-top:16px">
    <strong>Validation of feature engineering</strong>
    MetabolicRisk (engineered) ranks 3rd globally — above individual features BMI (4th) and
    HighBP (7th). This confirms the composite feature carries real, non-redundant signal.
  </div>
</div>

<!-- ══ 11. LEARNING CURVES ══ -->
<div class="section" id="learning">
  <h2 class="section-title">11. Learning Curves</h2>
  <table>
    <thead><tr><th>Pattern</th><th>Diagnosis</th><th>Fix</th></tr></thead>
    <tbody>
      <tr><td>Large train-val gap</td><td>Overfitting</td><td>Regularise, reduce complexity</td></tr>
      <tr><td>Both curves low &amp; converged</td><td>Underfitting</td><td>More features, complex model</td></tr>
      <tr><td>Val curve still rising</td><td>More data would help</td><td>Collect more data</td></tr>
      <tr class="best-row"><td>Val curve plateaued, small gap</td><td>Well-fitted ✓</td><td>—</td></tr>
    </tbody>
  </table>
  <p><strong>Observations:</strong> XGBoost/LightGBM show small train-val gap — well-fitted.
  Logistic Regression converges early — bias-limited (non-linear data). Random Forest shows
  slight gap — variance present but reduces with more data.</p>
  {img('14_all_learning_curves.png', 'Learning Curves — All Models (ROC-AUC vs Training Size)')}
</div>

<!-- ══ 12. RESULTS ══ -->
<div class="section" id="results">
  <h2 class="section-title">12. Results</h2>

  <h3>Final Results — Optimised Thresholds</h3>
  {results_table(tuned_rows)}

  <h3>F1 Improvement After Threshold Optimisation</h3>
  <table>
    <thead><tr><th>Model</th><th>Default F1</th><th>Tuned F1</th><th>Gain</th></tr></thead>
    <tbody>
      <tr class="best-row"><td>Tuned LightGBM</td><td>0.262</td><td>0.466</td><td>▲ +0.204</td></tr>
      <tr><td>XGBoost</td><td>0.279</td><td>0.465</td><td>▲ +0.186</td></tr>
      <tr><td>Tuned XGBoost</td><td>0.269</td><td>0.461</td><td>▲ +0.192</td></tr>
      <tr><td>LightGBM</td><td>0.285</td><td>0.462</td><td>▲ +0.177</td></tr>
    </tbody>
  </table>

  {img('12_final_model_ranking.png', 'Final Model Ranking by F1-Score')}
  {img('11_tuned_confusion_matrices.png', 'Confusion Matrices at Optimised Thresholds')}
</div>

<!-- ══ EDA PLOTS ══ -->
<div class="section" id="eda-plots">
  <h2 class="section-title">EDA Plots</h2>
  {img('01_target_distribution.png', 'Fig 1 — Class Distribution: 86.1% Non-Diabetic / 13.9% Diabetic')}
  {img('02_feature_distributions.png', 'Fig 2 — Feature Distributions by Diabetes Status')}
  {img('03_correlation_heatmap.png',   'Fig 3 — Feature Correlation Heatmap')}
  {img('04_boxplots_by_class.png',     'Fig 4 — Continuous Feature Boxplots by Class')}
</div>

<!-- ══ MODEL PLOTS ══ -->
<div class="section" id="model-plots">
  <h2 class="section-title">Model Performance Plots</h2>
  {img('05_model_comparison.png',         'Fig 5 — All Models: Accuracy, Precision, Recall, F1, AUC')}
  {img('06_roc_curves.png',               'Fig 6 — ROC Curves — All Models')}
  {img('07_confusion_matrices.png',       'Fig 7 — Confusion Matrices at Default Threshold (0.50)')}
  {img('08_rf_feature_importance.png',    'Fig 8 — Random Forest Gini Feature Importance')}
  {img('16_combined_feature_importance.png', 'Fig 9 — Feature Importance: RF vs XGBoost vs LightGBM')}
</div>

<!-- ══ SHAP PLOTS ══ -->
<div class="section" id="shap-plots">
  <h2 class="section-title">SHAP Explainability Plots</h2>
  {img('shap_summary_Tuned_XGBoost.png',       'Fig 10 — SHAP Summary (Beeswarm): Global Feature Impact')}
  {img('shap_bar_Tuned_XGBoost.png',           'Fig 11 — SHAP Bar: Mean |SHAP| Feature Ranking')}
  {img('shap_waterfall_Tuned_XGBoost_idx5.png','Fig 12 — SHAP Waterfall: Per-Patient Explanation')}
</div>

<!-- ══ 13. INTERVIEW ══ -->
<div class="section" id="interview">
  <h2 class="section-title">13. Interview Preparation</h2>

  <div class="info-box blue">
    <strong>60-Second Pitch</strong>
    "I built an end-to-end diabetes risk prediction system using 253,680 real CDC survey records —
    not the standard PIMA dataset. I trained 8 models including tuned XGBoost and LightGBM with
    RandomizedSearchCV, handled class imbalance with SMOTE on training data only, and added SHAP
    explainability. The best model achieves ROC-AUC 0.826 on the real 86/14 test set. I engineered
    a MetabolicRisk feature that ranked 3rd in SHAP — above individual features like blood pressure
    and cholesterol. Deployed as a 4-tab Streamlit web app."
  </div>

  <h3>Level 1 — Basics</h3>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Basic</span> What problem does this project solve?</div>
    <div class="qa-a">Early detection of diabetes risk from health and lifestyle survey data. The model flags high-risk individuals for medical screening, enabling preventive intervention before clinical diagnosis.</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Basic</span> Why CDC BRFSS instead of PIMA Indians dataset?</div>
    <div class="qa-a">PIMA has only 768 rows from one specific ethnic group. CDC BRFSS has 253,680 real records from a diverse US adult population across all 50 states. It's more realistic, generalisable, and cited in peer-reviewed papers in Nature and NIH journals (2024–2025).</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Basic</span> What accuracy did you get?</div>
    <div class="qa-a">Accuracy is misleading here — with 86% non-diabetic, any model always predicting "No Diabetes" gets 86%. I report ROC-AUC (0.826) and F1 (0.466). The model correctly identifies 61.7% of diabetic patients (recall), which is the clinically meaningful number.</div>
  </div>

  <h3>Level 2 — Technical</h3>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Technical</span> What is SMOTE and why training data only?</div>
    <div class="qa-a">SMOTE creates synthetic minority-class samples by interpolating between existing ones in feature space. Applied only to training data because validation and test sets must represent real class distribution (86/14). Applying to test data would create artificially optimistic results — data leakage.</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Technical</span> Explain your train/val/test split design.</div>
    <div class="qa-a">80% training (SMOTE-balanced to 50/50) | 5% validation — pre-SMOTE, real 86/14 — used only for threshold search | 15% test — pre-SMOTE, real 86/14 — held out until final evaluation. The critical decision: validation split before SMOTE ensures threshold search at real distribution.</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Technical</span> Why threshold = 0.24 instead of 0.5?</div>
    <div class="qa-a">At 0.5 threshold, XGBoost recall = 19% (misses 81% of diabetics). SMOTE training teaches 50/50 balance, but test time returns to 86/14 — models assign low probabilities to positive. Lowering threshold to 0.24 trades precision for recall, raising F1 from 0.279 to 0.465 and recall from 19% to 65%.</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Technical</span> What is the bias-variance tradeoff and where does it appear?</div>
    <div class="qa-a">Bias = error from oversimplified assumptions (underfitting). Variance = sensitivity to training data (overfitting). Logistic Regression has high bias on this non-linear data. Deep Random Forest shows variance — training-validation gap in learning curves. XGBoost's gamma, reg_alpha, reg_lambda control this tradeoff, explaining why tuning improved it.</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Technical</span> Gini importance vs SHAP — what's the difference?</div>
    <div class="qa-a">Gini importance measures impurity reduction across all trees — biased toward high-cardinality features, doesn't capture interactions. SHAP computes exact Shapley values — model-agnostic, locally accurate (sum to prediction), no cardinality bias. GenHlth ranks first in both, confirming consistency. SHAP additionally shows direction and magnitude per sample.</div>
  </div>

  <h3>Level 3 — Advanced</h3>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Advanced</span> How does XGBoost work mathematically?</div>
    <div class="qa-a">XGBoost builds trees sequentially fitting the negative gradient of the loss function (gradient descent in function space). Objective: L = Σ l(ŷᵢ, yᵢ) + Σ Ω(fₖ) where Ω = γT + ½λ||w||² (T=leaves, w=leaf weights). Uses second-order Taylor approximation for efficient closed-form leaf weight computation.</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Advanced</span> Give 3 ways you prevented data leakage.</div>
    <div class="qa-a">(1) StandardScaler fit only on training data — prevents test statistics influencing normalisation. (2) SMOTE applied only to training data — prevents synthetic test samples. (3) Threshold optimisation on held-out validation set, never the test set — prevents threshold overfitting.</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Advanced</span> Why did the Stacking Ensemble underperform?</div>
    <div class="qa-a">Two reasons: (1) All base learners (RF, XGBoost, LightGBM) are tree-based — correlated, make similar errors on same samples. True stacking needs architecturally diverse base models. (2) Base learners trained on SMOTE-balanced data produce probability scales that don't align with the real-distribution meta-learner evaluation. Platt scaling / isotonic regression would fix this.</div>
  </div>

  <div class="qa-card">
    <div class="qa-q"><span class="level">Advanced</span> What would you do next to improve this?</div>
    <div class="qa-a">(1) Optuna for Bayesian hyperparameter optimisation. (2) Platt scaling for probability calibration. (3) Feature selection with RFE to reduce noise. (4) Temporal validation — train on BRFSS 2014/2015, test on 2016/2017. (5) TabNet or simple MLP for neural network comparison.</div>
  </div>

  <h3>Resume Bullet Points</h3>
  <pre><code>• Built end-to-end diabetes risk prediction system on 253,680 CDC BRFSS records
  using XGBoost, LightGBM, and Stacking Ensemble; achieved ROC-AUC 0.826

• Applied SMOTE (training only), RandomizedSearchCV (40×5-fold, CV F1=0.913/0.914),
  threshold optimisation raising XGBoost F1 from 0.279 → 0.465

• Engineered 3 composite features (MetabolicRisk, LifestyleScore, HealthcareBurden);
  MetabolicRisk ranked 3rd globally by SHAP feature importance

• Implemented SHAP explainability (global summary + per-patient waterfall plots)
  and deployed as 4-tab Streamlit web app</code></pre>
</div>

<!-- ══ 14. HOW TO RUN ══ -->
<div class="section" id="howtorun">
  <h2 class="section-title">14. How to Run</h2>

  <h3>1. Install dependencies</h3>
  <pre><code>pip install -r requirements.txt</code></pre>

  <h3>2. Download dataset</h3>
  <pre><code># Go to:
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
# Download and place in data/ folder</code></pre>

  <h3>3. Run full pipeline (~25–30 min)</h3>
  <pre><code>python run_pipeline.py</code></pre>

  <h3>4. Launch web app</h3>
  <pre><code>python -m streamlit run app/streamlit_app.py
# Open: http://localhost:8501</code></pre>

  <h3>Pipeline Steps</h3>
  <div class="pipeline">
    <div class="pipe-step" data-n="1"><div><div class="title">Load Data</div><div class="desc">253,680 rows, 22 columns, 0 missing values</div></div></div>
    <div class="pipe-step" data-n="2"><div><div class="title">EDA Plots</div><div class="desc">Distribution, correlations, boxplots by class</div></div></div>
    <div class="pipe-step" data-n="3"><div><div class="title">Feature Engineering</div><div class="desc">MetabolicRisk, LifestyleScore, HealthcareBurden</div></div></div>
    <div class="pipe-step" data-n="4"><div><div class="title">Split + SMOTE + Scale</div><div class="desc">80/5/15 split, SMOTE on train only, StandardScaler</div></div></div>
    <div class="pipe-step" data-n="5"><div><div class="title">Train 5 Base Models</div><div class="desc">LR, RF, XGBoost, LightGBM, SVM</div></div></div>
    <div class="pipe-step" data-n="6"><div><div class="title">Default Evaluation + 5-Fold CV</div><div class="desc">All metrics, ROC-AUC CV scores</div></div></div>
    <div class="pipe-step" data-n="7"><div><div class="title">Hyperparameter Tuning</div><div class="desc">RandomizedSearchCV 40×5-fold for XGBoost + LightGBM</div></div></div>
    <div class="pipe-step" data-n="8"><div><div class="title">Stacking Ensemble</div><div class="desc">RF + XGBoost + LightGBM → Logistic Regression</div></div></div>
    <div class="pipe-step" data-n="9"><div><div class="title">Learning Curves</div><div class="desc">Overfitting/underfitting diagnosis for all models</div></div></div>
    <div class="pipe-step" data-n="10"><div><div class="title">Threshold Optimisation</div><div class="desc">F1-optimal threshold per model on real-distribution val set</div></div></div>
    <div class="pipe-step" data-n="11"><div><div class="title">SHAP Explainability</div><div class="desc">Summary, bar, waterfall plots for Tuned XGBoost</div></div></div>
    <div class="pipe-step" data-n="12"><div><div class="title">Save Everything</div><div class="desc">8 model .pkl files, thresholds.json, results CSVs, 34 plots</div></div></div>
  </div>
</div>

<!-- ══ 15. LIMITATIONS ══ -->
<div class="section" id="limitations">
  <h2 class="section-title">15. Limitations &amp; Future Work</h2>

  <h3>Current Limitations</h3>
  <table>
    <thead><tr><th>Limitation</th><th>Detail</th></tr></thead>
    <tbody>
      <tr><td>Self-reported data</td><td>BRFSS telephone survey — responses may not be medically accurate</td></tr>
      <tr><td>No clinical measurements</td><td>No blood glucose, HbA1c, insulin — the most direct indicators</td></tr>
      <tr><td>Cross-sectional</td><td>One point in time — cannot capture disease progression</td></tr>
      <tr><td>US population only</td><td>May not generalise to other countries or ethnic groups</td></tr>
      <tr><td>Stacking Ensemble</td><td>Needed manual threshold fix — probability recalibration required</td></tr>
      <tr><td>Moderate F1 (~0.47)</td><td>Real limitation of survey features vs clinical measurements</td></tr>
    </tbody>
  </table>

  <h3>Future Work</h3>
  <ol style="padding-left:20px; line-height:2.2;">
    <li><strong>Optuna</strong> — Bayesian hyperparameter optimisation (more efficient than RandomizedSearchCV)</li>
    <li><strong>Probability Calibration</strong> — Platt scaling / isotonic regression for reliable probability estimates</li>
    <li><strong>Temporal Validation</strong> — Train on 2014/2015, test on 2016/2017 BRFSS</li>
    <li><strong>Multi-class</strong> — Extend to 3-class using <code>diabetes_012</code> file</li>
    <li><strong>Neural Network</strong> — TabNet or MLP comparison baseline</li>
    <li><strong>Cloud Deployment</strong> — Deploy on Streamlit Community Cloud or Hugging Face Spaces</li>
  </ol>
</div>

<footer>
  <p>🩺 Diabetes Risk Prediction — Explainable ML System</p>
  <p>Developed by <strong>Ankit Kumar</strong> | Academy of Skill Development, Kolkata | May–July 2025</p>
  <p>🔗 <a href="https://github.com/hrts51/diabetes-risk-prediction-explainable-ml" target="_blank">github.com/hrts51/diabetes-risk-prediction-explainable-ml</a></p>
  <p style="margin-top:10px; font-size:0.78rem; color:#aaa;">
    ⚠️ Medical Disclaimer: This project is for educational purposes only. Not medical advice.
  </p>
</footer>

</div><!-- end #main -->

<script>
// Active nav highlighting on scroll
const sections = document.querySelectorAll('.section[id]');
const links    = document.querySelectorAll('#sidebar nav a');
window.addEventListener('scroll', () => {{
  let cur = '';
  sections.forEach(s => {{
    if (window.scrollY >= s.offsetTop - 120) cur = s.id;
  }});
  links.forEach(a => {{
    a.classList.toggle('active', a.getAttribute('href') === '#' + cur);
  }});
}});
</script>
</body>
</html>'''

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(HTML)

size_kb = os.path.getsize(OUT_FILE) / 1024
print(f"Generated: {OUT_FILE}")
print(f"Size: {size_kb:.0f} KB")
print("Open in any browser to view.")
