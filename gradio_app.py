import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd
import requests

MODEL_PATH = "fraud_detection_model.pkl"
DATA_PATH = "synthetic_health_claims.csv"
API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000")

# --------------------------------------------------------------------------
# Load the same .pkl bundle FastAPI uses. Gradio doesn't call the model
# directly to make its verdict (that happens once, inside POST /claim, so
# there's a single source of truth) -- it loads the bundle locally only to
# read the model's name/version and to draw the "what the model weighs
# heaviest" panel from the trained pipeline's feature importances.
# --------------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
model_name = bundle["model_name"]
model_version = bundle.get("model_version", "1.0")

_ref = pd.read_csv(DATA_PATH)

CHOICES = {
    "Patient_Gender": sorted(_ref["Patient_Gender"].unique().tolist()),
    "Patient_State": sorted(_ref["Patient_State"].unique().tolist()),
    "Provider_Type": sorted(_ref["Provider_Type"].unique().tolist()),
    "Provider_Specialty": sorted(_ref["Provider_Specialty"].unique().tolist()),
    "Provider_State": sorted(_ref["Provider_State"].unique().tolist()),
    "Diagnosis_Code": sorted(_ref["Diagnosis_Code"].unique().tolist()),
    "Procedure_Code": sorted(_ref["Procedure_Code"].unique().tolist()),
    "Admission_Type": sorted(_ref["Admission_Type"].unique().tolist()),
    "Discharge_Type": sorted(_ref["Discharge_Type"].unique().tolist()),
    "Service_Type": sorted(_ref["Service_Type"].unique().tolist()),
    "Patient_City": sorted(_ref["Patient_City"].unique().tolist()),
    "Provider_City": sorted(_ref["Provider_City"].unique().tolist()),
}


# --------------------------------------------------------------------------
# Tab 1 — Assess & Save: calls POST /claim
# --------------------------------------------------------------------------
def submit_claim(
    claim_date, service_date, policy_expiration_date,
    claim_amount, patient_age, patient_gender, patient_city, patient_state,
    provider_type, provider_specialty, provider_city, provider_state,
    diagnosis_code, procedure_code, number_of_procedures,
    admission_type, discharge_type, length_of_stay_days, service_type,
    deductible_amount, copay_amount,
    num_previous_claims_patient, num_previous_claims_provider,
    provider_patient_distance_miles, claim_submitted_late,
):
    payload = {
        "claim_date": claim_date,
        "service_date": service_date,
        "policy_expiration_date": policy_expiration_date,
        "claim_amount": claim_amount,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "patient_city": patient_city,
        "patient_state": patient_state,
        "provider_type": provider_type,
        "provider_specialty": provider_specialty,
        "provider_city": provider_city,
        "provider_state": provider_state,
        "diagnosis_code": diagnosis_code,
        "procedure_code": procedure_code,
        "number_of_procedures": number_of_procedures,
        "admission_type": admission_type,
        "discharge_type": discharge_type,
        "length_of_stay_days": length_of_stay_days,
        "service_type": service_type,
        "deductible_amount": deductible_amount,
        "copay_amount": copay_amount,
        "num_previous_claims_patient": num_previous_claims_patient,
        "num_previous_claims_provider": num_previous_claims_provider,
        "provider_patient_distance_miles": provider_patient_distance_miles,
        "claim_submitted_late": bool(claim_submitted_late),
    }

    try:
        resp = requests.post(f"{API_URL}/claim", json=payload, timeout=10)
    except (requests.ConnectionError, requests.Timeout) as e:
        return _render_unreachable_error(e), ""

    if not resp.ok:
        return _render_http_error(resp), ""

    data = resp.json()

    result_html = _render_result_card(
        claim_id=data["claim_id"],
        prediction=data["prediction"],
        probability=data["probability"],
        tier=data["risk_tier"],
    )
    explanation_html = _render_explanation()
    return result_html, explanation_html


# --------------------------------------------------------------------------
# Tab 2 — Look Up Claim: calls GET /claim/{id}
# --------------------------------------------------------------------------
def lookup_claim(claim_id):
    if claim_id is None:
        return _render_lookup_placeholder("Enter a claim ID to search.")

    try:
        claim_id_int = int(claim_id)
    except (TypeError, ValueError):
        return _render_lookup_placeholder("Claim ID must be a whole number.")

    try:
        resp = requests.get(f"{API_URL}/claim/{claim_id_int}", timeout=10)
    except (requests.ConnectionError, requests.Timeout) as e:
        return _render_unreachable_error(e)

    if resp.status_code == 404:
        return f"""
        <div class="notfound-card">
          <div class="notfound-icon">&#128269;</div>
          <div class="notfound-title">No claim found</div>
          <p class="notfound-sub">There's no record with claim ID <strong>{claim_id_int}</strong>. Double-check the ID and try again.</p>
        </div>
        """

    if not resp.ok:
        return _render_http_error(resp)

    data = resp.json()
    return _render_result_card(
        claim_id=data["claim_id"],
        prediction=data["prediction"],
        probability=data["probability"],
        tier=data["risk_tier"],
        created_at=data.get("created_at", ""),
    )


# --------------------------------------------------------------------------
# Tab 3 — Claim history: calls GET /claims
# --------------------------------------------------------------------------
def fetch_history(limit):
    try:
        limit_int = int(limit) if limit else 50
    except (TypeError, ValueError):
        limit_int = 50

    try:
        resp = requests.get(f"{API_URL}/claims", params={"limit": limit_int}, timeout=10)
    except (requests.ConnectionError, requests.Timeout) as e:
        return _render_unreachable_error(e)

    if not resp.ok:
        return _render_http_error(resp)

    claims = resp.json()
    return _render_history_table(claims)


def _render_history_table(claims):
    if not claims:
        return """
        <div class="placeholder-card">
          No claims recorded yet. Submit one on the <strong>Assess &amp; save claim</strong> tab first.
        </div>
        """

    rows = ""
    for c in claims:
        tier = c["risk_tier"]
        is_fraud = c["prediction"] == "Fraudulent"
        verdict_class = "hist-flag" if is_fraud else "hist-clear"
        date = c["created_at"].split("T")[0] if c.get("created_at") else "—"
        rows += f"""
        <tr>
          <td class="hist-id">#{c['claim_id']}</td>
          <td>{date}</td>
          <td>{c['patient_age']}</td>
          <td>{c['provider_type']}</td>
          <td>${c['claim_amount']:,.2f}</td>
          <td><span class="hist-verdict {verdict_class}">{c['prediction']}</span></td>
          <td><span class="tier-chip tier-{tier}">{tier.capitalize()}</span></td>
          <td class="hist-mono">{c['probability']*100:.1f}%</td>
        </tr>
        """

    return f"""
    <div class="history-wrap">
      <table class="history-table">
        <thead>
          <tr>
            <th>Claim</th><th>Date</th><th>Age</th><th>Provider</th>
            <th>Amount</th><th>Verdict</th><th>Risk</th><th>Prob.</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    """


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def _render_result_card(claim_id, prediction, probability, tier, created_at=None):
    pct = probability * 100
    marker_pos = min(max(pct, 2), 98)
    is_fraud = prediction == "Fraudulent"
    verdict_class = "verdict-flag" if is_fraud else "verdict-clear"
    icon = "&#9888;" if is_fraud else "&#10003;"
    tier_label = {"low": "Low risk", "medium": "Medium risk", "high": "High risk"}[tier]
    timestamp_html = f'<div class="claim-meta">Recorded {created_at.split("T")[0]}</div>' if created_at else ""

    return f"""
    <div class="result-card tier-glow-{tier}">
      <div class="claim-id-chip">Claim #{claim_id}</div>
      <div class="result-top">
        <div class="verdict {verdict_class}">
          <span class="verdict-icon">{icon}</span>
          <span>{prediction}</span>
        </div>
        <div class="tier-chip tier-{tier}">{tier_label}</div>
      </div>

      <div class="gauge-wrap">
        <div class="gauge-track">
          <div class="gauge-seg seg-low"></div>
          <div class="gauge-seg seg-med"></div>
          <div class="gauge-seg seg-high"></div>
          <div class="gauge-marker" style="left:{marker_pos}%;">
            <div class="gauge-marker-dot dot-{tier}"></div>
            <div class="gauge-marker-value">{pct:.1f}%</div>
          </div>
        </div>
        <div class="gauge-labels">
          <span>Low</span><span>Medium</span><span>High</span>
        </div>
      </div>
      {timestamp_html}
    </div>
    """


def _render_explanation(top_n: int = 6) -> str:
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return '<p class="dim-note">Feature-level explanation is only available for tree-based models.</p>'

    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except (AttributeError, ValueError):
        feature_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]
    max_imp = importances[top_idx[0]] if len(top_idx) else 1

    palette = ["#6D5EF5", "#8B5CF6", "#A855F7", "#C026D3", "#DB2777", "#E11D48"]
    rows = ""
    for rank, i in enumerate(top_idx):
        name = feature_names[i].split("__")[-1].replace("_", " ")
        width = (importances[i] / max_imp) * 100
        color = palette[rank % len(palette)]
        rows += f"""
        <div class="factor-row">
          <span class="factor-name">{name}</span>
          <div class="factor-bar-track"><div class="factor-bar-fill" style="width:{width:.0f}%; background:{color};"></div></div>
        </div>
        """

    return f"""
    <div class="explain-card">
      <div class="explain-title">What the model weighs most heavily overall</div>
      {rows}
      <p class="dim-note">Global feature importance from {model_name} (v{model_version}), not a per-claim causal explanation.</p>
    </div>
    """


def _render_lookup_placeholder(message):
    return f"""
    <div class="placeholder-card">
      {message}
    </div>
    """


def _render_unreachable_error(e):
    """Only for genuine connection failures -- the API process isn't
    reachable at all (not running, wrong host/port, network down)."""
    return f"""
    <div class="notfound-card error-card">
      <div class="notfound-icon">&#9888;</div>
      <div class="notfound-title">Can't reach the API</div>
      <p class="notfound-sub">Make sure FastAPI is running at <code>{API_URL}</code> (e.g. <code>uvicorn main:app --reload</code>).<br/>{type(e).__name__}</p>
    </div>
    """


def _render_http_error(resp):
    """The API responded, but with an error status (4xx/5xx) -- e.g. a
    validation error on the payload. Surfaces the real detail instead of
    pretending the API is unreachable."""
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text

    return f"""
    <div class="notfound-card error-card">
      <div class="notfound-icon">&#9888;</div>
      <div class="notfound-title">API returned an error ({resp.status_code})</div>
      <p class="notfound-sub">{detail}</p>
    </div>
    """


PLACEHOLDER_RESULT = """
<div class="placeholder-card">
  Fill in the claim details and press <strong>Assess &amp; save claim</strong><br/>
  to score it and store it in the database.
</div>
"""

PLACEHOLDER_LOOKUP = """
<div class="placeholder-card">
  Enter a claim ID above and press <strong>Search</strong> to retrieve its verdict.
</div>
"""


# --------------------------------------------------------------------------
# Visual identity — vivid, high-contrast "review desk at night" palette
#
#   Canvas       #F6F5FC  – app background, faint violet tint
#   Ink          #1B1533  – primary text
#   Header grad  #4338CA → #9333EA → #DB2777  – hero banner gradient
#   Signal       #6D5EF5  – primary accent / buttons
#   Flag/High    #E11D48  – fraud flag, high risk, glow
#   Amber/Med    #F59E0B  – medium risk
#   Emerald/Low  #10B981  – cleared, low risk
#   Card white   #FFFFFF with soft violet-tinted shadow
#
# Typography: "Space Grotesk" (display) + "Inter" (UI) + "IBM Plex Mono"
# (numeric readouts / claim IDs).
# --------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --canvas: #F6F5FC;
  --ink: #1B1533;
  --signal: #6D5EF5;
  --signal-dark: #4F46E5;
  --flag: #E11D48;
  --amber: #F59E0B;
  --emerald: #10B981;
  --hairline: #E4E0F5;
}

.gradio-container {
  background: var(--canvas) !important;
  font-family: 'Inter', sans-serif !important;
}

/* ---------- Hero header ---------- */
#app-header {
  border-radius: 18px;
  padding: 34px 32px;
  margin-bottom: 22px;
  background: linear-gradient(120deg, #4338CA 0%, #7C3AED 55%, #DB2777 100%);
  box-shadow: 0 12px 30px -10px rgba(109, 94, 245, 0.55);
  position: relative;
  overflow: hidden;
}
#app-header::after {
  content: "";
  position: absolute;
  top: -60px; right: -60px;
  width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
}
#app-header .eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #F3D9FF;
  margin: 0 0 8px 0;
}
#app-header h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 32px;
  color: white;
  margin: 0 0 8px 0;
  letter-spacing: -0.01em;
}
#app-header .sub {
  font-size: 14.5px;
  color: #EAE3FF;
  max-width: 640px;
  line-height: 1.55;
  position: relative;
}

/* ---------- Tabs ---------- */
.tabs {
  border: none !important;
}
.tab-nav button {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14.5px !important;
  border-radius: 10px 10px 0 0 !important;
}

/* ---------- Section labels ---------- */
.section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--signal-dark);
  border-bottom: 2px solid var(--hairline);
  padding-bottom: 6px;
  margin-bottom: 10px !important;
  margin-top: 4px;
}

/* ---------- Form panel ---------- */
#form-panel {
  background: white;
  border: 1px solid var(--hairline);
  border-radius: 14px;
  padding: 20px 22px 8px 22px;
  box-shadow: 0 6px 20px -12px rgba(76, 29, 149, 0.18);
}

.gradio-container input, .gradio-container select, .gradio-container textarea {
  border-radius: 8px !important;
}

/* ---------- Buttons ---------- */
#predict-btn, #search-btn {
  background: linear-gradient(100deg, var(--signal) 0%, #C026D3 100%) !important;
  border: none !important;
  color: white !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em;
  border-radius: 9px !important;
  padding: 13px 0 !important;
  font-size: 15px !important;
  box-shadow: 0 6px 16px -6px rgba(109, 94, 245, 0.65);
  transition: transform 0.12s ease;
}
#predict-btn:hover, #search-btn:hover { transform: translateY(-1px); }

/* ---------- Result column ---------- */
#result-column { position: sticky; top: 16px; }

.result-card {
  background: linear-gradient(160deg, #1B1533 0%, #241A44 100%);
  color: white;
  border-radius: 16px;
  padding: 22px 22px 18px 22px;
  position: relative;
}
.tier-glow-low    { box-shadow: 0 0 0 1px rgba(16,185,129,0.35), 0 18px 40px -18px rgba(16,185,129,0.55); }
.tier-glow-medium { box-shadow: 0 0 0 1px rgba(245,158,11,0.35), 0 18px 40px -18px rgba(245,158,11,0.55); }
.tier-glow-high   { box-shadow: 0 0 0 1px rgba(225,29,72,0.4), 0 18px 40px -18px rgba(225,29,72,0.6); }

.claim-id-chip {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  letter-spacing: 0.05em;
  color: #C9BFFF;
  background: rgba(109,94,245,0.22);
  border: 1px solid rgba(109,94,245,0.4);
  padding: 3px 10px;
  border-radius: 20px;
  margin-bottom: 14px;
}

.result-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
}
.verdict {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 21px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 8px;
}
.verdict-icon { font-size: 18px; }
.verdict-flag { color: #FF7A9C; }
.verdict-clear { color: #6EE7B7; }

.tier-chip {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 5px 11px;
  border-radius: 20px;
  font-weight: 700;
}
.tier-low    { background: rgba(16,185,129,0.22); color: #6EE7B7; }
.tier-medium { background: rgba(245,158,11,0.22); color: #FCD34D; }
.tier-high   { background: rgba(225,29,72,0.25); color: #FF7A9C; }

.gauge-wrap { margin: 22px 0 6px 0; }
.gauge-track {
  position: relative;
  height: 11px;
  border-radius: 6px;
  overflow: visible;
  display: flex;
}
.gauge-seg { height: 100%; }
.seg-low  { width: 40%; background: linear-gradient(90deg,#10B981,#34D399); border-radius: 6px 0 0 6px; }
.seg-med  { width: 35%; background: linear-gradient(90deg,#F59E0B,#FBBF24); }
.seg-high { width: 25%; background: linear-gradient(90deg,#E11D48,#F43F5E); border-radius: 0 6px 6px 0; }

.gauge-marker {
  position: absolute;
  top: -10px;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
}
.gauge-marker-dot {
  width: 15px; height: 15px;
  background: white;
  border: 3px solid var(--ink);
  border-radius: 50%;
}
.dot-low    { box-shadow: 0 0 0 5px rgba(16,185,129,0.35); }
.dot-medium { box-shadow: 0 0 0 5px rgba(245,158,11,0.35); }
.dot-high   { box-shadow: 0 0 0 5px rgba(225,29,72,0.4); }
.gauge-marker-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  margin-top: 5px;
  background: white;
  color: var(--ink);
  padding: 1px 7px;
  border-radius: 4px;
}

.gauge-labels {
  display: flex;
  justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #A79BE0;
  margin-top: 8px;
  padding: 0 2px;
}

.claim-meta {
  font-size: 12px;
  color: #A79BE0;
  margin-top: 14px;
  border-top: 1px solid rgba(255,255,255,0.14);
  padding-top: 12px;
}

/* ---------- Explanation card ---------- */
.explain-card {
  background: white;
  border: 1px solid var(--hairline);
  border-radius: 14px;
  padding: 18px 20px;
  margin-top: 14px;
  box-shadow: 0 6px 20px -14px rgba(76, 29, 149, 0.18);
}
.explain-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--signal-dark);
  margin-bottom: 14px;
}
.factor-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.factor-name {
  font-size: 12.5px; color: var(--ink); width: 42%;
  text-transform: capitalize; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.factor-bar-track { flex: 1; background: var(--canvas); border-radius: 4px; height: 8px; }
.factor-bar-fill { height: 100%; border-radius: 4px; }
.dim-note { font-size: 11.5px; color: #8B84A8; margin-top: 10px; }

/* ---------- Placeholder / not-found / error cards ---------- */
.placeholder-card {
  background: white;
  border: 1.5px dashed #C9BFFF;
  border-radius: 14px;
  padding: 40px 24px;
  text-align: center;
  color: #6B6390;
  font-size: 13.5px;
}
.notfound-card {
  background: white;
  border: 1px solid var(--hairline);
  border-radius: 14px;
  padding: 34px 24px;
  text-align: center;
  box-shadow: 0 6px 20px -14px rgba(76, 29, 149, 0.18);
}
.notfound-icon { font-size: 30px; margin-bottom: 10px; }
.notfound-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 18px;
  color: var(--ink);
  margin-bottom: 8px;
}
.notfound-sub { font-size: 13px; color: #6B6390; line-height: 1.6; }
.error-card .notfound-title { color: var(--flag); }

/* ---------- Lookup panel ---------- */
#lookup-panel {
  background: white;
  border: 1px solid var(--hairline);
  border-radius: 14px;
  padding: 22px;
  box-shadow: 0 6px 20px -12px rgba(76, 29, 149, 0.18);
}

/* ---------- History table ---------- */
.history-wrap {
  background: white;
  border: 1px solid var(--hairline);
  border-radius: 14px;
  padding: 8px;
  box-shadow: 0 6px 20px -12px rgba(76, 29, 149, 0.18);
  overflow-x: auto;
}
.history-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.history-table thead th {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--signal-dark);
  text-align: left;
  padding: 10px 14px;
  border-bottom: 2px solid var(--hairline);
  white-space: nowrap;
}
.history-table tbody td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--hairline);
  color: var(--ink);
  white-space: nowrap;
}
.history-table tbody tr:hover { background: var(--canvas); }
.history-table tbody tr:last-child td { border-bottom: none; }
.hist-id {
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 600;
  color: var(--signal-dark);
}
.hist-mono { font-family: 'IBM Plex Mono', monospace; }
.hist-verdict {
  font-weight: 700;
  font-size: 12.5px;
  padding: 3px 9px;
  border-radius: 20px;
}
.hist-flag { background: rgba(225,29,72,0.12); color: #C0184A; }
.hist-clear { background: rgba(16,185,129,0.12); color: #0A8F5F; }

#history-controls {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  margin-bottom: 16px;
}
"""


# --------------------------------------------------------------------------
# Build the Gradio interface
# --------------------------------------------------------------------------
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Claims Fraud Review Desk", theme=theme, css=CUSTOM_CSS) as demo:

    gr.HTML(f"""
    <div id="app-header">
      <p class="eyebrow">Health Insurance &middot; Special Investigations Unit</p>
      <h1>Claims Fraud Review Desk</h1>
      <p class="sub">Score a claim with <strong>{model_name}</strong>, save it to the database,
      look any claim back up by ID, or browse the full claim history — all backed
      by the FastAPI service at <code style="color:#FBEFFF">{API_URL}</code>.</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("🩺  Assess & save claim"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_id="form-panel"):

                    gr.Markdown("Dates", elem_classes="section-label")
                    with gr.Row():
                        claim_date = gr.Textbox(label="Claim date (YYYY-MM-DD)", value="2024-06-01")
                        service_date = gr.Textbox(label="Service date (YYYY-MM-DD)", value="2024-05-20")
                        policy_expiration_date = gr.Textbox(label="Policy expiration date (YYYY-MM-DD)", value="2026-01-01")

                    gr.Markdown("Financials", elem_classes="section-label")
                    with gr.Row():
                        claim_amount = gr.Number(label="Claim amount ($)", value=5000)
                        deductible_amount = gr.Number(label="Deductible amount ($)", value=500)
                        copay_amount = gr.Number(label="Co-pay amount ($)", value=50)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("Patient", elem_classes="section-label")
                            patient_age = gr.Number(label="Patient age", value=40, precision=0)
                            patient_gender = gr.Dropdown(CHOICES["Patient_Gender"], label="Patient gender", value=CHOICES["Patient_Gender"][0])
                            patient_city = gr.Dropdown(CHOICES["Patient_City"], label="Patient city", value=CHOICES["Patient_City"][0])
                            patient_state = gr.Dropdown(CHOICES["Patient_State"], label="Patient state", value=CHOICES["Patient_State"][0])
                            num_previous_claims_patient = gr.Number(label="Previous claims by patient", value=0, precision=0)

                        with gr.Column():
                            gr.Markdown("Provider", elem_classes="section-label")
                            provider_type = gr.Dropdown(CHOICES["Provider_Type"], label="Provider type", value=CHOICES["Provider_Type"][0])
                            provider_specialty = gr.Dropdown(CHOICES["Provider_Specialty"], label="Provider specialty", value=CHOICES["Provider_Specialty"][0])
                            provider_city = gr.Dropdown(CHOICES["Provider_City"], label="Provider city", value=CHOICES["Provider_City"][0])
                            provider_state = gr.Dropdown(CHOICES["Provider_State"], label="Provider state", value=CHOICES["Provider_State"][0])
                            num_previous_claims_provider = gr.Number(label="Previous claims by provider", value=0, precision=0)

                    gr.Markdown("Claim &amp; service details", elem_classes="section-label")
                    with gr.Row():
                        diagnosis_code = gr.Dropdown(CHOICES["Diagnosis_Code"], label="Diagnosis code", value=CHOICES["Diagnosis_Code"][0])
                        procedure_code = gr.Dropdown(CHOICES["Procedure_Code"], label="Procedure code", value=CHOICES["Procedure_Code"][0])
                        number_of_procedures = gr.Number(label="Number of procedures", value=1, precision=0)
                    with gr.Row():
                        admission_type = gr.Dropdown(CHOICES["Admission_Type"], label="Admission type", value=CHOICES["Admission_Type"][0])
                        discharge_type = gr.Dropdown(CHOICES["Discharge_Type"], label="Discharge type", value=CHOICES["Discharge_Type"][0])
                        service_type = gr.Dropdown(CHOICES["Service_Type"], label="Service type", value=CHOICES["Service_Type"][0])
                    with gr.Row():
                        length_of_stay_days = gr.Number(label="Length of stay (days)", value=1, precision=0)
                        provider_patient_distance_miles = gr.Number(label="Provider-patient distance (miles)", value=20)
                        claim_submitted_late = gr.Checkbox(label="Claim submitted late?", value=False)

                    predict_btn = gr.Button("Assess & save claim", elem_id="predict-btn")
                    gr.Markdown("&nbsp;")

                with gr.Column(scale=2, elem_id="result-column"):
                    result_html = gr.HTML(PLACEHOLDER_RESULT)
                    explanation_html = gr.HTML("")

            predict_btn.click(
                fn=submit_claim,
                inputs=[
                    claim_date, service_date, policy_expiration_date,
                    claim_amount, patient_age, patient_gender, patient_city, patient_state,
                    provider_type, provider_specialty, provider_city, provider_state,
                    diagnosis_code, procedure_code, number_of_procedures,
                    admission_type, discharge_type, length_of_stay_days, service_type,
                    deductible_amount, copay_amount,
                    num_previous_claims_patient, num_previous_claims_provider,
                    provider_patient_distance_miles, claim_submitted_late,
                ],
                outputs=[result_html, explanation_html],
            )

        with gr.Tab("🔍  Look up claim"):
            with gr.Row():
                with gr.Column(scale=1, elem_id="lookup-panel"):
                    gr.Markdown("Find a claim", elem_classes="section-label")
                    claim_id_input = gr.Number(label="Claim ID", precision=0)
                    search_btn = gr.Button("Search", elem_id="search-btn")
                with gr.Column(scale=1):
                    lookup_result_html = gr.HTML(PLACEHOLDER_LOOKUP)

            search_btn.click(
                fn=lookup_claim,
                inputs=[claim_id_input],
                outputs=[lookup_result_html],
            )
            claim_id_input.submit(
                fn=lookup_claim,
                inputs=[claim_id_input],
                outputs=[lookup_result_html],
            )

        with gr.Tab("📋  Claim history"):
            with gr.Row(elem_id="history-controls"):
                history_limit = gr.Number(label="Show last N claims", value=50, precision=0, scale=0, min_width=160)
                refresh_btn = gr.Button("Refresh", elem_id="search-btn", scale=0, min_width=120)
            history_table = gr.HTML(_render_lookup_placeholder("Loading claim history…"))

            refresh_btn.click(
                fn=fetch_history,
                inputs=[history_limit],
                outputs=[history_table],
            )
            history_limit.submit(
                fn=fetch_history,
                inputs=[history_limit],
                outputs=[history_table],
            )
            demo.load(
                fn=fetch_history,
                inputs=[history_limit],
                outputs=[history_table],
            )

if __name__ == "__main__":
    # server_name="0.0.0.0" is required for Docker's -p port mapping to reach
    # this process -- the Gradio default of 127.0.0.1 only accepts
    # connections from inside the container itself.
    gradio_share = os.getenv("GRADIO_SHARE", "true").lower() == "true"
    demo.launch(server_name="0.0.0.0", server_port=7860, share=gradio_share)