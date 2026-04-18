"""
VentureVerse – Flask Web Application (v6)
==========================================
Pages: Login, Signup, Predict, Charts, Insights, About
Features: PDF download, full state names, capitalized industries
"""

from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib, json, numpy as np, pandas as pd
import sqlite3, hashlib, os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

DB_FILE = "ventureverse.db"
MODEL_FILE = "ventureverse_model.joblib"
RESULTS_FILE = "model_results_summary.json"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        prediction_score REAL, pred_label TEXT, input_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id))""")
    conn.commit(); conn.close()


def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()


init_db()
model = joblib.load(MODEL_FILE)
try:
    with open(RESULTS_FILE) as f: model_results = json.load(f)
except: model_results = None

# Full state names for display
STATE_MAP = {
    "CA": "California", "NY": "New York", "MA": "Massachusetts",
    "TX": "Texas", "WA": "Washington", "CO": "Colorado",
    "IL": "Illinois", "FL": "Florida", "other": "Other"
}
STATES = list(STATE_MAP.keys())

# Capitalized industry names for display
INDUSTRY_MAP = {
    "biotech": "Biotech", "consulting": "Consulting", "ecommerce": "E-Commerce",
    "enterprise": "Enterprise", "games_video": "Games & Video", "mobile": "Mobile",
    "software": "Software", "web": "Web", "advertising": "Advertising", "other": "Other"
}
CATEGORIES = list(INDUSTRY_MAP.keys())


def build_input_df(form):
    funding_total = float(form.get("funding_total_usd", 0))
    funding_rounds = int(form.get("funding_rounds", 1))
    relationships = int(form.get("relationships", 0))
    milestones = int(form.get("milestones", 0))
    avg_participants = float(form.get("avg_participants", 1.0))
    age_first_fund = float(form.get("age_first_funding_year", 0))
    age_last_fund = float(form.get("age_last_funding_year", 0))
    age_first_mile = float(form.get("age_first_milestone_year", 0)) or np.nan
    age_last_mile = float(form.get("age_last_milestone_year", 0)) or np.nan
    has_vc = int(form.get("has_VC", 0)); has_angel = int(form.get("has_angel", 0))
    has_rA = int(form.get("has_roundA", 0)); has_rB = int(form.get("has_roundB", 0))
    has_rC = int(form.get("has_roundC", 0)); has_rD = int(form.get("has_roundD", 0))
    is_top = int(form.get("is_top500", 0))
    category = form.get("category_code", "other"); state = form.get("state_code", "other")
    funding_duration = max(age_last_fund - age_first_fund, 0)
    safe_rounds = funding_rounds if funding_rounds > 0 else 1
    return pd.DataFrame([{
        "age_first_funding_year": age_first_fund, "age_last_funding_year": age_last_fund,
        "age_first_milestone_year": age_first_mile, "age_last_milestone_year": age_last_mile,
        "relationships": relationships, "funding_rounds": funding_rounds,
        "funding_total_usd": funding_total, "milestones": milestones,
        "avg_participants": avg_participants, "funding_duration": funding_duration,
        "avg_funding_per_round": funding_total / safe_rounds,
        "log_funding": np.log1p(funding_total),
        "has_VC": has_vc, "has_angel": has_angel,
        "has_roundA": has_rA, "has_roundB": has_rB, "has_roundC": has_rC, "has_roundD": has_rD,
        "is_top500": is_top, "category_code": category, "state_code": state,
    }])


def compute_risk_breakdown(form):
    factors = []
    funding = float(form.get("funding_total_usd", 0))
    if funding >= 10e6: factors.append({"factor": "Total Funding", "score": 90, "status": "strong"})
    elif funding >= 2e6: factors.append({"factor": "Total Funding", "score": 65, "status": "moderate"})
    elif funding >= 5e5: factors.append({"factor": "Total Funding", "score": 40, "status": "moderate"})
    else: factors.append({"factor": "Total Funding", "score": 15, "status": "weak"})
    rounds = int(form.get("funding_rounds", 0))
    if rounds >= 4: factors.append({"factor": "Funding Rounds", "score": 85, "status": "strong"})
    elif rounds >= 2: factors.append({"factor": "Funding Rounds", "score": 55, "status": "moderate"})
    else: factors.append({"factor": "Funding Rounds", "score": 20, "status": "weak"})
    has_vc = int(form.get("has_VC", 0)); is_top = int(form.get("is_top500", 0))
    if is_top: factors.append({"factor": "Investor Quality", "score": 95, "status": "strong"})
    elif has_vc: factors.append({"factor": "Investor Quality", "score": 70, "status": "strong"})
    elif int(form.get("has_angel", 0)): factors.append({"factor": "Investor Quality", "score": 45, "status": "moderate"})
    else: factors.append({"factor": "Investor Quality", "score": 10, "status": "weak"})
    rels = int(form.get("relationships", 0))
    if rels >= 10: factors.append({"factor": "Network Strength", "score": 85, "status": "strong"})
    elif rels >= 4: factors.append({"factor": "Network Strength", "score": 55, "status": "moderate"})
    else: factors.append({"factor": "Network Strength", "score": 20, "status": "weak"})
    miles = int(form.get("milestones", 0))
    if miles >= 3: factors.append({"factor": "Early Traction", "score": 80, "status": "strong"})
    elif miles >= 1: factors.append({"factor": "Early Traction", "score": 50, "status": "moderate"})
    else: factors.append({"factor": "Early Traction", "score": 10, "status": "weak"})
    state = form.get("state_code", "other")
    if state == "CA": factors.append({"factor": "Location", "score": 85, "status": "strong"})
    elif state in ("NY", "MA", "WA"): factors.append({"factor": "Location", "score": 65, "status": "moderate"})
    else: factors.append({"factor": "Location", "score": 35, "status": "moderate"})
    return factors


def generate_insights(form, prediction, pred_label, risk_factors):
    insights = []
    if prediction >= 75:
        insights.append({"title": "Strong Success Indicators", "icon": "&#9733;", "type": "positive",
            "text": f"This startup profile shows a {prediction}% success probability, significantly above the 50% threshold. The combination of factors suggests strong market readiness."})
    elif prediction >= 50:
        insights.append({"title": "Moderate Success Potential", "icon": "&#9888;", "type": "neutral",
            "text": f"At {prediction}%, this startup is above the success threshold but has room for improvement. Strengthening weaker factors could push probability higher."})
    else:
        insights.append({"title": "High Risk Profile", "icon": "&#9888;", "type": "negative",
            "text": f"With {prediction}% success probability, this profile indicates elevated failure risk. Key areas need immediate attention."})

    funding = float(form.get("funding_total_usd", 0)); rounds = int(form.get("funding_rounds", 0))
    if funding >= 5e6 and rounds >= 3:
        insights.append({"title": "Funding Strength", "icon": "&#128176;", "type": "positive",
            "text": f"Total funding of ${funding:,.0f} across {rounds} rounds demonstrates strong investor confidence and repeated due diligence validation."})
    elif funding < 5e5:
        insights.append({"title": "Funding Gap", "icon": "&#128176;", "type": "negative",
            "text": f"Total funding of ${funding:,.0f} is below the typical threshold. Startups with under $500K historically have significantly higher failure rates."})
    else:
        insights.append({"title": "Moderate Funding", "icon": "&#128176;", "type": "neutral",
            "text": f"Total funding of ${funding:,.0f} across {rounds} round(s) is moderate. Consider pursuing additional rounds to strengthen the profile."})

    has_vc = int(form.get("has_VC", 0)); is_top = int(form.get("is_top500", 0))
    if is_top:
        insights.append({"title": "Elite VC Backing", "icon": "&#127942;", "type": "positive",
            "text": "Backed by a Top-500 VC firm — one of the strongest success signals. Elite VCs provide capital, strategic guidance, and credibility."})
    elif has_vc and int(form.get("has_angel", 0)):
        insights.append({"title": "Diversified Investors", "icon": "&#127942;", "type": "positive",
            "text": "Both VC and angel investors provide a balanced funding structure with early validation and institutional support."})
    elif not has_vc and not int(form.get("has_angel", 0)):
        insights.append({"title": "No Institutional Backing", "icon": "&#127942;", "type": "negative",
            "text": "No VC or angel backing detected. Startups without institutional investors face lower acquisition rates."})

    rels = int(form.get("relationships", 0))
    if rels >= 8:
        insights.append({"title": "Strong Network", "icon": "&#128101;", "type": "positive",
            "text": f"{rels} key connections — a robust network of advisors and co-founders correlates with better mentorship and deal flow."})
    elif rels <= 2:
        insights.append({"title": "Limited Network", "icon": "&#128101;", "type": "negative",
            "text": f"Only {rels} connection(s). Expanding the advisory board could significantly improve outcomes."})

    miles = int(form.get("milestones", 0))
    if miles == 0:
        insights.append({"title": "No Milestones", "icon": "&#127937;", "type": "negative",
            "text": "No milestones recorded. Product demos, partnerships, or press coverage are important signals investors look for."})
    elif miles >= 3:
        insights.append({"title": "Strong Traction", "icon": "&#127937;", "type": "positive",
            "text": f"{miles} milestones demonstrate execution capability and market validation."})

    state = form.get("state_code", "other")
    state_name = STATE_MAP.get(state, state)
    if state == "CA":
        insights.append({"title": "Silicon Valley Advantage", "icon": "&#128205;", "type": "positive",
            "text": f"Based in {state_name} — the world's largest startup ecosystem with unmatched investor density and talent pool."})
    elif state in ("NY", "MA", "WA"):
        insights.append({"title": f"{state_name} Ecosystem", "icon": "&#128205;", "type": "neutral",
            "text": f"Located in {state_name}, a strong secondary startup hub with solid ecosystem support."})

    weak = [rf for rf in risk_factors if rf["status"] == "weak"]
    if weak:
        names = ", ".join([w["factor"] for w in weak])
        insights.append({"title": "Key Recommendations", "icon": "&#128161;", "type": "action",
            "text": f"Priority areas: {names}. Focus on the lowest-scoring factor first for maximum impact on success probability."})
    return insights


def get_model_comparison():
    if not model_results or "all_model_results" not in model_results: return None
    names, roc, acc, f1s = [], [], [], []
    for m in model_results["all_model_results"]:
        names.append(m["name"])
        roc.append(round(m.get("cv_roc_auc_mean", 0) * 100, 1))
        acc.append(round(m.get("cv_accuracy_mean", 0) * 100, 1))
        f1s.append(round(m.get("cv_f1_mean", 0) * 100, 1))
    return {"names": names, "roc_aucs": roc, "accuracies": acc, "f1s": f1s, "winner": model_results.get("winner", "")}


def get_prediction_history(user_id):
    try:
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("SELECT prediction_score, pred_label, input_data, created_at FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 10", (user_id,))
        rows = c.fetchall(); conn.close()
        result = []
        for r in rows:
            dt = r[3] or ""
            try:
                dt_obj = datetime.strptime(dt[:19], "%Y-%m-%d %H:%M:%S")
                dt_display = dt_obj.strftime("%d %b %Y, %I:%M %p")
            except: dt_display = dt[:16] if dt else "-"
            result.append({"score": r[0], "label": r[1], "data": json.loads(r[2]) if r[2] else {}, "date": dt_display})
        return result
    except: return []


# ── AUTH ─────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip(); pw = request.form.get("password", "")
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("SELECT id, full_name, password_hash FROM users WHERE email=?", (email,))
        user = c.fetchone(); conn.close()
        if user and user[2] == hash_password(pw):
            session["user_id"] = user[0]; session["user_name"] = user[1]; session["user_email"] = email
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html", error=None)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        pw = request.form.get("password", ""); confirm = request.form.get("confirm_password", "")
        if not name or not email or not pw: return render_template("signup.html", error="All fields required.")
        if pw != confirm: return render_template("signup.html", error="Passwords do not match.")
        if len(pw) < 6: return render_template("signup.html", error="Password must be 6+ characters.")
        try:
            conn = sqlite3.connect(DB_FILE); c = conn.cursor()
            c.execute("INSERT INTO users (full_name, email, password_hash) VALUES (?,?,?)", (name, email, hash_password(pw)))
            conn.commit(); uid = c.lastrowid; conn.close()
            session["user_id"] = uid; session["user_name"] = name; session["user_email"] = email
            return redirect(url_for("home"))
        except sqlite3.IntegrityError: return render_template("signup.html", error="Email already registered.")
    return render_template("signup.html", error=None)

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("home"))


# ── LANDING + PREDICT ────────────────────────────────
@app.route("/")
def home():
    if "user_id" not in session:
        return render_template("landing.html")
    return render_template("index.html", prediction=session.get("last_prediction"), pred_label=session.get("last_pred_label"), error=None,
        categories=CATEGORIES, states=STATES, state_map=STATE_MAP, industry_map=INDUSTRY_MAP,
        form_data=session.get("last_form_data", {}), risk_factors=session.get("last_risk_factors"), user_name=session.get("user_name"), active_page="predict")

@app.route("/reset")
def reset():
    session.pop("last_prediction", None); session.pop("last_pred_label", None)
    session.pop("last_form_data", None); session.pop("last_risk_factors", None)
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session: return redirect(url_for("login"))
    try:
        form_data = {k: request.form.get(k, "") for k in request.form}
        
        required_fields = ['funding_total_usd', 'funding_rounds']
        for field in required_fields:
            if not form_data.get(field, "").strip():
                return render_template("index.html", prediction=None, pred_label=None, error="All fields are required. Please complete the form.",
                    categories=CATEGORIES, states=STATES, state_map=STATE_MAP, industry_map=INDUSTRY_MAP,
                    form_data=form_data, risk_factors=None, user_name=session.get("user_name"), active_page="predict")

        proba = model.predict_proba(build_input_df(form_data))[0][1]
        pred_pct = round(proba * 100, 2)
        pred_label = "Success" if proba >= 0.5 else "Failure"
        risk_factors = compute_risk_breakdown(form_data)
        session["last_prediction"] = pred_pct; session["last_pred_label"] = pred_label
        session["last_form_data"] = form_data; session["last_risk_factors"] = risk_factors
        try:
            conn = sqlite3.connect(DB_FILE); c = conn.cursor()
            c.execute("INSERT INTO predictions (user_id, prediction_score, pred_label, input_data) VALUES (?,?,?,?)",
                (session["user_id"], pred_pct, pred_label, json.dumps(form_data))); conn.commit(); conn.close()
        except: pass
        return render_template("index.html", prediction=pred_pct, pred_label=pred_label, error=None,
            categories=CATEGORIES, states=STATES, state_map=STATE_MAP, industry_map=INDUSTRY_MAP,
            form_data=form_data, risk_factors=risk_factors, user_name=session.get("user_name"), active_page="predict")
    except Exception as e:
        form_data = {k: request.form.get(k, "") for k in request.form}
        return render_template("index.html", prediction=None, pred_label=None, error=str(e),
            categories=CATEGORIES, states=STATES, state_map=STATE_MAP, industry_map=INDUSTRY_MAP,
            form_data=form_data, risk_factors=None, user_name=session.get("user_name"), active_page="predict")


# ── CHARTS ───────────────────────────────────────────
@app.route("/charts")
def charts():
    if "user_id" not in session: return redirect(url_for("login"))
    return render_template("charts.html", prediction=session.get("last_prediction"),
        pred_label=session.get("last_pred_label"), risk_factors=session.get("last_risk_factors"),
        model_comparison=get_model_comparison(), user_name=session.get("user_name"), active_page="charts")


# ── INSIGHTS ─────────────────────────────────────────
@app.route("/insights")
def insights():
    if "user_id" not in session: return redirect(url_for("login"))
    pred = session.get("last_prediction"); label = session.get("last_pred_label")
    form_data = session.get("last_form_data", {}); rf = session.get("last_risk_factors", [])
    cards = generate_insights(form_data, pred, label, rf) if pred else []
    return render_template("insights.html", prediction=pred, pred_label=label,
        insights=cards, history=get_prediction_history(session["user_id"]),
        state_map=STATE_MAP, industry_map=INDUSTRY_MAP,
        user_name=session.get("user_name"), active_page="insights")


# ── DOWNLOAD PDF ─────────────────────────────────────
@app.route("/download-insights")
def download_insights():
    if "user_id" not in session: return redirect(url_for("login"))
    pred = session.get("last_prediction"); label = session.get("last_pred_label")
    form_data = session.get("last_form_data", {}); rf = session.get("last_risk_factors", [])
    if not pred: return redirect(url_for("insights"))

    cards = generate_insights(form_data, pred, label, rf)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm

        filepath = f"/tmp/ventureverse_insights_{session['user_id']}.pdf"
        doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=25*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=22, textColor=colors.HexColor('#1a1a2e'))
        heading_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#1a1a2e'))
        body_style = ParagraphStyle('Body2', parent=styles['Normal'], fontSize=11, leading=16)

        story = []
        story.append(Paragraph("VentureVerse — Prediction Insights Report", title_style))
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph(f"Prediction: <b>{pred}%</b> — <b>{label}</b>", heading_style))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", body_style))
        story.append(Spacer(1, 8*mm))

        for c in cards:
            badge = "Strength" if c["type"]=="positive" else ("Risk" if c["type"]=="negative" else ("Action" if c["type"]=="action" else "Note"))
            story.append(Paragraph(f"<b>{c['title']}</b>  [{badge}]", heading_style))
            story.append(Paragraph(c["text"], body_style))
            story.append(Spacer(1, 4*mm))

        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("Factor Strength Breakdown", heading_style))
        if rf:
            tdata = [["Factor", "Score", "Status"]]
            for r in rf: tdata.append([r["factor"], f"{r['score']}%", r["status"].capitalize()])
            t = Table(tdata, colWidths=[55*mm, 25*mm, 30*mm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f8f8'), colors.white]),
            ]))
            story.append(t)

        story.append(Spacer(1, 10*mm))
        story.append(Paragraph("University of Westminster — BSc Computer Science Final Project", body_style))

        doc.build(story)
        return send_file(filepath, as_attachment=True, download_name="VentureVerse_Insights.pdf")
    except ImportError:
        # Fallback: plain text
        filepath = f"/tmp/ventureverse_insights_{session['user_id']}.txt"
        with open(filepath, "w") as f:
            f.write(f"VentureVerse Insights Report\nPrediction: {pred}% — {label}\n\n")
            for c in cards: f.write(f"[{c['type'].upper()}] {c['title']}\n{c['text']}\n\n")
        return send_file(filepath, as_attachment=True, download_name="VentureVerse_Insights.txt")


# ── ABOUT ────────────────────────────────────────────
@app.route("/about")
def about():
    return render_template("about.html", user_name=session.get("user_name"), active_page="about")


# ── CHATBOT (Gemini API) ─────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

@app.route("/chatbot")
def chatbot():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if not session.get("chat_history"):
        session["chat_history"] = []
    return render_template("chatbot.html",
        chat_history=session.get("chat_history", []),
        user_name=session.get("user_name"), active_page="chatbot",
        api_configured=bool(GEMINI_API_KEY))


@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return {"error": "Not logged in"}, 401

    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return {"error": "Empty message"}, 400

    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured. Set GEMINI_API_KEY environment variable."}, 500

    # Build context from last prediction
    prediction = session.get("last_prediction")
    pred_label = session.get("last_pred_label")
    form_data = session.get("last_form_data", {})
    risk_factors = session.get("last_risk_factors", [])

    context_parts = [
        "You are VentureBot, an AI startup advisor integrated into VentureVerse — a startup success prediction platform.",
        "You provide actionable, concise startup advice. Keep responses under 200 words unless the user asks for detail.",
        "Be encouraging but honest. Use specific data points when available.",
    ]

    if prediction is not None:
        funding = form_data.get("funding_total_usd", "N/A")
        industry = INDUSTRY_MAP.get(form_data.get("category_code", ""), "N/A")
        state = STATE_MAP.get(form_data.get("state_code", ""), "N/A")
        rounds = form_data.get("funding_rounds", "N/A")

        context_parts.append(f"\nThe user just ran a prediction with these results:")
        context_parts.append(f"- Success Probability: {prediction}% ({pred_label})")
        context_parts.append(f"- Total Funding: ${float(funding):,.0f}" if funding != "N/A" else "")
        context_parts.append(f"- Funding Rounds: {rounds}")
        context_parts.append(f"- Industry: {industry}")
        context_parts.append(f"- Location: {state}")

        if risk_factors:
            weak = [rf["factor"] for rf in risk_factors if rf["status"] == "weak"]
            strong = [rf["factor"] for rf in risk_factors if rf["status"] == "strong"]
            if weak:
                context_parts.append(f"- Weak factors: {', '.join(weak)}")
            if strong:
                context_parts.append(f"- Strong factors: {', '.join(strong)}")

    system_prompt = "\n".join(context_parts)

    # Build conversation history for Gemini
    chat_history = session.get("chat_history", [])

    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build contents list with history
        contents = []
        for msg in chat_history[-10:]:  # Last 10 messages for context
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["text"]}]
            })
        contents.append({"role": "user", "parts": [{"text": user_msg}]})

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.7,
                "max_output_tokens": 500,
            }
        )

        bot_reply = response.text

        # Save to session history
        chat_history.append({"role": "user", "text": user_msg})
        chat_history.append({"role": "bot", "text": bot_reply})
        session["chat_history"] = chat_history[-20:]  # Keep last 20

        return {"reply": bot_reply}

    except ImportError:
        return {"error": "google-genai package not installed. Run: pip install google-genai"}, 500
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/chat/clear", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    return {"status": "cleared"}


if __name__ == "__main__":
    app.run(debug=True)
