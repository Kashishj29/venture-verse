"""
VentureVerse – Flask Web Application
======================================
This is the main backend file that powers the entire VentureVerse website.

What this app does:
  1. Lets users sign up, log in, and log out  (authentication)
  2. Collects startup details via a form       (prediction input)
  3. Feeds the data into an ML model           (XGBoost pipeline)
  4. Shows a success probability + insights    (prediction output)
  5. Generates charts and a downloadable PDF   (visualisation)
  6. Provides an AI chatbot via Google Gemini  (chatbot)

Tech stack:
  - Flask     → lightweight Python web framework
  - SQLite    → simple file-based database (no server needed)
  - joblib    → loads the pre-trained ML model from disk
  - pandas    → builds the input dataframe for the model
  - reportlab → generates PDF reports
  - Gemini    → Google's AI for the chatbot feature

Author : Kashish Jadhav (w2035589)
Module : 6COSC023W — BSc Computer Science Final Project
Uni    : University of Westminster, 2025–2026
"""

# ═══════════════════════════════════════════════════════════════
#  IMPORTS — libraries we need to run the app
# ═══════════════════════════════════════════════════════════════

from flask import (
    Flask,              # The web framework itself
    render_template,    # Renders HTML pages with dynamic data
    request,            # Reads data from forms and URLs
    redirect,           # Sends the user to a different page
    url_for,            # Generates URLs for Flask routes
    session,            # Stores per-user data (like "logged in")
    send_file,          # Sends files (e.g. PDF) to the browser
)

import joblib           # Loads the saved ML model (.joblib file)
import json             # Reads/writes JSON data
import numpy as np      # Maths library (e.g. log transform)
import pandas as pd     # Creates dataframes for the ML model
import sqlite3          # Built-in Python database
import hashlib          # Hashes passwords (SHA-256)
import os               # Reads environment variables, file paths
from datetime import datetime, timedelta  # Timestamps for PDF and history
import re                   # Regex for password validation
import smtplib              # Sending emails
from email.mime.text import MIMEText
import feedparser           # Scraping RSS feeds


# ═══════════════════════════════════════════════════════════════
#  APP SETUP — create the Flask app and configure it
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)

# secret_key is needed for Flask sessions (stores login state).
# os.urandom(24) generates a random 24-byte key each time the
# server starts, which keeps session cookies secure.
app.secret_key = os.urandom(24)

# File paths — where to find the database, model, and results
DB_FILE = "ventureverse.db"             # SQLite database file
MODEL_FILE = "ventureverse_model.joblib"  # Trained ML model
RESULTS_FILE = "model_results_summary.json"  # Model comparison metrics

# ── MAIL CONFIGURATION (Gmail) ────────────────────────────────
MAIL_EMAIL = "ventureverseltd@gmail.com"
MAIL_PASSWORD = "ymad pvyb wrfy ejix"

# ── NEWS CACHE ───────────────────────────────────────────────
NEWS_CACHE = None
LAST_FETCH_TIME = None
CACHE_DURATION = timedelta(minutes=30)

# ── GEMINI API KEY ───────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBadRdKX8Zi_bhfapi9ZwvcL4ZJtArnnOQ"

# ── ADMIN CREDENTIALS (Hardcoded) ─────────────────────────────
ADMIN_EMAIL = "admin@ventureverse.com"
ADMIN_PASSWORD = "admin123"


# ═══════════════════════════════════════════════════════════════
#  DATABASE SETUP — create the tables if they don't exist
# ═══════════════════════════════════════════════════════════════

def init_db():
    """
    Creates two database tables (if they don't already exist):
      1. 'users'       — stores registered accounts
      2. 'predictions' — stores every prediction a user makes

    This function runs once when the server starts.
    """
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()

    # Table 1: Users — stores name, email, and hashed password
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name     TEXT NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 2: Predictions — stores every prediction result
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER,
            prediction_score REAL,
            pred_label       TEXT,
            input_data       TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.commit()
    connection.close()


def hash_password(password):
    """
    Converts a plain-text password into a SHA-256 hash.

    Why hash?  We never store passwords in plain text — that would
    be a security risk. Instead, we store the hash. When a user
    logs in, we hash what they typed and compare it to the stored
    hash. If they match, the password is correct.

    Example:
        hash_password("hello123")
        → "f6e0a1e2910d1...4a8b1c3" (64-character hex string)
    """
    return hashlib.sha256(password.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
#  STARTUP — run when the server first starts
# ═══════════════════════════════════════════════════════════════

# Create the database tables (safe to call multiple times)
init_db()

# Load the trained ML model from disk so we can make predictions.
# This model was created by train_model.py — it's an XGBoost
# pipeline that includes preprocessing + classification.
model = joblib.load(MODEL_FILE)

# Load the model comparison results (accuracy, ROC-AUC, etc.)
# for displaying on the Charts page.
try:
    with open(RESULTS_FILE) as file:
        model_results = json.load(file)
except Exception:
    model_results = None


# ═══════════════════════════════════════════════════════════════
#  LOOKUP MAPS — readable names for states and industries
# ═══════════════════════════════════════════════════════════════

# The dataset uses short codes like "CA" and "biotech".
# These dictionaries convert them into full, readable names
# for display on the website.

ECOSYSTEMMAP = {
    'major_hub': '🌍 Major Hub (Silicon Valley / London / NYC / Berlin)',
    'secondary_hub': '🏙️ Secondary Hub (Austin / Paris / Singapore / Dubai)',
    'emerging': '🌱 Emerging Market (Other regions worldwide)'
}
ECOSYSTEMS = list(ECOSYSTEMMAP.keys())  # ["major_hub", "secondary_hub", "emerging"]

INDUSTRY_MAP = {
    "biotech": "Biotech",
    "consulting": "Consulting",
    "ecommerce": "E-Commerce",
    "enterprise": "Enterprise",
    "games_video": "Games & Video",
    "mobile": "Mobile",
    "software": "Software",
    "web": "Web",
    "advertising": "Advertising",
    "other": "Other",
}
CATEGORIES = list(INDUSTRY_MAP.keys())  # ["biotech", "consulting", ...]


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTION: Build Input DataFrame
# ═══════════════════════════════════════════════════════════════

def build_input_df(form):
    """
    Takes the raw form data (strings from the HTML form) and
    builds a pandas DataFrame that the ML model can understand.

    This function also engineers three extra features:
      - funding_duration       = last_funding - first_funding
      - avg_funding_per_round  = total_funding / number_of_rounds
      - log_funding            = log(1 + total_funding)

    These match exactly what train_model.py creates during
    training, so the model receives the same feature set.

    Args:
        form: dictionary of form field names → string values

    Returns:
        A single-row pandas DataFrame ready for model.predict_proba()
    """

    # --- Read numeric fields from the form ---
    funding_total = float(form.get("funding_total_usd", 0))
    funding_rounds = int(form.get("funding_rounds", 1))
    relationships = int(form.get("relationships", 0))
    milestones = int(form.get("milestones", 0))
    avg_participants = float(form.get("avg_participants", 1.0))

    age_first_funding = float(form.get("age_first_funding_year", 0))
    age_last_funding = float(form.get("age_last_funding_year", 0))

    # Milestone ages can be 0 if the startup has none yet.
    # We treat 0 as "missing" (NaN) so the model handles it properly.
    age_first_milestone = float(form.get("age_first_milestone_year", 0)) or np.nan
    age_last_milestone = float(form.get("age_last_milestone_year", 0)) or np.nan

    # --- Read binary (yes/no) checkboxes ---
    has_vc = int(form.get("has_VC", 0))
    has_angel = int(form.get("has_angel", 0))
    has_round_a = int(form.get("has_roundA", 0))
    has_round_b = int(form.get("has_roundB", 0))
    has_round_c = int(form.get("has_roundC", 0))
    has_round_d = int(form.get("has_roundD", 0))
    is_top_500 = int(form.get("is_top500", 0))

    # --- Read category fields ---
    category = form.get("category_code", "other")
    ecosystem = form.get("ecosystem", "other")

    # Internal mapping for model compatibility
    ecosystem_to_state = {
        'major_hub': 'CA',
        'secondary_hub': 'NY',
        'emerging': 'other'
    }
    state = ecosystem_to_state.get(ecosystem, 'other')

    # --- Engineer derived features (must match train_model.py) ---
    # How long between first and last funding round (in years)
    funding_duration = max(age_last_funding - age_first_funding, 0)

    # Avoid dividing by zero if funding_rounds is 0
    safe_rounds = funding_rounds if funding_rounds > 0 else 1

    # Average amount raised per funding round
    avg_funding_per_round = funding_total / safe_rounds

    # Log-transform reduces the huge range of funding values
    # (e.g. $10K vs $100M) into a smaller, more manageable scale.
    log_funding = np.log1p(funding_total)

    # --- Build a single-row DataFrame with all features ---
    input_row = {
        "age_first_funding_year": age_first_funding,
        "age_last_funding_year": age_last_funding,
        "age_first_milestone_year": age_first_milestone,
        "age_last_milestone_year": age_last_milestone,
        "relationships": relationships,
        "funding_rounds": funding_rounds,
        "funding_total_usd": funding_total,
        "milestones": milestones,
        "avg_participants": avg_participants,
        "funding_duration": funding_duration,
        "avg_funding_per_round": avg_funding_per_round,
        "log_funding": log_funding,
        "has_VC": has_vc,
        "has_angel": has_angel,
        "has_roundA": has_round_a,
        "has_roundB": has_round_b,
        "has_roundC": has_round_c,
        "has_roundD": has_round_d,
        "is_top500": is_top_500,
        "category_code": category,
        "state_code": state,
    }

    return pd.DataFrame([input_row])


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTION: Compute Risk Breakdown
# ═══════════════════════════════════════════════════════════════

def compute_risk_breakdown(form):
    """
    Analyses 6 startup factors and assigns each a score (0–100)
    and a status label (strong / moderate / weak).

    These 6 factors are shown as coloured bars on the prediction
    page and in the charts:
      1. Total Funding
      2. Funding Rounds
      3. Investor Quality
      4. Network Strength
      5. Early Traction (milestones)
      6. Location

    The thresholds are based on patterns observed in the
    Crunchbase training data.

    Args:
        form: dictionary of form field names → string values

    Returns:
        list of 6 dicts, each with keys: factor, score, status
    """
    factors = []

    # ── Factor 1: Total Funding ──────────────────────────────
    funding = float(form.get("funding_total_usd", 0))
    if funding >= 10_000_000:  # $10M+
        factors.append({"factor": "Total Funding", "score": 90, "status": "strong"})
    elif funding >= 2_000_000:  # $2M–$10M
        factors.append({"factor": "Total Funding", "score": 65, "status": "moderate"})
    elif funding >= 500_000:  # $500K–$2M
        factors.append({"factor": "Total Funding", "score": 40, "status": "moderate"})
    else:  # Under $500K
        factors.append({"factor": "Total Funding", "score": 15, "status": "weak"})

    # ── Factor 2: Funding Rounds ─────────────────────────────
    rounds = int(form.get("funding_rounds", 0))
    if rounds >= 4:
        factors.append({"factor": "Funding Rounds", "score": 85, "status": "strong"})
    elif rounds >= 2:
        factors.append({"factor": "Funding Rounds", "score": 55, "status": "moderate"})
    else:
        factors.append({"factor": "Funding Rounds", "score": 20, "status": "weak"})

    # ── Factor 3: Investor Quality ───────────────────────────
    has_vc = int(form.get("has_VC", 0))
    is_top = int(form.get("is_top500", 0))
    has_angel = int(form.get("has_angel", 0))

    if is_top:  # Backed by a Top-500 VC firm
        factors.append({"factor": "Investor Quality", "score": 95, "status": "strong"})
    elif has_vc:  # Has venture capital
        factors.append({"factor": "Investor Quality", "score": 70, "status": "strong"})
    elif has_angel:  # Has angel investors only
        factors.append({"factor": "Investor Quality", "score": 45, "status": "moderate"})
    else:  # No institutional backing
        factors.append({"factor": "Investor Quality", "score": 10, "status": "weak"})

    # ── Factor 4: Network Strength ───────────────────────────
    relationships = int(form.get("relationships", 0))
    if relationships >= 10:
        factors.append({"factor": "Network Strength", "score": 85, "status": "strong"})
    elif relationships >= 4:
        factors.append({"factor": "Network Strength", "score": 55, "status": "moderate"})
    else:
        factors.append({"factor": "Network Strength", "score": 20, "status": "weak"})

    # ── Factor 5: Early Traction (milestones) ────────────────
    milestones = int(form.get("milestones", 0))
    if milestones >= 3:
        factors.append({"factor": "Early Traction", "score": 80, "status": "strong"})
    elif milestones >= 1:
        factors.append({"factor": "Early Traction", "score": 50, "status": "moderate"})
    else:
        factors.append({"factor": "Early Traction", "score": 10, "status": "weak"})

    # ── Factor 6: Ecosystem ───────────────────────────────────
    ecosystem = form.get("ecosystem", "emerging")
    if ecosystem == "major_hub":
        factors.append({"factor": "Ecosystem", "score": 85, "status": "strong"})
    elif ecosystem == "secondary_hub":
        factors.append({"factor": "Ecosystem", "score": 65, "status": "moderate"})
    else:
        factors.append({"factor": "Ecosystem", "score": 35, "status": "moderate"})

    return factors


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTION: Generate Insight Cards
# ═══════════════════════════════════════════════════════════════

def generate_insights(form, prediction, pred_label, risk_factors):
    """
    Creates a list of insight cards based on the prediction result
    and the startup's profile. Each card has:
      - title  : short heading (e.g. "Strong Success Indicators")
      - icon   : emoji HTML code
      - type   : "positive", "negative", "neutral", or "action"
      - text   : detailed explanation paragraph

    These cards are displayed on the Insights page and included
    in the downloadable PDF report.

    Args:
        form:         dictionary of form data
        prediction:   success probability percentage (e.g. 75.3)
        pred_label:   "Success" or "Failure"
        risk_factors: list from compute_risk_breakdown()

    Returns:
        list of insight card dictionaries
    """
    insights = []

    # ── Card 1: Overall assessment based on probability ──────
    if prediction >= 75:
        insights.append({
            "title": "Strong Success Indicators",
            "icon": "&#9733;",  # ★ star
            "type": "positive",
            "text": (
                f"This startup profile shows a {prediction}% success "
                f"probability, significantly above the 50% threshold. "
                f"The combination of factors suggests strong market readiness."
            ),
        })
    elif prediction >= 50:
        insights.append({
            "title": "Moderate Success Potential",
            "icon": "&#9888;",  # ⚠ warning
            "type": "neutral",
            "text": (
                f"At {prediction}%, this startup is above the success "
                f"threshold but has room for improvement. Strengthening "
                f"weaker factors could push probability higher."
            ),
        })
    else:
        insights.append({
            "title": "High Risk Profile",
            "icon": "&#9888;",
            "type": "negative",
            "text": (
                f"With {prediction}% success probability, this profile "
                f"indicates elevated failure risk. Key areas need "
                f"immediate attention."
            ),
        })

    # ── Card 2: Funding analysis ─────────────────────────────
    funding = float(form.get("funding_total_usd", 0))
    rounds = int(form.get("funding_rounds", 0))

    if funding >= 5_000_000 and rounds >= 3:
        insights.append({
            "title": "Funding Strength",
            "icon": "&#128176;",  # 💰
            "type": "positive",
            "text": (
                f"Total funding of ${funding:,.0f} across {rounds} rounds "
                f"demonstrates strong investor confidence and repeated "
                f"due diligence validation."
            ),
        })
    elif funding < 500_000:
        insights.append({
            "title": "Funding Gap",
            "icon": "&#128176;",
            "type": "negative",
            "text": (
                f"Total funding of ${funding:,.0f} is below the typical "
                f"threshold. Startups with under $500K historically have "
                f"significantly higher failure rates."
            ),
        })
    else:
        insights.append({
            "title": "Moderate Funding",
            "icon": "&#128176;",
            "type": "neutral",
            "text": (
                f"Total funding of ${funding:,.0f} across {rounds} round(s) "
                f"is moderate. Consider pursuing additional rounds to "
                f"strengthen the profile."
            ),
        })

    # ── Card 3: Investor backing ─────────────────────────────
    has_vc = int(form.get("has_VC", 0))
    is_top = int(form.get("is_top500", 0))
    has_angel = int(form.get("has_angel", 0))

    if is_top:
        insights.append({
            "title": "Elite VC Backing",
            "icon": "&#127942;",  # 🏆
            "type": "positive",
            "text": (
                "Backed by a Top-500 VC firm — one of the strongest "
                "success signals. Elite VCs provide capital, strategic "
                "guidance, and credibility."
            ),
        })
    elif has_vc and has_angel:
        insights.append({
            "title": "Diversified Investors",
            "icon": "&#127942;",
            "type": "positive",
            "text": (
                "Both VC and angel investors provide a balanced funding "
                "structure with early validation and institutional support."
            ),
        })
    elif not has_vc and not has_angel:
        insights.append({
            "title": "No Institutional Backing",
            "icon": "&#127942;",
            "type": "negative",
            "text": (
                "No VC or angel backing detected. Startups without "
                "institutional investors face lower acquisition rates."
            ),
        })

    # ── Card 4: Network / relationships ──────────────────────
    relationships = int(form.get("relationships", 0))
    if relationships >= 8:
        insights.append({
            "title": "Strong Network",
            "icon": "&#128101;",  # 👥
            "type": "positive",
            "text": (
                f"{relationships} key connections — a robust network of "
                f"advisors and co-founders correlates with better "
                f"mentorship and deal flow."
            ),
        })
    elif relationships <= 2:
        insights.append({
            "title": "Limited Network",
            "icon": "&#128101;",
            "type": "negative",
            "text": (
                f"Only {relationships} connection(s). Expanding the "
                f"advisory board could significantly improve outcomes."
            ),
        })

    # ── Card 5: Milestones ───────────────────────────────────
    milestones = int(form.get("milestones", 0))
    if milestones == 0:
        insights.append({
            "title": "No Milestones",
            "icon": "&#127937;",  # 🏁
            "type": "negative",
            "text": (
                "No milestones recorded. Product demos, partnerships, "
                "or press coverage are important signals investors look for."
            ),
        })
    elif milestones >= 3:
        insights.append({
            "title": "Strong Traction",
            "icon": "&#127937;",
            "type": "positive",
            "text": (
                f"{milestones} milestones demonstrate execution "
                f"capability and market validation."
            ),
        })

    # ── Card 6: Ecosystem advantage ───────────────────────────
    ecosystem = form.get("ecosystem", "emerging")

    if ecosystem == "major_hub":
        insights.append({
            "title": "Major Hub Advantage",
            "icon": "&#128205;",  # 📍
            "type": "positive",
            "text": (
                "Based in a top global startup hub with strong "
                "investor density and talent access."
            ),
        })
    elif ecosystem == "secondary_hub":
        insights.append({
            "title": "Strong Ecosystem",
            "icon": "&#128205;",
            "type": "neutral",
            "text": (
                "Located in a growing secondary hub with solid "
                "support networks."
            ),
        })

    # ── Card 7: Recommendations (based on weak factors) ──────
    weak_factors = [rf for rf in risk_factors if rf["status"] == "weak"]
    if weak_factors:
        weak_names = ", ".join([w["factor"] for w in weak_factors])
        insights.append({
            "title": "Key Recommendations",
            "icon": "&#128161;",  # 💡
            "type": "action",
            "text": (
                f"Priority areas: {weak_names}. Focus on the "
                f"lowest-scoring factor first for maximum impact "
                f"on success probability."
            ),
        })

    return insights


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTION: Get Model Comparison Data
# ═══════════════════════════════════════════════════════════════

def get_model_comparison():
    """
    Reads the model_results_summary.json file and extracts
    the performance metrics for all three ML models
    (Logistic Regression, Random Forest, XGBoost).

    Returns a dict with lists of names, ROC-AUC scores,
    accuracies, and F1 scores — used by the Charts page.
    Returns None if no results file is available.
    """
    if not model_results or "all_model_results" not in model_results:
        return None

    names = []
    roc_scores = []
    accuracies = []
    f1_scores = []

    for model_info in model_results["all_model_results"]:
        names.append(model_info["name"])
        roc_scores.append(round(model_info.get("cv_roc_auc_mean", 0) * 100, 1))
        accuracies.append(round(model_info.get("cv_accuracy_mean", 0) * 100, 1))
        f1_scores.append(round(model_info.get("cv_f1_mean", 0) * 100, 1))

    return {
        "names": names,
        "roc_aucs": roc_scores,
        "accuracies": accuracies,
        "f1s": f1_scores,
        "winner": model_results.get("winner", ""),
    }


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTION: Get Prediction History
# ═══════════════════════════════════════════════════════════════

def get_prediction_history(user_id):
    """
    Fetches the last 10 predictions made by a specific user
    from the database, ordered newest-first.

    Returns a list of dicts, each containing:
      - score : prediction percentage
      - label : "Success" or "Failure"
      - data  : the original form data (as a dict)
      - date  : formatted date string (e.g. "19 Apr 2026, 03:15 PM")
    """
    try:
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()

        cursor.execute(
            "SELECT prediction_score, pred_label, input_data, created_at "
            "FROM predictions WHERE user_id=? "
            "ORDER BY created_at DESC LIMIT 10",
            (user_id,),
        )

        rows = cursor.fetchall()
        connection.close()

        history = []
        for row in rows:
            raw_date = row[3] or ""

            # Try to format the date nicely; fall back to raw string
            try:
                date_obj = datetime.strptime(raw_date[:19], "%Y-%m-%d %H:%M:%S")
                display_date = date_obj.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                display_date = raw_date[:16] if raw_date else "-"

            history.append({
                "score": row[0],
                "label": row[1],
                "data": json.loads(row[2]) if row[2] else {},
                "date": display_date,
            })

        return history

    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTION: Send Welcome Email
# ═══════════════════════════════════════════════════════════════

def send_welcome_email(to_email, user_name):
    """
    Sends a welcome email via Gmail SMTP.
    Fails silently if configuration is missing or connection fails.
    """
    if MAIL_EMAIL == "your@gmail.com":
        return

    try:
        msg = MIMEText(
            f"Hi {user_name.split(' ')[0]},\n\n"
            f"Welcome to VentureVerse! 🚀\n\n"
            f"We're excited to have you on board. VentureVerse helps you predict "
            f"startup success using advanced machine learning models.\n\n"
            f"You can now log in and start predicting at: https://venture-verse-s9ch.onrender.com\n\n"
            f"Best regards,\n"
            f"The VentureVerse Team"
        )
        msg['Subject'] = "Welcome to VentureVerse! 🚀"
        msg['From'] = MAIL_EMAIL
        msg['To'] = to_email

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MAIL_EMAIL, MAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Success: Welcome email sent to {to_email}")
    except Exception as e:
        print(f"❌ Mail Error: {str(e)}")
        pass  # Skip silently for the user, but log for the developer


# ═══════════════════════════════════════════════════════════════
#  ROUTES: Authentication (Login, Signup, Logout)
# ═══════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    """
    GET  → shows the login form
    POST → checks email + password against the database

    If credentials are correct:
      - stores user info in the session (like a browser cookie)
      - redirects to the home/predict page

    If credentials are wrong:
      - re-renders the login page with an error message
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        # --- CASE 1: Check Admin Credentials ---
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session.clear()  # Clear any existing session
            session["user_id"] = "admin"
            session["user_name"] = "Administrator"
            session["user_email"] = email
            session["is_admin"] = True
            return redirect(url_for("admin"))

        # --- CASE 2: Check standard user in Database ---
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, full_name, password_hash FROM users WHERE email=?",
            (email,),
        )
        user = cursor.fetchone()
        connection.close()

        # Check if user exists AND the password hash matches
        if user and user[2] == hash_password(password):
            # Store login info in the session
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            session["user_email"] = email
            session["is_admin"] = False
            return redirect(url_for("home"))

        # If we get here, login failed
        return render_template("login.html", error="Invalid email or password.")

    # GET request — just show the login form
    return render_template("login.html", error=None)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """
    GET  → shows the signup form
    POST → creates a new user account in the database

    Validates:
      - All fields are filled in
      - Passwords match
      - Password is at least 6 characters
      - Email is not already registered
    """
    if request.method == "POST":
        name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        # Validation checks
        if not name or not email or not password:
            return render_template("signup.html", error="All fields required.")

        if password != confirm:
            return render_template("signup.html", error="Passwords do not match.")

        # Backend Password Validation (Live JS also checks this)
        # 8+ chars, 1 uppercase, 1 number, 1 special
        pw_regex = r"^(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*]).{8,}$"
        if not re.match(pw_regex, password):
            return render_template("signup.html", error="Password does not meet security requirements.")

        # Try to insert the new user into the database
        try:
            connection = sqlite3.connect(DB_FILE)
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO users (full_name, email, password_hash) VALUES (?,?,?)",
                (name, email, hash_password(password)),
            )
            connection.commit()
            new_user_id = cursor.lastrowid
            connection.close()

            # Auto-login after signup
            session["user_id"] = new_user_id
            session["user_name"] = name
            session["user_email"] = email

            # Send welcome email (async-like / non-blocking skip on fail)
            send_welcome_email(email, name)

            return redirect(url_for("home"))

        except sqlite3.IntegrityError:
            connection.close()
            # This error means the email already exists (UNIQUE constraint)
            return render_template("signup.html", error="Email already registered.")
        except Exception as e:
            if 'connection' in locals():
                connection.close()
            return render_template("signup.html", error=f"Database error: {str(e)}")

    # GET request — just show the signup form
    return render_template("signup.html", error=None)


@app.route("/logout")
def logout():
    """
    Clears the session (removes login info) and redirects
    the user back to the home page.
    """
    session.clear()
    return redirect(url_for("home"))


# ═══════════════════════════════════════════════════════════════
#  ROUTES: Landing Page and Prediction
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """
    The main page of the website.

    If the user is NOT logged in → show the landing page
    If the user IS logged in     → show the prediction form

    Also passes any previous prediction results stored in
    the session, so users can see their last result.
    """
    if "user_id" not in session:
        return render_template("landing.html")

    return render_template(
        "index.html",
        prediction=session.get("last_prediction"),
        pred_label=session.get("last_pred_label"),
        error=None,
        categories=CATEGORIES,
        ecosystems=ECOSYSTEMS,
        ecosystemmap=ECOSYSTEMMAP,
        industry_map=INDUSTRY_MAP,
        form_data=session.get("last_form_data", {}),
        risk_factors=session.get("last_risk_factors"),
        user_name=session.get("user_name"),
        active_page="predict",
    )


@app.route("/reset")
def reset():
    """
    Clears the last prediction from the session so the user
    gets a fresh, empty form when they return to the home page.
    """
    session.pop("last_prediction", None)
    session.pop("last_pred_label", None)
    session.pop("last_form_data", None)
    session.pop("last_risk_factors", None)
    return redirect(url_for("home"))


@app.route("/predict", methods=["POST"])
def predict():
    """
    The core route — takes form data, runs the ML model, and
    returns the prediction result.

    Steps:
      1. Check the user is logged in
      2. Read all form fields
      3. Validate that required fields are filled
      4. Build a DataFrame and feed it to the model
      5. Get the success probability
      6. Compute risk factor breakdown
      7. Save the prediction to the database
      8. Display the result on the page
    """
    # Step 1: Must be logged in to predict
    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        # Step 2: Read all form fields and clean whitespace.
        # Default empty numeric fields to "0" to prevent float conversion errors.
        numeric_fields = [
            "funding_total_usd", "funding_rounds", "relationships", "milestones",
            "avg_participants", "age_first_funding_year", "age_last_funding_year",
            "age_first_milestone_year", "age_last_milestone_year"
        ]
        
        form_data = {}
        for key in request.form:
            val = request.form.get(key, "").strip()
            if key in numeric_fields and not val:
                val = "0"
            form_data[key] = val

        # Step 3: Check required fields (must not be "0" if they are critical)
        # Note: funding_total_usd and funding_rounds are now at least "0"
        if form_data.get("funding_total_usd") == "0" or form_data.get("funding_rounds") == "0":
            return render_template(
                "index.html",
                prediction=None,
                pred_label=None,
                error="Funding Total and Funding Rounds are required.",
                categories=CATEGORIES,
                ecosystems=ECOSYSTEMS,
                ecosystemmap=ECOSYSTEMMAP,
                industry_map=INDUSTRY_MAP,
                form_data=form_data,
                risk_factors=None,
                user_name=session.get("user_name"),
                active_page="predict",
            )

        # Step 4: Build the input DataFrame
        input_df = build_input_df(form_data)

        # Step 5: Get the prediction from the ML model
        # predict_proba returns [[prob_failure, prob_success]]
        # We want index [0][1] = the success probability
        probability = model.predict_proba(input_df)[0][1]
        prediction_pct = round(probability * 100, 2)  # e.g. 0.754 → 75.40
        pred_label = "Success" if probability >= 0.5 else "At Risk"

        # Step 6: Compute the risk factor breakdown
        risk_factors = compute_risk_breakdown(form_data)

        # Store results in the session for other pages to use
        session["last_prediction"] = prediction_pct
        session["last_pred_label"] = pred_label
        session["last_form_data"] = form_data
        session["last_risk_factors"] = risk_factors

        # Step 7: Save the prediction to the database
        try:
            connection = sqlite3.connect(DB_FILE)
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO predictions "
                "(user_id, prediction_score, pred_label, input_data) "
                "VALUES (?,?,?,?)",
                (session["user_id"], prediction_pct, pred_label, json.dumps(form_data)),
            )
            connection.commit()
            connection.close()
        except Exception:
            if 'connection' in locals():
                connection.close()
            pass  # Don't crash if the database save fails

        # Step 8: Show the result
        return render_template(
            "index.html",
            prediction=prediction_pct,
            pred_label=pred_label,
            error=None,
            categories=CATEGORIES,
            ecosystems=ECOSYSTEMS,
            ecosystemmap=ECOSYSTEMMAP,
            industry_map=INDUSTRY_MAP,
            form_data=form_data,
            risk_factors=risk_factors,
            user_name=session.get("user_name"),
            active_page="predict",
        )

    except Exception as error:
        # If anything goes wrong, show the error on the page
        form_data = {key: request.form.get(key, "") for key in request.form}
        return render_template(
            "index.html",
            prediction=None,
            pred_label=None,
            error=str(error),
            categories=CATEGORIES,
            ecosystems=ECOSYSTEMS,
            ecosystemmap=ECOSYSTEMMAP,
            industry_map=INDUSTRY_MAP,
            form_data=form_data,
            risk_factors=None,
            user_name=session.get("user_name"),
            active_page="predict",
        )


# ═══════════════════════════════════════════════════════════════
#  ROUTES: Charts Page
# ═══════════════════════════════════════════════════════════════

@app.route("/charts")
def charts():
    """
    Shows interactive charts (polar area, radar, gauge) that
    visualise the prediction results using Chart.js.

    Requires login. If no prediction has been made yet,
    the page shows a "no data" message.
    """
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "charts.html",
        prediction=session.get("last_prediction"),
        pred_label=session.get("last_pred_label"),
        risk_factors=session.get("last_risk_factors"),
        model_comparison=get_model_comparison(),
        user_name=session.get("user_name"),
        active_page="charts",
    )


# ═══════════════════════════════════════════════════════════════
#  ROUTES: Insights Page
# ═══════════════════════════════════════════════════════════════

@app.route("/insights")
def insights():
    """
    Shows detailed insight cards and prediction history.

    The insight cards explain what each factor means for the
    startup's chances. The history table shows the user's
    last 10 predictions.
    """
    if "user_id" not in session:
        return redirect(url_for("login"))

    prediction = session.get("last_prediction")
    pred_label = session.get("last_pred_label")
    form_data = session.get("last_form_data", {})
    risk_factors = session.get("last_risk_factors", [])

    # Generate insight cards only if a prediction exists
    if prediction:
        insight_cards = generate_insights(form_data, prediction, pred_label, risk_factors)
    else:
        insight_cards = []

    return render_template(
        "insights.html",
        prediction=prediction,
        pred_label=pred_label,
        insights=insight_cards,
        history=get_prediction_history(session["user_id"]),
        ecosystemmap=ECOSYSTEMMAP,
        industry_map=INDUSTRY_MAP,
        user_name=session.get("user_name"),
        active_page="insights",
    )


# ═══════════════════════════════════════════════════════════════
#  ROUTES: Download PDF Report
# ═══════════════════════════════════════════════════════════════

@app.route("/download-insights")
def download_insights():
    """
    Generates a PDF report of the prediction insights and
    sends it to the user's browser as a download.

    Uses the ReportLab library for PDF generation.
    If ReportLab is not installed, falls back to a plain
    text (.txt) file instead.
    """
    if "user_id" not in session:
        return redirect(url_for("login"))

    prediction = session.get("last_prediction")
    pred_label = session.get("last_pred_label")
    form_data = session.get("last_form_data", {})
    risk_factors = session.get("last_risk_factors", [])

    if not prediction:
        return redirect(url_for("insights"))

    # Generate the insight cards to include in the PDF
    cards = generate_insights(form_data, prediction, pred_label, risk_factors)

    try:
        # ── Try to create a proper PDF using ReportLab ───────
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm

        filepath = f"/tmp/ventureverse_insights_{session['user_id']}.pdf"
        doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=25 * mm, bottomMargin=20 * mm)

        # Set up text styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title2", parent=styles["Title"],
            fontSize=22, textColor=colors.HexColor("#1a1a2e"),
        )
        heading_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=14, textColor=colors.HexColor("#1a1a2e"),
        )
        body_style = ParagraphStyle(
            "Body2", parent=styles["Normal"],
            fontSize=11, leading=16,
        )

        # Build the PDF content (called a "story" in ReportLab)
        story = []

        # Title
        story.append(Paragraph("VentureVerse — Prediction Insights Report", title_style))
        story.append(Spacer(1, 8 * mm))

        # Prediction result
        story.append(Paragraph(
            f"Prediction: <b>{prediction}%</b> — <b>{pred_label}</b>",
            heading_style,
        ))
        story.append(Spacer(1, 4 * mm))

        # Date generated
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            body_style,
        ))
        story.append(Spacer(1, 8 * mm))

        # Insight cards
        for card in cards:
            badge = {
                "positive": "Strength",
                "negative": "Risk",
                "action": "Action",
            }.get(card["type"], "Note")

            story.append(Paragraph(
                f"<b>{card['title']}</b>  [{badge}]",
                heading_style,
            ))
            story.append(Paragraph(card["text"], body_style))
            story.append(Spacer(1, 4 * mm))

        # Risk factor table
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Factor Strength Breakdown", heading_style))

        if risk_factors:
            table_data = [["Factor", "Score", "Status"]]
            for rf in risk_factors:
                table_data.append([
                    rf["factor"],
                    f"{rf['score']}%",
                    rf["status"].capitalize(),
                ])

            table = Table(table_data, colWidths=[55 * mm, 25 * mm, 30 * mm])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f8f8f8"), colors.white]),
            ]))
            story.append(table)

        # Footer
        story.append(Spacer(1, 10 * mm))
        story.append(Paragraph(
            "University of Westminster — BSc Computer Science Final Project",
            body_style,
        ))

        doc.build(story)
        return send_file(filepath, as_attachment=True, download_name="VentureVerse_Insights.pdf")

    except ImportError:
        # ── Fallback: plain text file if ReportLab not installed ──
        filepath = f"/tmp/ventureverse_insights_{session['user_id']}.txt"
        with open(filepath, "w") as file:
            file.write(f"VentureVerse Insights Report\n")
            file.write(f"Prediction: {prediction}% — {pred_label}\n\n")
            for card in cards:
                file.write(f"[{card['type'].upper()}] {card['title']}\n")
                file.write(f"{card['text']}\n\n")
        return send_file(filepath, as_attachment=True, download_name="VentureVerse_Insights.txt")


# ═══════════════════════════════════════════════════════════════
#  ROUTES: About Page
# ═══════════════════════════════════════════════════════════════

@app.route("/about")
def about():
    """Shows the About page with project info and tech stack."""
    return render_template(
        "about.html",
        user_name=session.get("user_name"),
        active_page="about",
    )


# ═══════════════════════════════════════════════════════════════
#  ROUTES: News Feed (RSS Scraper)
# ═══════════════════════════════════════════════════════════════

@app.route("/news")
def news():
    """
    Fetches startup news from TechCrunch and Forbes RSS feeds.
    Caches the results for 30 minutes.
    """
    global NEWS_CACHE, LAST_FETCH_TIME

    now = datetime.now()
    if NEWS_CACHE and LAST_FETCH_TIME and (now - LAST_FETCH_TIME < CACHE_DURATION):
        articles = NEWS_CACHE
    else:
        # Fetch fresh news
        feeds = [
            {"name": "TechCrunch", "url": "https://techcrunch.com/category/startups/feed/"},
            {"name": "Forbes", "url": "https://www.forbes.com/entrepreneurs/feed/"}
        ]

        articles = []
        for f in feeds:
            feed_data = feedparser.parse(f["url"])
            # Get latest 6 from each
            for entry in feed_data.entries[:6]:
                # Format date
                date_str = "-"
                if 'published' in entry:
                    date_str = entry.published[:16]

                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": re.sub('<[^<]+?>', '', entry.summary[:200]) + "...",
                    "published": date_str,
                    "source": f["name"]
                })

        # Update cache
        NEWS_CACHE = articles
        LAST_FETCH_TIME = now

    return render_template(
        "news.html",
        articles=articles,
        user_name=session.get("user_name"),
        active_page="news"
    )


# ═══════════════════════════════════════════════════════════════
#  ROUTES: AI Chatbot (Google Gemini)
# ═══════════════════════════════════════════════════════════════

# Using the GEMINI_API_KEY defined at the top of the file.


@app.route("/chatbot")
def chatbot():
    """
    Shows the chatbot page. Initialises an empty chat history
    in the session if this is the user's first visit.
    """
    if "user_id" not in session:
        return redirect(url_for("login"))

    if not session.get("chat_history"):
        session["chat_history"] = []

    return render_template(
        "chatbot.html",
        chat_history=session.get("chat_history", []),
        user_name=session.get("user_name"),
        active_page="chatbot",
        api_configured=bool(GEMINI_API_KEY),
    )


@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives a chat message from the user (as JSON), sends it
    to Google's Gemini AI along with context about the user's
    latest prediction, and returns the AI's response.

    The conversation history is stored in the session so the
    chatbot remembers previous messages in the same session.
    """
    if "user_id" not in session:
        return {"error": "Not logged in"}, 401

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return {"error": "Empty message"}, 400

    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured. Set GEMINI_API_KEY environment variable."}, 500

    # ── Build context about the user's prediction ────────────
    prediction = session.get("last_prediction")
    pred_label = session.get("last_pred_label")
    form_data = session.get("last_form_data", {})
    risk_factors = session.get("last_risk_factors", [])

    # System prompt tells Gemini how to behave
    context_parts = [
        "You are VentureBot, an AI startup advisor integrated into VentureVerse — a startup success prediction platform.",
        "You provide actionable, concise startup advice. Keep responses under 200 words unless the user asks for detail.",
        "Be encouraging but honest. Use specific data points when available.",
    ]

    # If the user has a prediction, include it as context
    if prediction is not None:
        funding = form_data.get("funding_total_usd", "N/A")
        industry = INDUSTRY_MAP.get(form_data.get("category_code", ""), "N/A")
        ecosystem = ECOSYSTEMMAP.get(form_data.get("ecosystem", ""), "N/A")
        rounds = form_data.get("funding_rounds", "N/A")

        context_parts.append(f"\nThe user just ran a prediction with these results:")
        context_parts.append(f"- Success Probability: {prediction}% ({pred_label})")
        context_parts.append(f"- Total Funding: ${float(funding):,.0f}" if funding != "N/A" else "")
        context_parts.append(f"- Funding Rounds: {rounds}")
        context_parts.append(f"- Industry: {industry}")
        context_parts.append(f"- Ecosystem: {ecosystem}")

        if risk_factors:
            weak = [rf["factor"] for rf in risk_factors if rf["status"] == "weak"]
            strong = [rf["factor"] for rf in risk_factors if rf["status"] == "strong"]
            if weak:
                context_parts.append(f"- Weak factors: {', '.join(weak)}")
            if strong:
                context_parts.append(f"- Strong factors: {', '.join(strong)}")

    system_prompt = "\n".join(context_parts)

    # ── Build conversation history for Gemini ────────────────
    chat_history = session.get("chat_history", [])

    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Include the last 10 messages for context
        contents = []
        for msg in chat_history[-10:]:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["text"]}],
            })
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        # Call the Gemini API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.7,       # Slightly creative
                "max_output_tokens": 500,  # Keep replies concise
            },
        )

        bot_reply = response.text

        # Save messages to session history
        chat_history.append({"role": "user", "text": user_message})
        chat_history.append({"role": "bot", "text": bot_reply})
        session["chat_history"] = chat_history[-20:]  # Keep last 20 messages

        return {"reply": bot_reply}

    except ImportError:
        return {"error": "google-genai package not installed. Run: pip install google-genai"}, 500
    except Exception as error:
        error_msg = str(error)
        # Check if this is a quota/rate limit error (status 429)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            bot_reply = "VentureBot is currently resting (API quota reached). Please try again in a few minutes or check your connection."
            return {"reply": bot_reply}
        
        return {"error": error_msg}, 500


@app.route("/chat/clear", methods=["POST"])
def clear_chat():
    """Clears the chatbot conversation history."""
    session["chat_history"] = []
    return {"status": "cleared"}


# ═══════════════════════════════════════════════════════════════
#  ROUTES: Admin Dashboard
# ═══════════════════════════════════════════════════════════════

@app.route("/admin")
def admin():
    """
    Shows high-level system statistics for administrators.
    Requires session['is_admin'] to be True.
    """
    if not session.get("is_admin"):
        return redirect(url_for("login"))

    try:
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()

        # Stats 1: Total Users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]

        # Stats 2: Total Predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        # Stats 3: Success vs At Risk counts
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE pred_label='Success'")
        success_count = cursor.fetchone()[0]
        at_risk_count = total_predictions - success_count

        # Stats 4: Most Active User
        cursor.execute("""
            SELECT u.full_name, u.email, COUNT(p.id) as pred_count
            FROM users u
            JOIN predictions p ON u.id = p.user_id
            GROUP BY u.id
            ORDER BY pred_count DESC
            LIMIT 1
        """)
        active_user_row = cursor.fetchone()
        most_active = {
            "name": active_user_row[0],
            "email": active_user_row[1],
            "count": active_user_row[2]
        } if active_user_row else None

        connection.close()

        return render_template(
            "admin.html",
            total_users=total_users,
            total_predictions=total_predictions,
            success_count=success_count,
            at_risk_count=at_risk_count,
            most_active=most_active,
            user_name=session.get("user_name"),
            active_tab="dashboard"
        )
    except Exception as e:
        return f"Admin Error: {str(e)}", 500


@app.route("/admin/users")
def admin_users():
    """Shows a list of all registered users."""
    if not session.get("is_admin"):
        return redirect(url_for("login"))

    try:
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()
        cursor.execute("SELECT id, full_name, email, created_at FROM users ORDER BY created_at DESC")
        rows = cursor.fetchall()
        connection.close()

        users_list = []
        for r in rows:
            users_list.append({
                "id": r[0],
                "name": r[1],
                "email": r[2],
                "joined": r[3][:10] if r[3] else "-"
            })

        return render_template(
            "admin.html",
            users=users_list,
            user_name=session.get("user_name"),
            active_tab="users"
        )
    except Exception as e:
        return str(e), 500


@app.route("/admin/predictions")
def admin_predictions():
    """Shows a list of all system predictions."""
    if not session.get("is_admin"):
        return redirect(url_for("login"))

    try:
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()
        # Join with users to get emails
        cursor.execute("""
            SELECT u.email, p.prediction_score, p.pred_label, p.created_at
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
        """)
        rows = cursor.fetchall()
        connection.close()

        preds_list = []
        for r in rows:
            preds_list.append({
                "email": r[0],
                "score": r[1],
                "label": r[2],
                "date": r[3][:16] if r[3] else "-"
            })

        return render_template(
            "admin.html",
            predictions=preds_list,
            user_name=session.get("user_name"),
            active_tab="predictions"
        )
    except Exception as e:
        return str(e), 500


@app.route("/admin/delete-user/<int:user_id>")
def delete_user(user_id):
    """Deletes a user and all their predictions."""
    if not session.get("is_admin"):
        return redirect(url_for("login"))

    try:
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()

        # 1. Delete user's predictions first (integrity)
        cursor.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))

        # 2. Delete the user
        cursor.execute("DELETE FROM users WHERE id=?", (user_id,))

        connection.commit()
        connection.close()
        return redirect(url_for("admin_users"))
    except Exception as e:
        return str(e), 500


# ═══════════════════════════════════════════════════════════════
#  START THE SERVER
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # debug=True enables auto-reload when you save changes
    # and shows detailed error pages in the browser.
    # IMPORTANT: Set debug=False in production!
    app.run(debug=True)
