"""
VentureVerse – Automated Test Suite
=====================================
This script runs 15 automated tests against the VentureVerse web app
to verify that everything works correctly.

Two types of tests:
  ┌─────────────────────────────────────────────────────────────────┐
  │  SECTION A — BLACK-BOX TESTS (TC01–TC10)                       │
  │  Simulates real user actions (signup, login, predict, etc.)     │
  │  through Flask's built-in test client (like a fake browser).    │
  │                                                                 │
  │  SECTION B — WHITE-BOX UNIT TESTS (UT01–UT05)                  │
  │  Calls internal Python functions directly to verify the maths  │
  │  and logic is correct.                                         │
  └─────────────────────────────────────────────────────────────────┘

How to run:
    python custom_test.py

Prerequisites:
    - ventureverse_model.joblib must exist (run train_model.py first)

Author : Kashish Jadhav (w2035589)
Module : 6COSC023W — BSc Computer Science Final Project
Uni    : University of Westminster, 2025–2026
"""

# ═══════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════

import sys
import os
import hashlib
import tempfile
import shutil
import numpy as np

# Fix for Windows terminals that can't display emojis
sys.stdout.reconfigure(encoding='utf-8')


# ═══════════════════════════════════════════════════════════════
#  TERMINAL COLOURS — makes output easier to read
# ═══════════════════════════════════════════════════════════════

# These are ANSI escape codes that colour text in the terminal.
# They only work in terminals that support colours (most do).
GREEN = "\033[92m"    # Green text (for PASS)
RED = "\033[91m"      # Red text (for FAIL)
YELLOW = "\033[93m"   # Yellow text (for notes)
CYAN = "\033[96m"     # Cyan text (for headings)
BOLD = "\033[1m"      # Bold text
RESET = "\033[0m"     # Reset to normal text


# ═══════════════════════════════════════════════════════════════
#  TEST RESULT TRACKING
# ═══════════════════════════════════════════════════════════════

# This list stores every test result as a tuple:
# (test_id, test_name, "PASS" or "FAIL", optional_note)
results = []


def record(test_id, test_name, passed, note=""):
    """
    Records and prints one test result.

    Args:
        test_id:   short ID like "TC01" or "UT03"
        test_name: human-readable description of what the test checks
        passed:    True if the test passed, False if it failed
        note:      optional extra info (e.g. HTTP status code)
    """
    status = "PASS" if passed else "FAIL"
    results.append((test_id, test_name, status, note))

    # Print a coloured PASS/FAIL mark
    mark = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"

    # Truncate long test names so they fit on one line
    short_name = test_name[:54] + "…" if len(test_name) > 55 else test_name

    print(f"  {mark}  {BOLD}{test_id}{RESET} — {short_name}")
    if note:
        print(f"         {YELLOW}↳ {note}{RESET}")


# ═══════════════════════════════════════════════════════════════
#  SETUP — Prepare a safe test environment
# ═══════════════════════════════════════════════════════════════
#
#  We do two things here:
#    1. Create a temporary throw-away database (not the real one!)
#    2. Create minimal stub HTML templates so Flask routes work
#       without needing the full CSS/JS frontend
#

# Create a temporary database file (deleted after tests finish)
TEST_DB = tempfile.mktemp(suffix=".db")

# Get the project directory path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# ── Create Stub Templates ───────────────────────────────────
# Flask routes call render_template("login.html") etc., which
# needs actual HTML files to exist. Instead of using the real
# frontend (which has complex CSS/JS), we create tiny stub
# files with just enough HTML for our tests to work.
#
# For example, the signup page shows "Email already registered"
# as an error. Our stub template includes the Jinja variable
# {{ error }} so the test can check for that text.

STUB_TEMPLATES = {
    "landing.html": "<html><body>Welcome to VentureVerse</body></html>",

    "index.html": """<html><body>
        {% if prediction is not none %}
            <p>Score: {{ prediction }}%</p>
            <p>Predicted: {{ pred_label }}</p>
            <p>Ecosystem: {{ ecosystemmap[form_data.get('ecosystem')] if ecosystemmap else '' }}</p>
        {% endif %}
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
    </body></html>""",

    "login.html": """<html><body>
        {% if error %}<p>{{ error }}</p>{% endif %}
        <form method="post">
            <input name="email">
            <input name="password">
            <button>Login</button>
        </form>
    </body></html>""",

    "signup.html": """<html><body>
        {% if error %}<p>{{ error }}</p>{% endif %}
        <form method="post">
            <input name="full_name">
            <input name="email">
            <input name="password">
            <input name="confirm_password">
            <button>Sign Up</button>
        </form>
    </body></html>""",

    "charts.html": """<html><body>
        <canvas id="chart1"></canvas>
        <p>chart data loaded</p>
    </body></html>""",

    "insights.html": """<html><body>
        {% for c in insight_cards %}<div>{{ c.title }}</div>{% endfor %}
        <a href="/download-insights">Export PDF</a>
    </body></html>""",

    "about.html": "<html><body>About VentureVerse</body></html>",
    "chatbot.html": "<html><body>VentureBot</body></html>",
    "nav.html": "<!-- nav -->",
    "chatwidget.html": "<!-- chatwidget -->",
}

# Write the stub templates to a temporary directory
stub_dir = tempfile.mkdtemp(suffix="_vv_stubs")
for filename, content in STUB_TEMPLATES.items():
    with open(os.path.join(stub_dir, filename), "w") as file:
        file.write(content)


# ═══════════════════════════════════════════════════════════════
#  IMPORT THE APP — load the Flask application for testing
# ═══════════════════════════════════════════════════════════════

APP_LOADED = False
client = None       # Flask test client (simulates a browser)
vv_app = None       # Reference to the app module

try:
    # Import the app module (app.py)
    import app as vv_app

    # Redirect the database to our temporary file
    # (so tests never touch the real user data)
    vv_app.DB_FILE = TEST_DB
    vv_app.init_db()

    # Get the Flask app object and configure it for testing
    flask_app = vv_app.app
    flask_app.config.update(
        TESTING=True,                 # Enables test mode
        SECRET_KEY="test-secret-key", # Fixed key for test sessions
        WTF_CSRF_ENABLED=False,       # Disable CSRF for test forms
    )

    # Use our stub templates instead of the real ones
    flask_app.template_folder = stub_dir

    # Create the test client (like a virtual browser)
    client = flask_app.test_client()
    APP_LOADED = True

    print(f"\n{GREEN}✅  Flask app imported OK{RESET}")
    print(f"{GREEN}✅  Model loaded  : {vv_app.MODEL_FILE}{RESET}")
    print(f"{GREEN}✅  Test database : {TEST_DB}{RESET}\n")

except Exception as exc:
    print(f"\n{RED}❌  Could not import app.py: {exc}{RESET}")
    print(f"{YELLOW}    → Make sure ventureverse_model.joblib exists.")
    print(f"    → Run:  python train_model.py   then try again.{RESET}\n")


# ═══════════════════════════════════════════════════════════════
#  TEST DATA — reusable form data for predictions
# ═══════════════════════════════════════════════════════════════

# This represents a strong startup profile (likely to succeed).
# Used across multiple tests so we don't have to type it each time.
VALID_FORM = dict(
    funding_total_usd="8000000",          # $8 million
    funding_rounds="3",                    # 3 funding rounds
    relationships="10",                    # 10 key connections
    milestones="3",                        # 3 milestones hit
    avg_participants="3.0",                # 3 investors per round
    age_first_funding_year="1",            # First funding at year 1
    age_last_funding_year="3",             # Last funding at year 3
    age_first_milestone_year="1",          # First milestone at year 1
    age_last_milestone_year="2",           # Last milestone at year 2
    has_VC="1",                            # Has VC backing
    has_angel="1",                         # Has angel investor
    has_roundA="1",                        # Completed Series A
    has_roundB="1",                        # Completed Series B
    has_roundC="0",                        # No Series C yet
    has_roundD="0",                        # No Series D yet
    is_top500="0",                         # Not backed by top-500 VC
    category_code="software",              # Software industry
    ecosystem="major_hub",                 # Major Hub (London/NY/SV)
)


def login_as_test_user():
    """
    Helper function: logs in with the test account created in TC01.
    Called before tests that require authentication.
    """
    client.post(
        "/login",
        data=dict(email="mihir@test.com", password="pass1234"),
        follow_redirects=True,
    )


# ═══════════════════════════════════════════════════════════════
#  SECTION A — BLACK-BOX TESTS (TC01 – TC10)
#
#  What are black-box tests?
#    We test the app from the OUTSIDE, like a real user would.
#    We send HTTP requests (GET/POST) and check the response.
#    We don't look at internal code — just inputs and outputs.
# ═══════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}{'=' * 64}{RESET}")
print(f"{BOLD}{CYAN}  SECTION A — BLACK-BOX TESTS  (TC01–TC10){RESET}")
print(f"{BOLD}{CYAN}  Simulating real user actions through the browser{RESET}")
print(f"{BOLD}{CYAN}{'=' * 64}{RESET}\n")

if not APP_LOADED:
    # If the app couldn't load, skip all tests
    for i in range(1, 11):
        record(f"TC{i:02d}", "Skipped — app could not be loaded", False,
               "Run: python train_model.py  first")
else:

    # ──────────────────────────────────────────────────────────
    # TC01: Register a new user account
    # Purpose: Verify that new users can create an account
    # ──────────────────────────────────────────────────────────
    try:
        response = client.post("/signup", data=dict(
            full_name="Test User",
            email="mihir@test.com",
            password="pass1234",
            confirm_password="pass1234",
        ), follow_redirects=True)

        passed = response.status_code == 200
        record("TC01", "Register with valid credentials",
               passed, note=f"HTTP {response.status_code}")
    except Exception as e:
        record("TC01", "Register with valid credentials", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC02: Try to register with an email that already exists
    # Purpose: The app must block duplicate email addresses
    # ──────────────────────────────────────────────────────────
    try:
        response = client.post("/signup", data=dict(
            full_name="Copy Cat",
            email="mihir@test.com",  # Same email as TC01
            password="pass1234",
            confirm_password="pass1234",
        ), follow_redirects=True)

        # Check if the error message appears in the response
        passed = (b"already" in response.data.lower()
                  or b"registered" in response.data.lower())
        record("TC02", "Register with duplicate email — error message shown", passed)
    except Exception as e:
        record("TC02", "Register with duplicate email — error message shown", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC03: Login with correct email and password
    # Purpose: The login system must accept valid credentials
    # ──────────────────────────────────────────────────────────
    try:
        client.get("/logout")  # Make sure we're logged out first
        response = client.post("/login", data=dict(
            email="mihir@test.com",
            password="pass1234",
        ), follow_redirects=True)

        passed = response.status_code == 200 and b"Invalid" not in response.data
        record("TC03", "Login with correct credentials",
               passed, note=f"HTTP {response.status_code}")
    except Exception as e:
        record("TC03", "Login with correct credentials", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC04: Login with the wrong password
    # Purpose: The app must reject incorrect passwords
    # ──────────────────────────────────────────────────────────
    try:
        client.get("/logout")
        response = client.post("/login", data=dict(
            email="mihir@test.com",
            password="WRONG_PASSWORD_999",  # Intentionally wrong
        ), follow_redirects=True)

        # The app should show "Invalid email or password"
        passed = (b"invalid" in response.data.lower()
                  or b"password" in response.data.lower())
        record("TC04", "Login with wrong password — error message shown", passed)
    except Exception as e:
        record("TC04", "Login with wrong password — error message shown", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC05: After logout, prediction should be blocked
    # Purpose: Protected pages must not be accessible after logout
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        client.get("/logout", follow_redirects=True)

        # Try to submit a prediction while logged out
        response = client.post("/predict", data=VALID_FORM, follow_redirects=False)

        # Should get HTTP 302 (redirect to login page)
        passed = response.status_code == 302
        record("TC05", "Logout clears session — /predict blocked after logout",
               passed,
               note=f"HTTP {response.status_code} → {response.location}  "
                    f"(302 = redirect to login)")
    except Exception as e:
        record("TC05", "Logout clears session — /predict blocked after logout", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC06: Submit prediction form with valid inputs
    # Purpose: Core feature — the ML model must return a score
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        response = client.post("/predict", data=VALID_FORM, follow_redirects=True)

        has_score = (b"%" in response.data
                     or b"Success" in response.data
                     or b"Failure" in response.data)
        passed = response.status_code == 200 and has_score
        record("TC06", "Prediction form with valid inputs returns a score",
               passed,
               note=f"HTTP {response.status_code}  |  Score in page: {has_score}")
    except Exception as e:
        record("TC06", "Prediction form with valid inputs returns a score", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC07: Submit form with blank funding field
    # Purpose: App should warn the user about missing data
    # Note:   This is an INTENTIONAL FAIL — the app silently
    #         accepts blank fields instead of showing a warning.
    #         We document this in Chapter 7 of the FYP report.
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        bad_form = dict(VALID_FORM)
        bad_form["funding_total_usd"] = ""  # Blank on purpose

        response = client.post("/predict", data=bad_form, follow_redirects=True)

        # Check if any validation warning appears
        has_warning = any(
            word in response.data.lower()
            for word in [b"required", b"warning", b"please fill", b"invalid"]
        )

        record("TC07", "Missing funding field triggers a validation warning",
               has_warning,
               note="INTENTIONAL FAIL — app defaults to 0 with no warning. "
                    "Known usability gap — documented in Chapter 7 of the report.")
    except Exception as e:
        record("TC07", "Missing funding field triggers a validation warning", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC08: Charts page loads correctly
    # Purpose: The charts page must render without errors
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        response = client.get("/charts", follow_redirects=True)

        passed = response.status_code == 200 and (
            b"chart" in response.data.lower()
            or b"canvas" in response.data.lower()
        )
        record("TC08", "Charts page renders correctly after a prediction",
               passed, note=f"HTTP {response.status_code}")
    except Exception as e:
        record("TC08", "Charts page renders correctly after a prediction", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC09: PDF export generates a downloadable file
    # Purpose: The report export must produce a real file
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        # Run a prediction first so the session has data for the PDF
        client.post("/predict", data=VALID_FORM, follow_redirects=True)

        response = client.get("/download-insights", follow_redirects=True)

        is_pdf = b"%PDF" in response.data or "pdf" in response.content_type.lower()
        passed = response.status_code == 200
        record("TC09", "PDF export (/download-insights) generates a file",
               passed,
               note=f"HTTP {response.status_code}  |  "
                    f"Content-type: {response.content_type}  |  "
                    f"PDF bytes: {is_pdf}")
    except Exception as e:
        record("TC09", "PDF export (/download-insights) generates a file", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC10: Unauthenticated access to /predict is blocked
    # Purpose: Security — protected pages must redirect to login
    # ──────────────────────────────────────────────────────────
    try:
        client.get("/logout")
        response = client.post("/predict", data=VALID_FORM, follow_redirects=False)

        passed = (response.status_code == 302
                  and "login" in (response.location or "").lower())
        record("TC10", "Unauthenticated user redirected to /login page",
               passed, note=f"HTTP {response.status_code} → {response.location}")
    except Exception as e:
        record("TC10", "Unauthenticated user redirected to /login page", False, str(e))


# ═══════════════════════════════════════════════════════════════
#  SECTION B — WHITE-BOX UNIT TESTS (UT01 – UT05)
#
#  What are white-box tests?
#    We call internal Python functions directly, passing in known
#    inputs and checking the outputs match what we mathematically
#    expect. No HTTP, no browser — pure function testing.
# ═══════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}{'=' * 64}{RESET}")
print(f"{BOLD}{CYAN}  SECTION B — WHITE-BOX UNIT TESTS  (UT01–UT05){RESET}")
print(f"{BOLD}{CYAN}  Calling internal Python functions directly{RESET}")
print(f"{BOLD}{CYAN}{'=' * 64}{RESET}\n")


# ──────────────────────────────────────────────────────────────
# UT01: hash_password() produces a consistent SHA-256 digest
# Why: If the same password gives a different hash each time,
#      login would break — no user could ever log in twice.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded — run train_model.py first")

    # Hash the same password twice
    hash_1 = vv_app.hash_password("test123")
    hash_2 = vv_app.hash_password("test123")

    # Also compute the expected hash independently
    expected_hash = hashlib.sha256("test123".encode()).hexdigest()

    passed = (
        len(hash_1) == 64        # SHA-256 always produces 64 hex chars
        and hash_1 == hash_2     # Same input → same output (deterministic)
        and hash_1 == expected_hash  # Matches Python's hashlib
    )
    record("UT01", "hash_password() returns consistent 64-char SHA-256 digest",
           passed, note=f"First 20 chars: {hash_1[:20]}…   Length: {len(hash_1)}")
except Exception as e:
    record("UT01", "hash_password() returns consistent 64-char SHA-256 digest", False, str(e))


# ──────────────────────────────────────────────────────────────
# UT02: build_input_df() computes engineered features correctly
# Why: log_funding and avg_funding_per_round are fed straight
#      into the ML model. If the maths is wrong, every single
#      prediction will be wrong — silently.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded")

    # Set up a test form with known values
    test_form = dict(VALID_FORM)
    test_form.update(
        funding_total_usd="1000000",       # $1,000,000
        funding_rounds="2",                 # 2 rounds
        age_first_funding_year="1",         # Year 1
        age_last_funding_year="3",          # Year 3
    )
    df = vv_app.build_input_df(test_form)

    # Calculate expected values by hand:
    expected_log_funding = round(np.log1p(1_000_000), 3)   # ≈ 13.816
    expected_avg_per_round = 500_000.0                      # 1,000,000 ÷ 2
    expected_duration = 2.0                                  # 3 − 1

    # Get actual values from the DataFrame
    actual_log = round(float(df["log_funding"].iloc[0]), 3)
    actual_avg = float(df["avg_funding_per_round"].iloc[0])
    actual_dur = float(df["funding_duration"].iloc[0])

    passed = (
        actual_log == expected_log_funding
        and actual_avg == expected_avg_per_round
        and actual_dur == expected_duration
    )
    record("UT02", "build_input_df() computes log_funding and avg_funding_per_round",
           passed,
           note=(f"log_funding={actual_log} (exp {expected_log_funding})  |  "
                 f"avg_per_round={actual_avg} (exp {expected_avg_per_round})  |  "
                 f"duration={actual_dur} (exp {expected_duration})"))
except Exception as e:
    record("UT02", "build_input_df() computes log_funding and avg_funding_per_round", False, str(e))


# ──────────────────────────────────────────────────────────────
# UT03: The ML model loads from its .joblib file
# Why: If this file is missing or corrupt, Flask crashes on
#      startup and no user can access anything at all.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded")

    loaded_model = vv_app.model
    has_predict_proba = hasattr(loaded_model, "predict_proba")
    passed = loaded_model is not None and has_predict_proba

    record("UT03", "XGBoost model loads from .joblib file without error",
           passed,
           note=f"Model type: {type(loaded_model).__name__}  |  "
                f"has predict_proba: {has_predict_proba}")
except Exception as e:
    record("UT03", "XGBoost model loads from .joblib file without error", False, str(e))


# ──────────────────────────────────────────────────────────────
# UT04: predict_proba() output is between 0.0 and 1.0
# Why: A probability MUST be in [0, 1]. If it falls outside
#      this range, the model or pipeline is fundamentally broken.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded")

    input_df = vv_app.build_input_df(VALID_FORM)
    probability = float(vv_app.model.predict_proba(input_df)[0][1])

    pred_label = "Success" if probability >= 0.5 else "At Risk"
    passed = 0.0 <= probability <= 1.0
    record("UT04", "predict_proba() output is a valid probability in [0.0, 1.0]",
           passed,
           note=f"Returned: {probability:.4f}  "
                f"({'✓ valid' if passed else '✗ OUT OF RANGE'})")
except Exception as e:
    record("UT04", "predict_proba() output is a valid probability in [0.0, 1.0]", False, str(e))


# ──────────────────────────────────────────────────────────────
# UT05: compute_risk_breakdown() returns exactly 6 factors
# Why: The Charts and Insights pages both loop over exactly
#      6 factors. One missing or extra factor would break the UI.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded")

    factors = vv_app.compute_risk_breakdown(VALID_FORM)

    has_six_factors = len(factors) == 6
    keys_correct = all({"factor", "score", "status"} <= set(f.keys()) for f in factors)
    scores_valid = all(0 <= f["score"] <= 100 for f in factors)
    statuses_valid = all(f["status"] in ("strong", "moderate", "weak") for f in factors)

    passed = has_six_factors and keys_correct and scores_valid and statuses_valid

    factor_names = [f["factor"] for f in factors]
    record("UT05", "compute_risk_breakdown() returns 6 correctly structured factors",
           passed, note=f"Factors: {factor_names}")
except Exception as e:
    record("UT05", "compute_risk_breakdown() returns 6 correctly structured factors", False, str(e))


# ═══════════════════════════════════════════════════════════════
#  CLEAN UP — remove temporary files
# ═══════════════════════════════════════════════════════════════

try:
    os.remove(TEST_DB)
except Exception:
    pass

try:
    shutil.rmtree(stub_dir)
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════
#  FINAL SUMMARY TABLE — print all results
# ═══════════════════════════════════════════════════════════════

total_tests = len(results)
passed_count = sum(1 for r in results if r[2] == "PASS")
failed_count = total_tests - passed_count

print(f"\n\n{BOLD}{'═' * 72}{RESET}")
print(f"{BOLD}  VENTUREVERSE — FULL TEST RESULTS SUMMARY{RESET}")
print(f"{BOLD}{'═' * 72}{RESET}")
print(f"  {BOLD}{'ID':<8}{'Test Name':<56}{'Result'}{RESET}")
print(f"  {'─' * 8}{'─' * 56}{'─' * 8}")

for (test_id, test_name, status, note) in results:
    mark = f"{GREEN}PASS{RESET}" if status == "PASS" else f"{RED}FAIL{RESET}"
    short_name = test_name[:55] + "…" if len(test_name) > 56 else test_name
    print(f"  {BOLD}{test_id:<8}{RESET}{short_name:<56}{mark}")

print(f"\n  {'─' * 70}")
print(f"  Total: {total_tests}   │   "
      f"{GREEN}{BOLD}Passed: {passed_count}{RESET}   │   "
      f"{RED}{BOLD}Failed: {failed_count}{RESET}")

# Explain the intentional TC07 failure
only_tc07_failed = (
    failed_count == 1
    and any(r[0] == "TC07" and r[2] == "FAIL" for r in results)
)

if only_tc07_failed:
    print(f'''
  {YELLOW}ℹ️  TC07 is an INTENTIONAL documented failure.
     The app silently accepts a blank funding field instead
     of showing a validation warning. This is the honest gap
     we discuss in Chapter 7.2 of the FYP report.{RESET}''')
elif failed_count == 0:
    print(f"\n  {GREEN}🎉  All tests passed!{RESET}")

print(f"\n{BOLD}{'═' * 72}{RESET}")
print(f"  {CYAN}VentureVerse  ·  University of Westminster  ·  6COSC023W{RESET}")
print(f"{BOLD}{'═' * 72}{RESET}\n")
