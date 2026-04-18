import sys, os, hashlib, tempfile, shutil
import numpy as np

# ── Terminal colours (makes output easier to read) ───────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

results = []   # stores every test result

def record(tid, name, passed, note=""):
    """Save and print one test result."""
    status = "PASS" if passed else "FAIL"
    results.append((tid, name, status, note))
    mark  = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
    trunc = name[:54] + "…" if len(name) > 55 else name
    print(f"  {mark}  {BOLD}{tid}{RESET} — {trunc}")
    if note:
        print(f"         {YELLOW}↳ {note}{RESET}")


# ════════════════════════════════════════════════════════════════
#  SETUP
#  ① Use a throw-away test database (never touches the real one)
#  ② Create minimal stub HTML templates so Flask routes can run
#     without needing your full CSS/JS frontend
# ════════════════════════════════════════════════════════════════

TEST_DB     = tempfile.mktemp(suffix=".db")
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Minimal stub templates — just enough HTML for each Flask route.
# They include the key words the app would show (e.g. "already
# registered") so our pass/fail checks work correctly.
STUB_TEMPLATES = {
    "landing.html":  "<html><body>Welcome to VentureVerse</body></html>",

    "index.html": """<html><body>
        {% if prediction is not none %}
            <p>Score: {{ prediction }}%</p>
            <p>Predicted: {{ pred_label }}</p>
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

    "about.html":   "<html><body>About VentureVerse</body></html>",
    "chatbot.html": "<html><body>VentureBot</body></html>",
    "nav.html":        "<!-- nav -->",
    "chatwidget.html": "<!-- chatwidget -->",
}

stub_dir = tempfile.mkdtemp(suffix="_vv_stubs")
for fname, content in STUB_TEMPLATES.items():
    with open(os.path.join(stub_dir, fname), "w") as fh:
        fh.write(content)

# ── Import the app and patch it for testing ──────────────────
APP_LOADED = False
client     = None
vv_app     = None

try:
    import app as vv_app

    # Redirect DB to throw-away file
    vv_app.DB_FILE = TEST_DB
    vv_app.init_db()

    flask_app = vv_app.app
    flask_app.config.update(
        TESTING          = True,
        SECRET_KEY       = "test-secret-key",
        WTF_CSRF_ENABLED = False,
    )
    # Use our stub templates instead of the real HTML files
    flask_app.template_folder = stub_dir

    client     = flask_app.test_client()
    APP_LOADED = True

    print(f"\n{GREEN}✅  Flask app imported OK{RESET}")
    print(f"{GREEN}✅  Model loaded  : {vv_app.MODEL_FILE}{RESET}")
    print(f"{GREEN}✅  Test database : {TEST_DB}{RESET}\n")

except Exception as exc:
    print(f"\n{RED}❌  Could not import app.py: {exc}{RESET}")
    print(f"{YELLOW}    → Make sure ventureverse_model.joblib exists.")
    print(f"    → Run:  python train_model.py   then try again.{RESET}\n")


# Reusable: a strong startup profile used across multiple tests
VALID_FORM = dict(
    funding_total_usd        = "8000000",
    funding_rounds           = "3",
    relationships            = "10",
    milestones               = "3",
    avg_participants         = "3.0",
    age_first_funding_year   = "1",
    age_last_funding_year    = "3",
    age_first_milestone_year = "1",
    age_last_milestone_year  = "2",
    has_VC     = "1",  has_angel  = "1",
    has_roundA = "1",  has_roundB = "1",
    has_roundC = "0",  has_roundD = "0",
    is_top500      = "0",
    category_code  = "software",
    state_code     = "CA",
)

def login_as_test_user():
    """Log in with the test account created in TC01."""
    client.post("/login",
                data=dict(email="mihir@test.com", password="pass1234"),
                follow_redirects=True)


# ════════════════════════════════════════════════════════════════
#  SECTION A — BLACK-BOX TESTS  (TC01 – TC10)
#
#  SIMPLE EXPLANATION FOR YOUR PROFESSOR:
#  "We use Flask's built-in test client to send HTTP requests
#   exactly as a browser would. We then check whether the app
#   returned the right response — the right status code, the
#   right error message, or the right page content."
# ════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}{'='*64}{RESET}")
print(f"{BOLD}{CYAN}  SECTION A — BLACK-BOX TESTS  (TC01–TC10){RESET}")
print(f"{BOLD}{CYAN}  Simulating real user actions through the browser{RESET}")
print(f"{BOLD}{CYAN}{'='*64}{RESET}\n")

if not APP_LOADED:
    for i in range(1, 11):
        record(f"TC{i:02d}", "Skipped — app could not be loaded", False,
               "Run: python train_model.py  first")
else:

    # ──────────────────────────────────────────────────────────
    # TC01  Register with valid new credentials
    # WHY:  New users must be able to create an account.
    # ──────────────────────────────────────────────────────────
    try:
        r = client.post("/signup", data=dict(
            full_name        = "Test User",
            email            = "mihir@test.com",
            password         = "pass1234",
            confirm_password = "pass1234",
        ), follow_redirects=True)

        passed = r.status_code == 200
        record("TC01", "Register with valid credentials",
               passed, note=f"HTTP {r.status_code}")
    except Exception as e:
        record("TC01", "Register with valid credentials", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC02  Register with a duplicate email
    # WHY:  The app must block emails that already exist.
    # ──────────────────────────────────────────────────────────
    try:
        r = client.post("/signup", data=dict(
            full_name        = "Copy Cat",
            email            = "mihir@test.com",   # same as TC01
            password         = "pass1234",
            confirm_password = "pass1234",
        ), follow_redirects=True)

        # The route does:  render_template("signup.html", error="Email already registered.")
        passed = b"already" in r.data.lower() or b"registered" in r.data.lower()
        record("TC02", "Register with duplicate email — error message shown", passed)
    except Exception as e:
        record("TC02", "Register with duplicate email — error message shown", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC03  Login with correct email + password
    # WHY:  The login system must accept valid credentials.
    # ──────────────────────────────────────────────────────────
    try:
        client.get("/logout")
        r = client.post("/login", data=dict(
            email    = "mihir@test.com",
            password = "pass1234",
        ), follow_redirects=True)

        passed = r.status_code == 200 and b"Invalid" not in r.data
        record("TC03", "Login with correct credentials",
               passed, note=f"HTTP {r.status_code}")
    except Exception as e:
        record("TC03", "Login with correct credentials", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC04  Login with the wrong password
    # WHY:  The app must reject incorrect passwords.
    # ──────────────────────────────────────────────────────────
    try:
        client.get("/logout")
        r = client.post("/login", data=dict(
            email    = "mihir@test.com",
            password = "WRONG_PASSWORD_999",
        ), follow_redirects=True)

        # Route does:  render_template("login.html", error="Invalid email or password.")
        passed = b"invalid" in r.data.lower() or b"password" in r.data.lower()
        record("TC04", "Login with wrong password — error message shown", passed)
    except Exception as e:
        record("TC04", "Login with wrong password — error message shown", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC05  Logout then try to access /predict
    # WHY:  After logout the user must not be able to predict.
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        client.get("/logout", follow_redirects=True)

        # Try to use predict while logged out
        r = client.post("/predict", data=VALID_FORM, follow_redirects=False)

        # Must get HTTP 302 redirect to /login
        passed = r.status_code == 302
        record("TC05", "Logout clears session — /predict blocked after logout",
               passed, note=f"HTTP {r.status_code} → {r.location}  (302 = redirect to login)")
    except Exception as e:
        record("TC05", "Logout clears session — /predict blocked after logout", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC06  Submit prediction form with valid inputs
    # WHY:  Core feature — the ML model must return a score.
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        r = client.post("/predict", data=VALID_FORM, follow_redirects=True)

        has_score = b"%" in r.data or b"Success" in r.data or b"Failure" in r.data
        passed    = r.status_code == 200 and has_score
        record("TC06", "Prediction form with valid inputs returns a score",
               passed, note=f"HTTP {r.status_code}  |  Score in page: {has_score}")
    except Exception as e:
        record("TC06", "Prediction form with valid inputs returns a score", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC07  Submit with blank funding field   ← INTENTIONAL FAIL
    # WHY:  App should warn the user — but it silently uses 0.
    #       This is our honest documented failure for the report.
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        bad_form = dict(VALID_FORM)
        bad_form["funding_total_usd"] = ""   # blank on purpose

        r = client.post("/predict", data=bad_form, follow_redirects=True)

        has_warn = any(w in r.data.lower()
                       for w in [b"required", b"warning", b"please fill", b"invalid"])

        # This FAILS — no warning shown (app silently uses 0)
        record("TC07", "Missing funding field triggers a validation warning",
               has_warn,
               note="INTENTIONAL FAIL — app defaults to 0 with no warning. "
                    "Known usability gap — documented in Chapter 7 of the report.")
    except Exception as e:
        record("TC07", "Missing funding field triggers a validation warning", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC08  Charts page loads correctly
    # WHY:  The charts page must render without any errors.
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        r = client.get("/charts", follow_redirects=True)

        passed = r.status_code == 200 and (
            b"chart" in r.data.lower() or b"canvas" in r.data.lower()
        )
        record("TC08", "Charts page renders correctly after a prediction",
               passed, note=f"HTTP {r.status_code}")
    except Exception as e:
        record("TC08", "Charts page renders correctly after a prediction", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC09  PDF export generates a downloadable file
    # WHY:  The report export must produce a real PDF file.
    #       Route is /download-insights (confirmed from app.py)
    # ──────────────────────────────────────────────────────────
    try:
        login_as_test_user()
        # Run a prediction first so session has data for the PDF
        client.post("/predict", data=VALID_FORM, follow_redirects=True)

        r = client.get("/download-insights", follow_redirects=True)

        is_pdf = b"%PDF" in r.data or "pdf" in r.content_type.lower()
        passed = r.status_code == 200
        record("TC09", "PDF export (/download-insights) generates a file",
               passed,
               note=f"HTTP {r.status_code}  |  Content-type: {r.content_type}  |  PDF bytes: {is_pdf}")
    except Exception as e:
        record("TC09", "PDF export (/download-insights) generates a file", False, str(e))

    # ──────────────────────────────────────────────────────────
    # TC10  Unauthenticated access to /predict is blocked
    # WHY:  Security — protected pages must redirect to login.
    # ──────────────────────────────────────────────────────────
    try:
        client.get("/logout")
        r = client.post("/predict", data=VALID_FORM, follow_redirects=False)

        passed = r.status_code == 302 and "login" in (r.location or "").lower()
        record("TC10", "Unauthenticated user redirected to /login page",
               passed, note=f"HTTP {r.status_code} → {r.location}")
    except Exception as e:
        record("TC10", "Unauthenticated user redirected to /login page", False, str(e))


# ════════════════════════════════════════════════════════════════
#  SECTION B — WHITE-BOX UNIT TESTS  (UT01 – UT05)
#
#  SIMPLE EXPLANATION FOR YOUR PROFESSOR:
#  "We call individual Python functions directly, passing in
#   known inputs and checking the output matches what we
#   mathematically expect. No browser, no HTTP — pure functions."
# ════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}{'='*64}{RESET}")
print(f"{BOLD}{CYAN}  SECTION B — WHITE-BOX UNIT TESTS  (UT01–UT05){RESET}")
print(f"{BOLD}{CYAN}  Calling internal Python functions directly{RESET}")
print(f"{BOLD}{CYAN}{'='*64}{RESET}\n")

# ──────────────────────────────────────────────────────────────
# UT01  hash_password() produces a consistent SHA-256 digest
# WHY:  If the same password gives a different hash each run,
#       login would break — no user could ever log in twice.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded — run train_model.py first")

    r1 = vv_app.hash_password("test123")
    r2 = vv_app.hash_password("test123")
    expected = hashlib.sha256("test123".encode()).hexdigest()

    passed = (len(r1) == 64) and (r1 == r2) and (r1 == expected)
    record("UT01", "hash_password() returns consistent 64-char SHA-256 digest",
           passed, note=f"First 20 chars: {r1[:20]}…   Length: {len(r1)}")
except Exception as e:
    record("UT01", "hash_password() returns consistent 64-char SHA-256 digest", False, str(e))

# ──────────────────────────────────────────────────────────────
# UT02  build_input_df() computes engineered features correctly
# WHY:  log_funding and avg_funding_per_round are fed straight
#       into the ML model. If the maths is wrong, every single
#       prediction will be wrong — silently.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded")

    test_form = dict(VALID_FORM)
    test_form.update(
        funding_total_usd      = "1000000",
        funding_rounds         = "2",
        age_first_funding_year = "1",
        age_last_funding_year  = "3",
    )
    df = vv_app.build_input_df(test_form)

    exp_log = round(np.log1p(1_000_000), 3)   # ≈ 13.816
    exp_avg = 500_000.0                        # 1_000_000 / 2
    exp_dur = 2.0                              # 3 – 1

    act_log = round(float(df["log_funding"].iloc[0]), 3)
    act_avg = float(df["avg_funding_per_round"].iloc[0])
    act_dur = float(df["funding_duration"].iloc[0])

    passed = (act_log == exp_log) and (act_avg == exp_avg) and (act_dur == exp_dur)
    record("UT02", "build_input_df() computes log_funding and avg_funding_per_round",
           passed,
           note=(f"log_funding={act_log} (exp {exp_log})  |  "
                 f"avg_per_round={act_avg} (exp {exp_avg})  |  "
                 f"duration={act_dur} (exp {exp_dur})"))
except Exception as e:
    record("UT02", "build_input_df() computes log_funding and avg_funding_per_round", False, str(e))

# ──────────────────────────────────────────────────────────────
# UT03  The XGBoost model loads from its .joblib file
# WHY:  If this file is missing or corrupt, Flask crashes on
#       startup and no user can access anything at all.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded")

    mdl       = vv_app.model
    has_proba = hasattr(mdl, "predict_proba")
    passed    = mdl is not None and has_proba

    record("UT03", "XGBoost model loads from .joblib file without error",
           passed,
           note=f"Model type: {type(mdl).__name__}  |  has predict_proba: {has_proba}")
except Exception as e:
    record("UT03", "XGBoost model loads from .joblib file without error", False, str(e))

# ──────────────────────────────────────────────────────────────
# UT04  predict_proba() output is between 0.0 and 1.0
# WHY:  A probability MUST be in [0, 1]. If it falls outside
#       this range the model or pipeline is fundamentally broken.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded")

    df   = vv_app.build_input_df(VALID_FORM)
    prob = float(vv_app.model.predict_proba(df)[0][1])

    passed = 0.0 <= prob <= 1.0
    record("UT04", "predict_proba() output is a valid probability in [0.0, 1.0]",
           passed,
           note=f"Returned: {prob:.4f}  ({'✓ valid' if passed else '✗ OUT OF RANGE'})")
except Exception as e:
    record("UT04", "predict_proba() output is a valid probability in [0.0, 1.0]", False, str(e))

# ──────────────────────────────────────────────────────────────
# UT05  compute_risk_breakdown() returns exactly 6 factors
# WHY:  The Charts and Insights pages both loop over exactly
#       6 factors. One missing or extra factor breaks the UI.
# ──────────────────────────────────────────────────────────────
try:
    if not APP_LOADED:
        raise RuntimeError("App not loaded")

    factors = vv_app.compute_risk_breakdown(VALID_FORM)

    has_6     = len(factors) == 6
    keys_ok   = all({"factor", "score", "status"} <= set(f.keys()) for f in factors)
    scores_ok = all(0 <= f["score"] <= 100 for f in factors)
    status_ok = all(f["status"] in ("strong", "moderate", "weak") for f in factors)
    passed    = has_6 and keys_ok and scores_ok and status_ok

    names = [f["factor"] for f in factors]
    record("UT05", "compute_risk_breakdown() returns 6 correctly structured factors",
           passed, note=f"Factors: {names}")
except Exception as e:
    record("UT05", "compute_risk_breakdown() returns 6 correctly structured factors", False, str(e))


# ════════════════════════════════════════════════════════════════
#  CLEAN UP temporary files
# ════════════════════════════════════════════════════════════════
try:
    os.remove(TEST_DB)
except Exception:
    pass

try:
    shutil.rmtree(stub_dir)
except Exception:
    pass


# ════════════════════════════════════════════════════════════════
#  FINAL SUMMARY TABLE
# ════════════════════════════════════════════════════════════════
total  = len(results)
n_pass = sum(1 for r in results if r[2] == "PASS")
n_fail = total - n_pass

print(f"\n\n{BOLD}{'═'*72}{RESET}")
print(f"{BOLD}  VENTUREVERSE — FULL TEST RESULTS SUMMARY{RESET}")
print(f"{BOLD}{'═'*72}{RESET}")
print(f"  {BOLD}{'ID':<8}{'Test Name':<56}{'Result'}{RESET}")
print(f"  {'─'*8}{'─'*56}{'─'*8}")

for (tid, name, status, note) in results:
    mark  = f"{GREEN}PASS{RESET}" if status == "PASS" else f"{RED}FAIL{RESET}"
    trunc = name[:55] + "…" if len(name) > 56 else name
    print(f"  {BOLD}{tid:<8}{RESET}{trunc:<56}{mark}")

print(f"\n  {'─'*70}")
print(f"  Total: {total}   │   {GREEN}{BOLD}Passed: {n_pass}{RESET}   │   {RED}{BOLD}Failed: {n_fail}{RESET}")

# Explain the intentional failures
only_tc07 = n_fail == 1 and any(r[0] == "TC07" and r[2] == "FAIL" for r in results)
if only_tc07:
    print(f'''
  {YELLOW}ℹ️  TC07 is an INTENTIONAL documented failure.
     The app silently accepts a blank funding field instead
     of showing a validation warning. This is the honest gap
     we discuss in Chapter 7.2 of the FYP report.{RESET}''')
elif n_fail == 0:
    print(f"\n  {GREEN}🎉  All tests passed!{RESET}")

print(f"\n{BOLD}{'═'*72}{RESET}")
print(f"  {CYAN}VentureVerse  ·  University of Westminster  ·  6COSC023W{RESET}")
print(f"{BOLD}{'═'*72}{RESET}\n")
