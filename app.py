"""Bank AI Early Warning & Retention Platform.

The primary workflow is intentionally small: connect bank data, surface churn
risk, inspect a customer, and turn the signal into a retention action. Legacy
analyst tools remain available from Advanced Tools.
"""
from io import BytesIO
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import plotly.express as px
import streamlit as st
from src.bank_client import MockBankClient
from src.config import get_settings
from src.audit import AuditService
from src.health import HealthService
from src.persistence import SQLiteStore
from src.opportunity import OpportunityConfig, OpportunityEngine, simulate_campaign
from src.predictor import ModelPredictor
from src.sync_service import SyncService
from src.reports import pdf_bytes

ROOT = Path(__file__).resolve().parent
MODEL_PATH, DATA_PATH = ROOT / "best_model.joblib", ROOT / "data" / "Churn_Modelling.csv"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
LABELS = {"Low Risk": "Low Risk", "Medium Risk": "Medium Risk",
          "High Risk": "High Risk", "Critical Risk": "Critical Risk"}

st.set_page_config(page_title="Bank AI Early Warning & Retention Platform", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{padding:2rem 2.2rem;border-radius:24px;background:radial-gradient(circle at 88% 12%,#42c2c9 0,#147d92 28%,#102a43 75%);color:#fff;margin-bottom:1.1rem;box-shadow:0 16px 40px rgba(16,42,67,.25)}
.hero h1{margin:0;font-size:2.35rem;letter-spacing:-.03em}.hero p{margin:.55rem 0 0;opacity:.94;font-size:1.08rem;max-width:850px}
.hero .flow{margin-top:1rem;font-size:.95rem;opacity:.95;padding:.7rem 1rem;border:1px solid rgba(255,255,255,.25);border-radius:12px;background:rgba(0,0,0,.12)}
.status-ribbon{display:flex;gap:.55rem;flex-wrap:wrap;margin:-.3rem 0 1.1rem}
.status-ribbon span{padding:.35rem .7rem;border-radius:999px;background:rgba(20,125,146,.14);border:1px solid rgba(20,125,146,.3);font-size:.82rem}
.decision-card{padding:1.2rem 1.3rem;border-radius:18px;background:linear-gradient(135deg,rgba(20,125,146,.16),rgba(20,125,146,.04));border:1px solid rgba(20,125,146,.32);margin:.8rem 0 1.2rem}
.decision-card h3{margin:0 0 .3rem}.decision-card p{margin:0;opacity:.8}
.product-card{padding:1rem 1.1rem;border:1px solid rgba(128,128,128,.28);border-radius:14px;background:var(--secondary-background-color,rgba(128,128,128,.12));min-height:112px}
.product-card h4{margin:0 0 .3rem;color:var(--text-color,inherit)}.product-card p{margin:0;color:var(--text-color,inherit);opacity:.8}
[data-testid="stMetric"]{background:var(--secondary-background-color,rgba(128,128,128,.12));padding:15px 14px;border-radius:16px;border:1px solid rgba(128,128,128,.28);min-height:94px}
[data-testid="stMetric"] label,[data-testid="stMetric"] [data-testid="stMetricLabel"]{color:var(--text-color,inherit) !important;opacity:.78}
[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--text-color,inherit) !important}
[data-testid="stMetric"] [data-testid="stMetricDelta"]{color:var(--text-color,inherit) !important;opacity:.8}
.section-note{color:var(--text-color,inherit);opacity:.78}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model, data = load_model(), load_data()
    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    threshold = float(metadata.get("decision_threshold", .5))
except Exception as exc:
    st.error(f"Startup failed: {exc}")
    st.stop()

settings = get_settings()
integration_store = SQLiteStore(settings.database_path)
demo_client = MockBankClient()
integration_predictor = ModelPredictor(MODEL_PATH, METRICS_PATH)
opportunity_engine = OpportunityEngine()
sync_service = SyncService(demo_client, integration_store, integration_predictor)
audit_service = AuditService(integration_store)
health_service = HealthService(integration_store, demo_client)

def risk_for(prob):
    if prob < .30: return "Low Risk", "Routine follow-up"
    if prob < .60: return "Medium Risk", "Contact and support"
    if prob < .75: return "High Risk", "Personalized retention offer"
    return "Critical Risk", "Immediate retention intervention"

def predict(frame):
    if frame.empty: raise ValueError("Empty input")
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing: raise ValueError("Missing columns: " + ", ".join(missing))
    probs = model.predict_proba(frame[FEATURES])[:, 1]
    output = frame.copy()
    output["Churn_Probability"] = probs
    output["Prediction"] = ["May Exit" if p >= threshold else "Likely Stay" for p in probs]
    levels_actions = [risk_for(p) for p in probs]
    output["Risk_Level"] = [LABELS[x[0]] for x in levels_actions]
    output["Recommended_Action"] = [x[1] for x in levels_actions]
    opportunity_rows = [
        opportunity_engine.score(row.to_dict(), float(prob), level[0])
        for (_, row), prob, level in zip(frame.iterrows(), probs, levels_actions)
    ]
    output["Opportunity_Score"] = [x["score"] for x in opportunity_rows]
    output["Customer_Value_Proxy"] = [x["relationship_value_proxy"] for x in opportunity_rows]
    output["Intervention_Opportunity"] = [x["intervention_component"] for x in opportunity_rows]
    output["Priority_Tier"] = [x["priority_tier"] for x in opportunity_rows]
    output["Opportunity_Segment"] = [x["segment"] for x in opportunity_rows]
    output["Recommended_Playbook"] = [x["playbook"] for x in opportunity_rows]
    return output.sort_values("Churn_Probability", ascending=False)

def feature_importance():
    try:
        prep, estimator = model.named_steps["preprocessor"], model.named_steps["model"]
        names = prep.get_feature_names_out()
        values = getattr(estimator, "feature_importances_", None)
        if values is None:
            values = abs(estimator.coef_[0])
        result = pd.DataFrame({"Feature": names, "Importance": values})
        result["Feature"] = result["Feature"].str.replace(r"^(numeric__|categorical__)", "", regex=True)
        return result.sort_values("Importance", ascending=False).head(12)
    except (AttributeError, KeyError, ValueError):
        return pd.DataFrame()

def csv_download(frame, name):
    st.download_button("⬇️ Download CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=name, mime="text/csv", use_container_width=True)

def report_downloads(frame, stem):
    csv_download(frame, stem + ".csv")
    try:
        st.download_button("⬇️ Download PDF", pdf_bytes(frame, stem),
                           file_name=stem + ".pdf", mime="application/pdf",
                           use_container_width=True)
    except RuntimeError as exc:
        st.caption(str(exc))

def demo_predictions(config=None):
    """Return the current demo-bank customers with the operational risk fields."""
    customers = pd.DataFrame(demo_client.list_customers())
    if customers.empty:
        return customers
    if config is None:
        return integration_predictor.predict_frame(customers)
    engine = OpportunityEngine(config)
    rows = []
    for _, customer in customers.iterrows():
        raw = customer.to_dict()
        prediction = integration_predictor.predict(raw)
        opportunity = engine.score(raw, prediction["probability"], prediction["risk"]["level"])
        rows.append(dict(raw, Churn_Probability=prediction["probability"],
                         Risk_Score=prediction["risk"]["score"], Risk_Level=prediction["risk"]["level"],
                         Recommended_Action=prediction["risk"]["action"], Prediction=prediction["prediction"],
                         Opportunity_Score=opportunity["score"],
                         Customer_Value_Proxy=opportunity["relationship_value_proxy"],
                         Intervention_Opportunity=opportunity["intervention_component"],
                         Priority_Tier=opportunity["priority_tier"],
                         Opportunity_Segment=opportunity["segment"],
                         Recommended_Playbook=opportunity["playbook"]))
    return pd.DataFrame(rows).sort_values("Opportunity_Score", ascending=False)

def customer_drivers(customer):
    """Give the relationship manager transparent, non-causal signal labels."""
    drivers = []
    if not customer.get("IsActiveMember", 1):
        drivers.append("Inactive relationship")
    if customer.get("Geography") == "Germany":
        drivers.append("Germany segment signal")
    if customer.get("NumOfProducts", 1) == 1:
        drivers.append("Single product relationship")
    if customer.get("Age", 0) >= 45:
        drivers.append("Older age segment")
    if customer.get("Balance", 0) == 0:
        drivers.append("Zero balance")
    return drivers or ["Combined model signal"]

def latest_sync():
    row = integration_store.conn.execute(
        "SELECT payload, created_at FROM sync_runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    try:
        return dict(json.loads(row["payload"]), created_at=row["created_at"])
    except (TypeError, json.JSONDecodeError):
        return None

st.markdown(
    '<div class="hero" dir="rtl"><h1>🛡️ Bank AI Retention Opportunity Engine</h1>'
    '<p>An academic platform that turns churn probability into an explainable retention opportunity.</p>'
    '<div class="flow"><b>Customer data</b> ← AI analysis ← opportunity (risk × demo value × intervention) ← playbook</div></div>'
    '<div class="status-ribbon"><span>🟢 Demo Bank Connected</span><span>🤖 Random Forest AI</span>'
    '<span>🔒 No real banking data</span><span>⚡ Early Warning Enabled</span></div>',
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("### Platform navigation")
    primary_page = st.radio(
        "Choose a workflow",
        ["Retention Command Center", "Executive Overview", "Bank Sync", "Early Warning",
         "Customer Profile", "Advanced Tools"],
        label_visibility="collapsed",
    )
    page = primary_page
    if primary_page == "Advanced Tools":
        with st.expander("Open analyst tools", expanded=True):
            page = st.selectbox(
                "Advanced analyst pages",
                ["Individual Customer", "Batch Analysis", "Data Quality",
                 "AI Prediction", "Bank Integration", "Synchronization",
                 "What-if", "Analytics", "Model Arena", "Explainable AI",
                 "Security & Audit", "Settings / Demo Mode", "Model Performance"],
                label_visibility="collapsed",
            )
    st.divider()
    synced_count = integration_store.conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    st.caption("Demo Mode • no credentials or external bank network")
    st.metric("Demo bank customers", f"{len(demo_client.list_customers()):,}")
    st.metric("Synced locally", f"{synced_count:,}")
    st.metric("Model decision threshold", f"{threshold:.0%}")

if page == "Retention Command Center":
    st.subheader("Retention Command Center | Opportunity Engine")
    st.markdown(
        '<div class="decision-card"><h3>From risk alert to retention opportunity</h3>'
        '<p>The system balances churn risk with demo relationship value and intervention opportunity, '
        'so the team knows where to start and why.</p></div>',
        unsafe_allow_html=True,
    )
    st.caption("Academic demo only: value-at-risk figures are demo-data proxies, not real financial claims.")
    with st.expander("Configure and explain the opportunity score", expanded=True):
        w1, w2, w3 = st.columns(3)
        with w1:
            risk_weight = st.slider("Risk weight", 0.0, 1.0, .45, .05)
        with w2:
            value_weight = st.slider("Demo value weight", 0.0, 1.0, .35, .05)
        with w3:
            intervention_weight = st.slider("Intervention opportunity weight", 0.0, 1.0, .20, .05)
        value_reference = st.number_input(
            "Demo value reference (proxy units)", min_value=1_000., value=100_000., step=5_000.,
            help="Calibration only. It is not a currency valuation.",
        )
        try:
            opportunity_config = OpportunityConfig(
                risk_weight=risk_weight, value_weight=value_weight,
                intervention_weight=intervention_weight, value_reference=value_reference,
            )
        except ValueError:
            st.error("Set at least one score weight above zero.")
            st.stop()
        st.caption(
            "Score = weighted churn probability + normalized balance/salary demo proxy + "
            "transparent intervention signals, normalized to 0–100. Weights are normalized by their sum."
        )
    opportunity_result = demo_predictions(opportunity_config)
    if opportunity_result.empty:
        st.info("No demo customer records are available. Run Bank Sync first.")
    else:
        opportunity_result["Proxy_Value_At_Risk"] = (
            opportunity_result["Churn_Probability"] * opportunity_result["Customer_Value_Proxy"]
        )
        portfolio_proxy = float(opportunity_result["Proxy_Value_At_Risk"].sum())
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Demo customers", f"{len(opportunity_result):,}")
        p2.metric("P1 / P2 opportunities", f"{int(opportunity_result['Priority_Tier'].str.startswith(('P1', 'P2')).sum()):,}")
        p3.metric("Top opportunity score", f"{opportunity_result['Opportunity_Score'].max():.1f}")
        p4.metric("Estimated value at risk (proxy)", f"{portfolio_proxy:,.0f}")
        st.caption("Estimated value at risk is a demo proxy, not bank revenue or a financial forecast.")

        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("#### Top 10 next actions")
            top10 = opportunity_result.sort_values("Opportunity_Score", ascending=False).head(10)
            st.dataframe(
                top10[["customer_id", "Opportunity_Score", "Priority_Tier", "Risk_Level",
                       "Opportunity_Segment", "Recommended_Playbook"]],
                use_container_width=True, hide_index=True,
            )
        with right:
            st.markdown("#### Opportunity by segment")
            segment = opportunity_result.groupby("Opportunity_Segment", as_index=False).agg(
                Customers=("customer_id", "count"),
                Mean_Opportunity=("Opportunity_Score", "mean"),
                Proxy_Value_At_Risk=("Proxy_Value_At_Risk", "sum"),
            ).sort_values("Mean_Opportunity", ascending=False)
            st.plotly_chart(
                px.bar(segment, x="Mean_Opportunity", y="Opportunity_Segment", orientation="h",
                       color="Proxy_Value_At_Risk", title="Mean score and proxy value at risk"),
                use_container_width=True,
            )
        st.markdown("#### Campaign simulator | scenario, not a factual claim")
        st.caption("Choose a target tier and assumptions to explore arithmetic using demo records only.")
        s1, s2, s3 = st.columns(3)
        with s1:
            tiers = st.multiselect("Target priority tiers", sorted(opportunity_result["Priority_Tier"].unique()),
                                   default=[t for t in ["P1 · Act now", "P2 · Next"] if t in set(opportunity_result["Priority_Tier"])])
        with s2:
            intervention_cost = st.number_input("Intervention cost / customer (demo units)", 0., 10_000., 100., 25.)
        with s3:
            expected_save_rate = st.slider("Expected save rate (assumption)", 0, 100, 20, 5) / 100
        target = opportunity_result[opportunity_result["Priority_Tier"].isin(tiers)]
        scenario = simulate_campaign(target.to_dict("records"), intervention_cost, expected_save_rate)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Targeted", scenario["targeted_customers"])
        c2.metric("Scenario preserved proxy", f"{scenario['scenario_estimated_value_preserved']:,.0f}")
        c3.metric("Scenario cost", f"{scenario['scenario_total_cost']:,.0f}")
        c4.metric("Scenario net proxy", f"{scenario['scenario_net_proxy_value']:,.0f}")
        st.caption(
            f"Illustrative estimate: {scenario['scenario_estimated_saved_customers']:.1f} customers "
            f"and {scenario['scenario_roi_proxy'] if scenario['scenario_roi_proxy'] is not None else '—'}× proxy ratio. "
            "Assumptions are user-entered and are not observed outcomes."
        )
        with st.expander("Recommended playbooks by risk and segment"):
            st.dataframe(
                opportunity_result[["Risk_Level", "Opportunity_Segment", "Recommended_Playbook"]]
                .drop_duplicates().sort_values(["Risk_Level", "Opportunity_Segment"]),
                use_container_width=True, hide_index=True,
            )
        csv_download(opportunity_result, "retention_opportunity_engine.csv")

elif page == "Executive Overview":
    st.subheader("Executive Command Center")
    st.markdown(
        '<p class="section-note">Turn customer data into a clear decision: who needs intervention now and what action is recommended?</p>',
        unsafe_allow_html=True,
    )
    demo_result = demo_predictions()
    high_or_critical = int(demo_result["Risk_Level"].isin(["HIGH", "CRITICAL"]).sum()) if not demo_result.empty else 0
    avg_probability = float(demo_result["Churn_Probability"].mean()) if not demo_result.empty else 0
    a,b,c,d = st.columns(4)
    a.metric("Demo customers", f"{len(demo_result):,}")
    b.metric("High / critical queue", f"{high_or_critical:,}")
    c.metric("Average churn probability", f"{avg_probability:.1%}")
    d.metric("Locally synced", f"{integration_store.conn.execute('SELECT COUNT(*) FROM customers').fetchone()[0]:,}")
    st.markdown(
        '<div class="decision-card"><h3>🎯 Decision of the day</h3>'
        f'<p><b>{high_or_critical}</b> customers are in the priority queue, with an average churn probability of <b>{avg_probability:.1%}</b>. '
        'Start with sync, then open Early Warning to take action.</p></div>',
        unsafe_allow_html=True,
    )
    st.caption("Demo data is safe and illustrative. Probability is a predictive signal, not a certainty or direct cause.")
    st.markdown("#### Demo narrative")
    flow = st.columns(4)
    flow_text = [
        ("1 · Connect", "Bring in bank customer records through Bank Sync."),
        ("2 · Predict", "The persisted model estimates churn probability."),
        ("3 · Prioritize", "Probability becomes a 0–100 risk score and band."),
        ("4 · Retain", "A recommended action gives the team a next step."),
    ]
    for column, (title, text) in zip(flow, flow_text):
        with column:
            st.markdown(f'<div class="product-card"><h4>{title}</h4><p>{text}</p></div>',
                        unsafe_allow_html=True)
    if st.button("▶ Run demo sync and open the risk queue", type="primary", use_container_width=True):
        st.session_state["last_sync"] = sync_service.run().__dict__
        st.success("Demo bank synced. Open Early Warning to work the prioritized queue.")
    latest = latest_sync()
    if latest:
        st.caption(f"Last sync: {latest.get('created_at', '—')} · fetched {latest.get('fetched', 0)} · "
                   f"created {latest.get('created', 0)} · updated {latest.get('updated', 0)}")
    left, right = st.columns(2)
    with left:
        geo = data.groupby("Geography", as_index=False).Exited.mean()
        geo["Churn rate"] = geo.Exited * 100
        st.plotly_chart(px.bar(geo, x="Geography", y="Churn rate", color="Geography",
                               title="Churn by geography", labels={"Churn rate":"%"}), use_container_width=True)
    with right:
        activity = data.assign(Activity=data.IsActiveMember.map({0:"Inactive",1:"Active"}))
        act = activity.groupby("Activity", as_index=False).Exited.mean()
        act["Churn rate"] = act.Exited * 100
        st.plotly_chart(px.bar(act, x="Activity", y="Churn rate", color="Activity",
                               title="Activity vs churn"), use_container_width=True)
    imp = feature_importance()
    if not imp.empty:
        st.plotly_chart(px.bar(imp.iloc[::-1], x="Importance", y="Feature", orientation="h",
                               title="Model feature importance"), use_container_width=True)

elif page == "Bank Sync":
    st.subheader("Bank Sync")
    st.markdown(
        '<p class="section-note">Connect the platform to a controlled in-process demo bank. '
        'No credentials, external network, or real-bank claims are used.</p>',
        unsafe_allow_html=True,
    )
    status_col, action_col = st.columns([2, 1])
    with status_col:
        st.markdown("#### Connection status")
        st.success("Connected · Demo Bank Client")
        st.json({"mode": "demo", "health": demo_client.health(),
                 "customers_available": len(demo_client.list_customers())})
    with action_col:
        st.markdown("#### Sync control")
        st.caption("Sync upserts customer records and creates a prediction snapshot.")
        run_sync = st.button("⟳ Sync demo bank", type="primary", use_container_width=True)
    if run_sync:
        with st.spinner("Syncing demo customer records…"):
            sync_result = sync_service.run()
        st.session_state["last_sync"] = sync_result.__dict__
        st.success("Sync completed successfully.")
    sync_result = st.session_state.get("last_sync") or latest_sync()
    if sync_result:
        st.markdown("#### Latest sync result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fetched", sync_result.get("fetched", 0))
        r2.metric("Created", sync_result.get("created", 0))
        r3.metric("Updated", sync_result.get("updated", 0))
        r4.metric("Failed", sync_result.get("failed", 0))
        st.caption(f"Sync ID: {sync_result.get('sync_id', '—')} · "
                   f"{float(sync_result.get('duration_ms', 0)):.1f} ms")
        if sync_result.get("errors"):
            st.warning("Some records could not be processed: " + "; ".join(sync_result["errors"]))
        st.dataframe(demo_predictions()[["customer_id", "Risk_Level", "Risk_Score",
                                         "Recommended_Action"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No sync has been run in this session. Start a demo sync to populate the local queue.")

elif page == "Individual Customer":
    st.subheader("Individual Customer Risk Assessment")
    with st.form("customer_form"):
        l, r = st.columns(2)
        with l:
            credit = st.number_input("Credit score", 300, 900, 650)
            geography = st.selectbox("Geography", sorted(data.Geography.dropna().unique()))
            gender = st.selectbox("Gender", sorted(data.Gender.dropna().unique()))
            age = st.number_input("Age", 18, 100, 40)
            tenure = st.number_input("Tenure", 0, 20, 5)
        with r:
            balance = st.number_input("Balance", min_value=0., value=75000., step=1000.)
            products = st.number_input("Products", 1, 4, 1)
            card = st.selectbox("Credit card?", [1,0], format_func=lambda x:"Yes" if x else "No")
            active = st.selectbox("Active member?", [1,0], format_func=lambda x:"Yes" if x else "No")
            salary = st.number_input("Estimated salary", min_value=0., value=100000., step=1000.)
        submit = st.form_submit_button("🔍 Analyze risk", type="primary", use_container_width=True)
    if submit:
        customer = pd.DataFrame([{"CreditScore":credit,"Geography":geography,"Gender":gender,"Age":age,
            "Tenure":tenure,"Balance":balance,"NumOfProducts":products,"HasCrCard":card,
            "IsActiveMember":active,"EstimatedSalary":salary}])
        try: result = predict(customer); p = float(result.Churn_Probability.iloc[0])
        except (ValueError, TypeError, KeyError) as exc: st.error(f"Invalid input: {exc}"); st.stop()
        raw_level, action = risk_for(p)
        x,y,z = st.columns(3); x.metric("Probability", f"{p:.1%}"); y.metric("Risk", LABELS[raw_level]); z.metric("Decision", "Review" if p >= threshold else "Approve")
        st.progress(min(max(p,0.),1.)); st.success(f"Recommendation: {action}")
        st.caption("These are transparent indicators, not proof of causality.")
        reasons = []
        if active == 0: reasons.append("Inactive membership is associated with higher churn.")
        if geography == "Germany": reasons.append("Germany has a higher historical churn rate in this dataset.")
        if products >= 3: reasons.append("Three or more products.")
        if age >= 45: reasons.append("Older age segment.")
        st.markdown("\n".join(f"- {r}" for r in (reasons or ["The model uses all features together."])))
        report_downloads(result, "individual_churn_report")

elif page == "Batch Analysis":
    st.subheader("Batch Retention Workflow")
    st.write("Upload a CSV with the required columns. The system will validate and prioritize the records.")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        try: batch = pd.read_csv(uploaded)
        except Exception as exc: st.error(f"Read error: {exc}"); st.stop()
        if batch.empty: st.warning("Empty file"); st.stop()
        missing = [c for c in FEATURES if c not in batch.columns]
        if missing: st.error("Missing columns: " + ", ".join(missing)); st.stop()
        try: result = predict(batch)
        except (ValueError, TypeError, KeyError) as exc: st.error(f"Invalid values: {exc}"); st.stop()
        high = result.Risk_Level.str.contains("High").sum()
        m1,m2,m3 = st.columns(3); m1.metric("Analyzed", f"{len(result):,}"); m2.metric("High", f"{high:,}"); m3.metric("Above threshold", f"{(result.Churn_Probability>=threshold).sum():,}")
        st.dataframe(result.head(100), use_container_width=True, hide_index=True)
        st.plotly_chart(px.histogram(result, x="Churn_Probability", nbins=20, color="Risk_Level",
                                     title="Risk distribution"), use_container_width=True)
        report_downloads(result, "customer_retention_report")

elif page == "Data Quality":
    st.subheader("Data Quality Checks")
    q = pd.DataFrame({"Column": data.columns, "Missing": data.isna().sum().values,
                      "Unique": data.nunique().values, "Dtype": data.dtypes.astype(str).values})
    a,b,c = st.columns(3); a.metric("Rows", f"{len(data):,}"); b.metric("Missing cells", f"{int(data.isna().sum().sum()):,}"); c.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
    st.dataframe(q, use_container_width=True, hide_index=True); csv_download(q, "data_quality_report.csv")

elif page == "AI Prediction":
    st.subheader("AI Prediction")
    st.write("The same persisted model and threshold used by the existing dashboard.")
    customer = {c: st.number_input(c, value=float(data[c].median())) if pd.api.types.is_numeric_dtype(data[c])
                else st.selectbox(c, sorted(data[c].dropna().unique())) for c in FEATURES}
    if st.button("Run prediction", type="primary"):
        try:
            result = integration_predictor.predict(customer)
            st.json(result); audit_service.record("prediction.created", {"customer_id": "ui"})
        except ValueError as exc: st.error(str(exc))

elif page == "Customer Profile":
    st.subheader("Customer Profile")
    customers = demo_client.list_customers()
    if not customers:
        st.info("No customer records are available. Run Bank Sync first.")
    else:
        customer_map = {c["customer_id"]: c for c in customers}
        cid = st.selectbox("Select a demo customer", list(customer_map))
        customer = customer_map[cid]
        prediction = integration_predictor.predict(customer)
        risk = prediction["risk"]
        st.markdown(
            '<p class="section-note">Move from a portfolio signal to an individual conversation: '
            'review the score, understand the drivers, and test a retention scenario.</p>',
            unsafe_allow_html=True,
        )
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Risk score", f"{risk['score']:.0f} / 100")
        p2.metric("Churn probability", f"{prediction['probability']:.1%}")
        p3.metric("Risk band", risk["level"])
        p4.metric("Opportunity score", f"{prediction['opportunity']['score']:.0f} / 100")
        st.progress(min(max(prediction["probability"], 0.0), 1.0))
        profile_left, profile_right = st.columns([1, 1])
        with profile_left:
            st.markdown("#### Recommended retention action")
            st.success(risk["action"])
            st.markdown("#### Opportunity playbook")
            st.info(prediction["opportunity"]["playbook"])
            st.markdown("#### Transparent model drivers")
            for driver in customer_drivers(customer):
                st.markdown(f"- {driver}")
        with profile_right:
            st.markdown("#### Customer record")
            st.dataframe(pd.DataFrame([customer]), use_container_width=True, hide_index=True)
        st.markdown("#### What-if: test a retention lever")
        w1, w2 = st.columns(2)
        with w1:
            what_if_active = st.checkbox("Customer is active", value=bool(customer["IsActiveMember"]),
                                         key=f"profile_active_{cid}")
        with w2:
            what_if_products = st.slider("Number of products", 1, 4, int(customer["NumOfProducts"]),
                                         key=f"profile_products_{cid}")
        scenario = dict(customer, IsActiveMember=int(what_if_active), NumOfProducts=what_if_products)
        scenario_prediction = integration_predictor.predict(scenario)
        delta = scenario_prediction["risk"]["score"] - risk["score"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Scenario risk score", f"{scenario_prediction['risk']['score']:.0f} / 100",
                  delta=f"{delta:+.0f} points vs current", delta_color="inverse")
        s2.metric("Scenario probability", f"{scenario_prediction['probability']:.1%}")
        s3.metric("Scenario action", scenario_prediction["risk"]["action"])
        st.caption("What-if changes are illustrative scenarios; they do not alter the bank record.")

elif page == "Early Warning":
    st.subheader("Early Warning Queue")
    st.markdown(
        '<p class="section-note">A prioritized work queue for relationship teams. Filter by risk '
        'band and score, then use the recommended action as the starting point.</p>',
        unsafe_allow_html=True,
    )
    result = demo_predictions()
    if result.empty:
        st.info("No customer records are available. Run Bank Sync first.")
    else:
        f1, f2 = st.columns([1, 1])
        with f1:
            levels = st.multiselect("Risk bands", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                    default=["CRITICAL", "HIGH", "MEDIUM"])
        with f2:
            minimum_score = st.slider("Minimum risk score", 0, 100, 25)
        queue = result[result["Risk_Level"].isin(levels) & (result["Risk_Score"] >= minimum_score)]
        q1, q2, q3 = st.columns(3)
        q1.metric("Customers in queue", len(queue))
        q2.metric("High / critical", int(queue["Risk_Level"].isin(["HIGH", "CRITICAL"]).sum()))
        q3.metric("Average score", f"{queue['Risk_Score'].mean():.0f}" if not queue.empty else "—")
        if queue.empty:
            st.warning("No customers match these filters. Broaden the bands or lower the score.")
        else:
            st.dataframe(queue[["customer_id", "Churn_Probability", "Risk_Score",
                                "Risk_Level", "Recommended_Action"]],
                         use_container_width=True, hide_index=True)
            csv_download(queue, "early_warning_report.csv")
            st.info("Next step: select a customer in Customer Profile to review drivers and test a what-if action.")

elif page == "Bank Integration":
    st.subheader("Bank Integration | Demo Mode")
    st.info("Demo Mode is in-process and requires no credentials or external bank network.")
    st.json({"mode": "demo", "health": demo_client.health(), "statistics": demo_client.statistics()})
    st.write("Webhook simulation")
    event = st.text_input("Event type", "customer.updated")
    if st.button("Send webhook"):
        response = demo_client.webhook({"type": event})
        audit_service.record("webhook.received", {"type": event})
        st.json(response)

elif page == "Synchronization":
    st.subheader("Synchronization")
    if st.button("Synchronize demo bank", type="primary"):
        result = sync_service.run()
        st.json(result.__dict__)
    st.json(health_service.check())

elif page == "What-if":
    st.subheader("What-if Simulator")
    base = demo_client.get_customer("DEMO-001")
    active = st.checkbox("Active member", value=bool(base["IsActiveMember"]))
    products = st.slider("Number of products", 1, 4, base["NumOfProducts"])
    scenario = dict(base, IsActiveMember=int(active), NumOfProducts=products)
    st.json(integration_predictor.predict(scenario))

elif page == "Analytics":
    st.subheader("Operational Analytics")
    st.metric("Demo customers", len(demo_client.list_customers()))
    st.metric("Audit events", len(audit_service.recent()))
    st.dataframe(pd.DataFrame(audit_service.recent()), use_container_width=True)

elif page == "Model Arena":
    st.subheader("Model Arena")
    st.dataframe(pd.DataFrame(metadata.get("results", [])), use_container_width=True, hide_index=True)
    st.caption("Metrics are read from outputs/metrics.json; no metrics are invented.")

elif page == "Explainable AI":
    st.subheader("Explainable AI")
    importance = integration_predictor.feature_importance().head(15)
    st.dataframe(importance, use_container_width=True, hide_index=True)
    if not importance.empty:
        st.plotly_chart(px.bar(importance.iloc[::-1], x="Importance", y="Feature", orientation="h"),
                        use_container_width=True)

elif page == "Security & Audit":
    st.subheader("Security & Audit")
    st.write("Secrets are loaded from environment only and redacted in audit details.")
    st.dataframe(pd.DataFrame(audit_service.recent()), use_container_width=True, hide_index=True)

elif page == "Settings / Demo Mode":
    st.subheader("Settings / Demo Mode")
    st.toggle("Demo Mode", value=settings.demo_mode, disabled=True)
    st.json({"database": str(settings.database_path), "api_url_configured": bool(settings.api_url),
             "webhook_secret_configured": bool(settings.webhook_secret)})

else:
    st.subheader("Model Performance")
    st.caption(f"Best model: {metadata.get('best_model','—')} | Threshold selected on validation: {threshold:.3f}")
    results = pd.DataFrame(metadata.get("results", []))
    if results.empty: st.warning("No metadata found. Run train_final.py first."); st.stop()
    st.dataframe(results.style.format({c:"{:.3f}" for c in results.columns if c not in ["Model"]}), use_container_width=True)
    st.plotly_chart(px.bar(results, x="Model", y=["F1-score","Recall","Precision"], barmode="group",
                           title="Model comparison"), use_container_width=True)
    cm = metadata.get("test_confusion_matrix")
    if cm:
        st.write("Test confusion matrix (rows: actual, columns: predicted)")
        st.dataframe(pd.DataFrame(cm, index=["Stayed","Exited"], columns=["Predicted stayed","Predicted exited"]))
