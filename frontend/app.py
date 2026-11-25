import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

SAMPLES = {
    "Rear-end collision": (
        "POL-LIAB-000001",
        "I was rear-ended at a red light on Main Street on 2026-08-15 while "
        "driving my sedan. The other driver's insurance admitted fault. Repair "
        "estimate came to $4,200. No injuries, no other parties involved "
        "beyond the other driver.",
    ),
    "Unreadable submission": ("", "asdf qwer zxcv 123 ???"),
}

STATUS_COLORS = {
    "approved": "green",
    "denied": "red",
    "escalated": "orange",
    "pending_human_review": "orange",
}

st.set_page_config(page_title="Claims Triage", page_icon="🗂️", layout="wide")
st.title("Claims Triage")

api_url = st.sidebar.text_input("API URL", DEFAULT_API_URL)
if st.sidebar.button("Check API"):
    try:
        st.sidebar.success(httpx.get(f"{api_url}/health", timeout=5).json())
    except Exception as e:
        st.sidebar.error(f"Unreachable: {e}")

try:
    model_info = httpx.get(f"{api_url}/models", timeout=10).json()
    model_choices = model_info["choices"]
    default_index = model_choices.index(model_info["default"])
except Exception:
    model_choices, default_index = [], 0

CUSTOM = "Custom (use my own API key)"
model = st.sidebar.selectbox("Model", [*model_choices, CUSTOM], index=default_index)
custom_llm = None
if model == CUSTOM:
    custom_provider = st.sidebar.selectbox("Provider", ["gemini", "groq"])
    custom_model = st.sidebar.text_input("Model name", placeholder="e.g. llama-3.3-70b-versatile")
    custom_key = st.sidebar.text_input("API key", type="password")
    st.sidebar.caption("Sent to the API for this request only. It is not stored or logged.")
    if custom_model and custom_key:
        custom_llm = {"provider": custom_provider, "model": custom_model, "api_key": custom_key}
    model = None

sample = st.selectbox("Load a sample", ["(none)", *SAMPLES])
sample_policy, sample_text = SAMPLES.get(sample, ("", ""))

with st.form("claim"):
    raw_text = st.text_area("Claim description", value=sample_text, height=180)
    policy_number = st.text_input("Policy number", value=sample_policy)
    submitted = st.form_submit_button("Submit claim", type="primary")

if submitted:
    if not raw_text.strip():
        st.warning("Enter a claim description.")
        st.stop()
    if model is None and custom_llm is None:
        st.warning("Enter a model name and API key for the custom option, or pick a listed model.")
        st.stop()
    try:
        with st.spinner("Running pipeline..."):
            response = httpx.post(
                f"{api_url}/claims",
                json={"raw_text": raw_text, "policy_number": policy_number or None, "model": model, "custom_llm": custom_llm},
                timeout=300,
            )
            response.raise_for_status()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()
    st.session_state["result"] = response.json()

result = st.session_state.get("result")
if result:
    decision = result.get("decision")
    packet = result.get("escalation_packet")

    for error in result.get("errors", []):
        st.error(error)

    if decision:
        color = STATUS_COLORS.get(decision["status"], "gray")
        st.subheader(f":{color}[{decision['status'].replace('_', ' ').title()}]")
        if decision.get("approved_amount") is not None:
            st.metric("Approved amount", f"${decision['approved_amount']:,.2f}")
        st.write(decision["reasoning"])

    if packet:
        st.warning(
            f"Routed to **{packet['queue'].replace('_', ' ')}** queue. "
            f"Reasons: {', '.join(r.replace('_', ' ') for r in packet['escalation_reasons'])}"
        )

    claim_tab, fraud_tab, policy_tab, raw_tab = st.tabs(
        ["Extracted claim", "Fraud risk", "Policy", "Raw JSON"]
    )

    with claim_tab:
        claim = result.get("extracted_claim")
        if claim:
            st.metric("Extraction confidence", f"{claim['extraction_confidence']:.0%}")
            st.json(claim)
        else:
            st.info("No claim extracted.")

    with fraud_tab:
        fraud = result.get("fraud_result")
        if fraud:
            left, right = st.columns(2)
            left.metric("Risk score", f"{fraud['risk_score']:.2f}")
            right.metric("Risk tier", fraud["risk_tier"])
            if fraud["top_contributing_features"]:
                st.dataframe(fraud["top_contributing_features"], hide_index=True)
        else:
            st.info("Fraud scoring did not run.")

    with policy_tab:
        policy = result.get("policy_result")
        if policy:
            a, b, c, d = st.columns(4)
            a.metric("Covered", {True: "Yes", False: "No"}.get(policy["is_covered"], "Unknown"))
            b.metric("Deductible", f"${policy['deductible']:,.0f}" if policy["deductible"] is not None else "—")
            c.metric("Limit", f"${policy['coverage_limit']:,.0f}" if policy["coverage_limit"] is not None else "—")
            d.metric("Retrieval confidence", f"{policy['retrieval_confidence']:.0%}")
            if policy["exclusions_matched"]:
                st.write("Exclusions matched:", ", ".join(policy["exclusions_matched"]))
            for clause in policy["relevant_clauses"]:
                with st.expander(f"{clause['document_id']} — {clause['section']} ({clause['relevance_score']:.2f})"):
                    st.write(clause["text_excerpt"])
        else:
            st.info("Policy retrieval did not run.")

    with raw_tab:
        st.json(result)
