"""
app.py
Frontend for the Self-Corrective RAG pipeline (Streamlit).

Shows the full pipeline running live: retrieval -> adaptive verification ->
self-correction loop (with reformulated queries) -> answer generation with
citations -> post-hoc hallucination check.

Run with:
    streamlit run src/app.py
"""

import streamlit as st

from retriever import Retriever
from verifier import Verifier
from corrector import Corrector
from generator import Generator
from hallucination_check import HallucinationChecker


st.set_page_config(page_title="Self-Corrective RAG", layout="wide")


@st.cache_resource(show_spinner="Loading models and index (first run only)...")
def load_pipeline():
    r = Retriever()
    v = Verifier()
    c = Corrector(retriever=r, verifier=v)
    g = Generator()
    hc = HallucinationChecker()
    return r, v, c, g, hc


st.title("Self-Corrective RAG")
st.caption("Adaptive Evidence Verification for Hallucination-Resistant Answer Generation")

try:
    retriever, verifier, corrector, generator, checker = load_pipeline()
except Exception as e:
    st.error(
        f"Could not load the pipeline: {e}\n\n"
        "Make sure you've run `python src/indexing.py` first, and that your "
        "`.env` file has a valid GROQ_API_KEY."
    )
    st.stop()

query = st.text_input(
    "Ask a question",
    placeholder="e.g. Were Scott Derrickson and Ed Wood of the same nationality?",
)
run_button = st.button("Run", type="primary")

if run_button and query.strip():
    with st.spinner("Running retrieval + adaptive verification + self-correction..."):
        outcome = corrector.run(query)

    # ---- Correction loop trace ----
    st.subheader("Self-correction trace")
    for attempt in outcome.attempts:
        status_icon = "✅" if attempt.verification_passed else "❌"
        with st.expander(
            f"{status_icon} Iteration {attempt.iteration} — "
            f"score={attempt.verification_score}, strictness={attempt.strictness}",
            expanded=(attempt.iteration == len(outcome.attempts)),
        ):
            st.markdown(f"**Query used:** {attempt.query_used}")
            st.markdown(f"**Retrieval confidence:** {attempt.retrieval_confidence}")
            st.markdown(f"**Verification score:** {attempt.verification_score}")
            if attempt.trigger_reason:
                st.markdown(f"**Why correction triggered:** {attempt.trigger_reason}")

    st.divider()

    if not outcome.final_passed:
        st.error(
            "Verification never passed after all correction attempts — "
            "the system is refusing to answer rather than risk hallucination. "
            "This usually means your corpus doesn't contain the needed evidence."
        )
    else:
        st.success(f"Verification passed on iteration {len(outcome.attempts)}")

        with st.spinner("Generating answer with citations..."):
            result = generator.generate(outcome.final_query, outcome.final_retrieval)

        with st.spinner("Running post-hoc hallucination check..."):
            report = checker.check(result)

        # ---- Final answer ----
        st.subheader("Answer")
        st.markdown(f"**{result.answer_text}**")

        col1, col2 = st.columns(2)
        col1.metric("Faithfulness score", report.faithfulness_score)
        col2.metric("Flagged claims", report.flagged_count)

        if report.flagged_count > 0:
            st.warning(f"Verified (clean) answer: {report.clean_answer}")

        # ---- Claim-level citations ----
        st.subheader("Claims & evidence")
        for check in report.claim_checks:
            ok = check.entailment_score >= 0.5
            icon = "✅" if ok else "⚠️"
            with st.expander(f"{icon} \"{check.claim_text}\" (entailment={check.entailment_score})"):
                st.markdown(f"**Verdict:** {check.verdict}")
                st.markdown(f"**Reasoning:** {check.reasoning}")
                for idx in check.supporting_chunk_indices:
                    chunk = result.used_chunks.get(idx)
                    if chunk:
                        st.markdown(f"**Source:** `{chunk['metadata']['source']}`")
                        st.text(chunk["text"])

elif run_button:
    st.warning("Please enter a question first.")

st.divider()
st.caption(
    "Pipeline: Retrieval → Adaptive Evidence Verification → Self-Correction Loop "
    "→ Generation with Citations → Post-hoc Hallucination Check"
)
