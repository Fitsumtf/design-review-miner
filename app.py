"""design-review-miner — web app.

Search past engineering lessons with hybrid retrieval (BM25 + semantic, RRF
fusion), and compare v1 TF-IDF against v2 hybrid side by side.

Run locally:  streamlit run app.py
"""

import streamlit as st

from design_review_miner.index import KnowledgeIndex
from design_review_miner.hybrid_index import HybridIndex

st.set_page_config(page_title="design-review-miner", page_icon="🔍", layout="wide")

st.title("🔍 design-review-miner")
st.caption(
    "AI-assisted retrieval of past engineering lessons — describe a new design "
    "issue and surface the most similar past issues with their root causes and "
    "resolutions. Hybrid engine: BM25 (lexical) + semantic embeddings, fused "
    "with Reciprocal Rank Fusion. Demo dataset is synthetic — no OEM data. "
    "[GitHub](https://github.com/Fitsumtf/design-review-miner)"
)

@st.cache_resource
def _load():
    hybrid = HybridIndex.from_csv("data/design_review_records.csv")
    return hybrid  # HybridIndex subclasses KnowledgeIndex, so it can do both

index = _load()

st.sidebar.header("Engine")
mode = st.sidebar.radio("Retrieval mode", [
    "Hybrid v2 (BM25 + semantic)",
    "TF-IDF v1 (baseline)",
    "⚖️ Compare v1 vs v2",
])
st.sidebar.metric("Records in knowledge base", len(index.records))
st.sidebar.write(
    "**Why hybrid?** Lexical matching catches exact part words and codes; the "
    "semantic channel bridges vocabulary mismatch (e.g. *bolts backing out* → "
    "*torque relaxation*). Every hybrid match reports which channel found it — "
    "auditability engineers can trust."
)

st.subheader("Describe a new design issue")
examples = [
    "Fasteners on skid plate corroding in coastal climate durability test",
    "Bolts backing out during shaker table runs",
    "Plastic clip on center console fails to snap in during assembly",
    "Laser weld penetration inconsistent on battery module busbar joints",
]
col1, col2 = st.columns([3, 1])
query = col1.text_input("Issue description",
                        placeholder=examples[0], label_visibility="collapsed")
if col2.button("🎲 Try an example"):
    import random
    st.session_state["q"] = random.choice(examples)
query = st.session_state.get("q", "") if not query else query
top_k = st.slider("Results", 1, 8, 3)

if query:
    st.markdown(f"**Query:** *{query}*")

    if mode.startswith("⚖️"):
        v1 = KnowledgeIndex.query(index, query, top_k=top_k)
        v2 = index.query(query, top_k=top_k)
        c1, c2 = st.columns(2)
        c1.subheader("v1 — TF-IDF")
        for m in v1:
            with c1.expander(f"{m.record_id} · {m.component} · sim {m.score:.2f}"):
                st.write(f"**Issue:** {m.description}")
                st.error(f"**Root cause:** {m.root_cause}")
                st.success(f"**Resolution:** {m.resolution}")
        c2.subheader("v2 — Hybrid (BM25 + semantic, RRF)")
        for m in v2:
            chan = ", ".join(
                f"{name} #{rank}" for name, rank in
                (("lexical", m.lexical_rank), ("semantic", m.semantic_rank))
                if rank is not None)
            with c2.expander(f"{m.record_id} · {m.component} · via {chan}"):
                st.write(f"**Issue:** {m.description}")
                st.error(f"**Root cause:** {m.root_cause}")
                st.success(f"**Resolution:** {m.resolution}")
        st.info(
            "Watch the ranking differences: on the corrosion example, v1 ranks "
            "the truly relevant record (DR-007) third; the hybrid promotes it "
            "to #1 because both channels agree on it."
        )
    else:
        if mode.startswith("Hybrid"):
            matches = index.query(query, top_k=top_k)
        else:
            matches = KnowledgeIndex.query(index, query, top_k=top_k)
        if not matches:
            st.info("No similar past issues found — try different wording.")
        for m in matches:
            header = f"{m.record_id} · {m.component} · {m.issue_type} ({m.severity})"
            if hasattr(m, "channels") and getattr(m, "channels", None):
                chan = ", ".join(
                    f"{name} #{rank}" for name, rank in
                    (("lexical", m.lexical_rank), ("semantic", m.semantic_rank))
                    if rank is not None)
                header += f" · via {chan}"
            with st.expander(header, expanded=True):
                st.write(f"**Past issue:** {m.description}")
                st.error(f"**Root cause:** {m.root_cause}")
                st.success(f"**Resolution:** {m.resolution}")
else:
    st.info("Type a new issue above — or click **Try an example**.")
