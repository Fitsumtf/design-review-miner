# design-review-miner

**AI-assisted retrieval of past engineering lessons — so design review knowledge stops dying in folders.**

## The problem

In 20+ years of mechanical and manufacturing engineering (Tesla, Thermo Fisher Scientific), I saw the same failure pattern over and over: an issue gets caught on the line, root-caused, and fixed — and six months later a different program hits the *same* issue, because the lesson lived in one engineer's head or a buried spreadsheet.

This project is a working prototype of the fix: index historical design review and failure records with machine learning, and **automatically surface similar past issues** whenever a new problem is described.

## How it works

1. Historical records (component, issue type, description, root cause, resolution) are loaded from CSV
2. Text fields are vectorized with **TF-IDF (unigrams + bigrams)** using scikit-learn
3. A new issue description is vectorized and matched via **cosine similarity**
4. The top-k most similar past records are returned — with their root causes and resolutions

```
NEW ISSUE: Fasteners on skid plate corroding in coastal climate durability test
------------------------------------------------------------------------
[DR-007] similarity=0.14 | 12V battery tray | Corrosion (Medium)
    Issue      : Corrosion found on battery tray fasteners after salt spray test
    Root cause : Dissimilar metal contact between zinc-plated bolt and aluminum tray
    Resolution : Changed to stainless fastener with isolation washer
```

A brand-new skid plate problem just inherited the lesson learned on a battery tray a year earlier. That is the entire point.

## Hybrid retrieval (v2) — BM25 + semantic fusion

`design_review_miner/hybrid_index.py` upgrades retrieval from single-method TF-IDF to a **hybrid engine**: Okapi BM25 (lexical — exact part numbers and error codes) runs in parallel with a semantic embedding channel (vocabulary mismatch — "joint loosening under vibration" finds "torque relaxation"), fused via **Reciprocal Rank Fusion (RRF)**. Every match reports *which channel found it and at what rank*, preserving the auditability engineers need.

```python
from design_review_miner.hybrid_index import HybridIndex

index = HybridIndex.from_csv("data/design_review_records.csv")
print(index.review("Bolts backing out during shaker table runs"))
print(index.compare_to_v1("Bolts backing out during shaker table runs"))  # A/B vs v1
```

On the README example above, v1 TF-IDF ranks the truly relevant corrosion record (DR-007) *third*; the hybrid promotes it to **#1** because both channels agree on it. The embedding backend is pluggable: sentence-transformers when available, an offline LSA fallback (works air-gapped), or any callable for API embedders.

## Quick start

```bash
pip install -r requirements.txt
python examples/demo.py
```

Or in your own code:

```python
from design_review_miner import KnowledgeIndex

index = KnowledgeIndex.from_csv("data/design_review_records.csv")
print(index.review("Laser weld penetration inconsistent on busbar joints"))
```

The included dataset (`data/design_review_records.csv`) contains 15 realistic automotive/EV design review records — welds, tolerancing, DFM, corrosion, mechanisms — written from manufacturing floor experience.

## Design notes & roadmap

- **TF-IDF is the deliberate v1 choice**: transparent, fast, zero external dependencies, and auditable — an engineer can see exactly why a match scored what it did. Its known limit is vocabulary mismatch (it matches words, not meaning).
- **v2: semantic embeddings** (sentence-transformers) to match *meaning* — "joint loosening under vibration" should find "torque relaxation" records even with zero shared words.
- **v3: LLM summarization layer** — generate a "lessons to check" briefing for a new design review from the retrieved records.
- Structured export to `.docx` via my companion package [`pfmea-doc-gen`](https://github.com/Fitsumtf/pfmea-doc-gen).

## Project structure

```
design-review-miner/
├── design_review_miner/
│   ├── __init__.py
│   └── index.py                 # KnowledgeIndex: TF-IDF + cosine retrieval
├── data/
│   └── design_review_records.csv
├── examples/
│   └── demo.py
├── requirements.txt
└── README.md
```

## License

MIT

---

*Built by [Dr. Fitsum Taye Feyissa](https://github.com/Fitsumtf) — manufacturing & process engineer (Tesla, Thermo Fisher Scientific) pursuing an M.Sc. in Applied Data Science. This project sits exactly where I work best: the intersection of engineering knowledge and AI.*
