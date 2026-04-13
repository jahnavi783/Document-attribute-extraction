"""
Document Attribute Normalization System
Streamlit UI

Fixes applied:
  - Issue 4: UI threshold sliders are forwarded to engine.process()
  - Issue 7: Preview tab shows ONLY normalized output (Attribute + Value);
             match debug details moved to a separate "Match Details" tab
  - Issue 8: Download produces clean normalized file (canonical attrs + original values)
"""

import sys
import io
import json
import pandas as pd
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.normalization_engine import NormalizationEngine, NormalizationReport
from services.attribute_matcher import (
    MatchResult,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_SEMANTIC_THRESHOLD,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocNorm — Attribute Normalizer",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
    padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
  }
  .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
  .main-header p  { color: #c5cae9; margin: 0.3rem 0 0; font-size: 0.95rem; }

  .metric-card {
    background: white; border: 1px solid #e0e0e0; border-radius: 10px;
    padding: 1rem 1.5rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; }
  .metric-card .lbl { font-size: 0.82rem; color: #666; margin-top: 0.1rem; }

  .badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
           font-size: 0.75rem; font-weight: 600; }
  .badge-exact     { background:#e8f5e9; color:#2e7d32; }
  .badge-synonym   { background:#e3f2fd; color:#1565c0; }
  .badge-fuzzy     { background:#fff3e0; color:#e65100; }
  .badge-semantic  { background:#f3e5f5; color:#6a1b9a; }
  .badge-unmatched { background:#ffebee; color:#c62828; }

  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #1a237e;
    margin: 1rem 0 0.5rem; padding-bottom: 6px;
    border-bottom: 2px solid #e8eaf6;
  }
  div[data-testid="stFileUploader"] {
    border: 2px dashed #7986cb; border-radius: 10px;
    padding: 0.5rem; background: #f8f9ff;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MASTER_PATH = str(Path(__file__).parent / "data" / "master_attributes.json")
MATCH_LABELS = {
    "exact":     "✅ Exact",
    "synonym":   "🔵 Synonym",
    "fuzzy":     "🟠 Fuzzy",
    "semantic":  "🟣 Semantic",
    "unmatched": "❌ Unmatched",
}

@st.cache_resource
def get_engine():
    return NormalizationEngine(MASTER_PATH)

def badge_html(match_type: str) -> str:
    label = MATCH_LABELS.get(match_type, match_type)
    return f'<span class="badge badge-{match_type}">{label}</span>'

def conf_bar(confidence: float) -> str:
    pct = int(confidence * 100)
    color = "#4caf50" if pct >= 80 else "#ff9800" if pct >= 50 else "#f44336"
    return f"""
    <div style="background:#eee;border-radius:4px;height:8px;width:100%">
      <div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>
    </div>
    <small style="color:#666">{pct}%</small>"""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Issue 4: sliders define the thresholds that get passed to the engine
    fuzzy_threshold = st.slider(
        "Fuzzy Match Threshold", 50, 99, int(DEFAULT_FUZZY_THRESHOLD),
        help="Minimum RapidFuzz score (0–100) for fuzzy matching"
    )
    semantic_threshold = st.slider(
        "Semantic Match Threshold", 0.10, 0.90,
        float(DEFAULT_SEMANTIC_THRESHOLD), 0.05,
        help="Minimum cosine similarity (0–1) for semantic matching"
    )

    st.markdown("---")
    st.markdown("### 📚 Master Attributes")
    with open(MASTER_PATH) as f:
        master = json.load(f)
    st.info(f"{len(master['master_attributes'])} canonical attributes loaded")
    with st.expander("View Master List"):
        for entry in master["master_attributes"]:
            st.markdown(f"**{entry['canonical']}**")
            st.caption(", ".join(entry.get("variations", [])[:5]))

    st.markdown("---")
    st.markdown("### 🧪 Sample Files")
    if st.button("Generate Sample Files"):
        from src.sample_generator import (
            create_sample_pdf_kv, create_sample_excel_tabular, create_sample_csv,
        )
        st.download_button("📄 Sample PDF", create_sample_pdf_kv(),
                           "sample_workorder.pdf", "application/pdf")
        st.download_button("📊 Sample Excel", create_sample_excel_tabular(),
                           "sample_invoices.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("📋 Sample CSV", create_sample_csv(),
                           "sample_data.csv", "text/csv")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🔄 Document Attribute Normalization System</h1>
  <p>Upload PDF, Excel or CSV files — raw attribute names are mapped to canonical
     master attributes automatically.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📂 Upload Document</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a PDF, Excel (.xlsx) or CSV file",
    type=["pdf", "xlsx", "xls", "csv"],
    label_visibility="collapsed",
)

if uploaded:
    engine = get_engine()

    with st.spinner("🔍 Extracting attributes and normalizing…"):
        try:
            # Issue 4: pass slider values directly into the engine
            report: NormalizationReport = engine.process(
                io.BytesIO(uploaded.read()),
                uploaded.name,
                fuzzy_threshold=float(fuzzy_threshold),
                semantic_threshold=float(semantic_threshold),
            )
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            st.stop()

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Normalization Summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    match_counts: dict[str, int] = {}
    for r in report.match_details:
        match_counts[r.match_type] = match_counts.get(r.match_type, 0) + 1

    cards = [
        (c1, report.total_attributes, "Total Attributes", "#1a237e"),
        (c2, report.matched,          "Matched",          "#2e7d32"),
        (c3, report.unmatched,        "Unmatched",        "#c62828"),
        (c4, match_counts.get("exact", 0) + match_counts.get("synonym", 0),
             "Exact / Synonym", "#1565c0"),
        (c5, match_counts.get("fuzzy", 0) + match_counts.get("semantic", 0),
             "Fuzzy / Semantic", "#6a1b9a"),
    ]
    for col, val, lbl, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="val" style="color:{color}">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Deduplicate match results for display
    unique_results: list[MatchResult] = []
    seen: set[str] = set()
    for r in report.match_details:
        if r.raw_attr not in seen:
            seen.add(r.raw_attr)
            unique_results.append(r)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    # Issue 7: 4 tabs — Preview (clean output only), Match Details (debug),
    #          Analytics, Download
    tab_preview, tab_debug, tab_analytics, tab_download = st.tabs(
        ["📋 Normalized Preview", "🔍 Match Details", "📈 Analytics", "⬇️ Download"]
    )

    # ── Tab 1: Clean normalized preview ───────────────────────────────────────
    with tab_preview:
        st.markdown('<div class="section-title">Normalized Output Preview</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Final business output: only normalized attribute names and their original values."
        )

        preview_rows = [
            {"Output Attribute": rec.get("attribute", ""), "Value": rec.get("value", "")}
            for rec in report.normalized_records
            if str(rec.get("attribute", "")).strip() or str(rec.get("value", "")).strip()
        ]

        if preview_rows:
            preview_df = pd.DataFrame(preview_rows)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        else:
            st.info("No attributes found in the document.")

    # ── Tab 2: Match debug details (Issue 7) ──────────────────────────────────
    with tab_debug:
        st.markdown('<div class="section-title">Attribute Matching Details</div>',
                    unsafe_allow_html=True)
        st.caption("Technical matching diagnostics — match type, confidence, matched variation.")

        rows_html = ""
        for r in unique_results:
            badge = badge_html(r.match_type)
            bar   = conf_bar(r.confidence)
            var   = r.matched_variation or "—"
            rows_html += f"""
            <tr>
              <td style="padding:8px 10px;border-bottom:1px solid #eee">{r.raw_attr}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #eee;
                         font-weight:600;color:#1a237e">{r.canonical_attr}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #eee">{badge}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #eee;
                         min-width:120px">{bar}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #eee;
                         color:#666;font-size:0.82rem">{var}</td>
            </tr>"""

        st.markdown(f"""
        <div style="overflow-x:auto;border:1px solid #e0e0e0;border-radius:8px">
          <table style="width:100%;border-collapse:collapse;font-size:0.88rem">
            <thead style="background:#1a237e;color:white">
              <tr>
                <th style="padding:10px;text-align:left">Raw Attribute</th>
                <th style="padding:10px;text-align:left">Canonical Attribute</th>
                <th style="padding:10px;text-align:left">Match Type</th>
                <th style="padding:10px;text-align:left">Confidence</th>
                <th style="padding:10px;text-align:left">Matched Variation</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        unmatched_list = [r for r in unique_results if r.match_type == "unmatched"]
        if unmatched_list:
            st.markdown('<div class="section-title">⚠️ Unmatched Attributes</div>',
                        unsafe_allow_html=True)
            for r in unmatched_list:
                st.warning(
                    f"**{r.raw_attr}** — no confident match found. "
                    "Add to master file if needed."
                )

    # ── Tab 3: Analytics ───────────────────────────────────────────────────────
    with tab_analytics:
        st.markdown('<div class="section-title">Match Type Distribution</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            dist_df = pd.DataFrame([
                {"Match Type": t.title(), "Count": c}
                for t, c in match_counts.items()
            ]).sort_values("Count", ascending=False)
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        with col_b:
            if not dist_df.empty:
                st.bar_chart(dist_df.set_index("Match Type")["Count"])

        st.markdown('<div class="section-title">Confidence Score Distribution</div>',
                    unsafe_allow_html=True)
        conf_data = pd.DataFrame({
            "Attribute":  [r.raw_attr for r in unique_results],
            "Confidence": [round(r.confidence * 100, 1) for r in unique_results],
            "Match Type": [r.match_type for r in unique_results],
        })
        st.dataframe(conf_data, use_container_width=True, hide_index=True)

    # ── Tab 4: Download (Issue 8) ──────────────────────────────────────────────
    with tab_download:
        st.markdown('<div class="section-title">Download Normalized Output</div>',
                    unsafe_allow_html=True)
        st.success(
            f"✅ Output format: **{report.output_ext.upper()}** — "
            "same as input, canonical attributes, original values preserved."
        )

        ext_mime = {
            "pdf":  "application/pdf",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "csv":  "text/csv",
        }
        out_name = Path(uploaded.name).stem + "_normalized." + report.output_ext

        st.download_button(
            label=f"⬇️ Download {out_name}",
            data=report.output_bytes,   # Issue 8: only normalized attrs + values
            file_name=out_name,
            mime=ext_mime[report.output_ext],
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.info(
                f"**Input:** {uploaded.name}  \n"
                f"**Format:** {report.input_format.upper()}  \n"
                f"**Structure:** {report.doc_type}"
            )
        with col2:
            rate = (
                report.matched / report.total_attributes * 100
                if report.total_attributes else 0
            )
            st.info(
                f"**Attributes normalized:** {report.matched}/{report.total_attributes}  \n"
                f"**Match rate:** {rate:.0f}%  \n"
                f"**Output file:** {out_name}"
            )

else:
    # Landing page
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;background:#f8f9ff;
                border-radius:12px;border:2px dashed #7986cb">
      <div style="font-size:3rem">📁</div>
      <h3 style="color:#3949ab;margin:0.5rem 0">Upload a document to get started</h3>
      <p style="color:#666;max-width:500px;margin:0 auto">
        Supports <strong>PDF</strong>, <strong>Excel (.xlsx)</strong>,
        and <strong>CSV</strong> files.<br>
        Raw attribute names are mapped to canonical master attributes
        using exact, fuzzy, and semantic matching.
      </p>
    </div>

    <div style="display:flex;gap:1rem;margin-top:1.5rem">
      <div style="flex:1;background:white;border-radius:10px;padding:1.2rem;
                  border:1px solid #e0e0e0">
        <div style="font-size:1.5rem">✅</div><strong>Exact Match</strong>
        <p style="color:#666;font-size:0.85rem;margin-top:0.3rem">
          Direct or case-insensitive hit against canonical name or known synonyms
        </p>
      </div>
      <div style="flex:1;background:white;border-radius:10px;padding:1.2rem;
                  border:1px solid #e0e0e0">
        <div style="font-size:1.5rem">🔵</div><strong>Synonym Match</strong>
        <p style="color:#666;font-size:0.85rem;margin-top:0.3rem">
          High-confidence fuzzy match against registered variations (≥95)
        </p>
      </div>
      <div style="flex:1;background:white;border-radius:10px;padding:1.2rem;
                  border:1px solid #e0e0e0">
        <div style="font-size:1.5rem">🟠</div><strong>Fuzzy Match</strong>
        <p style="color:#666;font-size:0.85rem;margin-top:0.3rem">
          RapidFuzz token_sort_ratio; only accepted above the fuzzy threshold slider
        </p>
      </div>
      <div style="flex:1;background:white;border-radius:10px;padding:1.2rem;
                  border:1px solid #e0e0e0">
        <div style="font-size:1.5rem">🟣</div><strong>Semantic Match</strong>
        <p style="color:#666;font-size:0.85rem;margin-top:0.3rem">
          Embedding cosine similarity (Ollama nomic-embed-text or TF-IDF fallback);
          only accepted above the semantic threshold slider
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)
