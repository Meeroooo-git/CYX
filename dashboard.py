import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberEx Recommender",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { color: #8b9ab5; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #e2e8f0; font-size: 1.5rem; font-weight: 700; margin-top: 0.2rem; }
    .tag-pill {
        display: inline-block;
        background: #1e3a5f;
        color: #7ec8e3;
        border: 1px solid #2563a8;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin: 2px 3px 2px 0;
    }
    .tag-pill-tactic {
        background: #1e3d2f;
        color: #6ee7b7;
        border: 1px solid #065f46;
    }
    .tag-pill-threat {
        background: #3d1e1e;
        color: #fca5a5;
        border: 1px solid #991b1b;
    }
    .score-bar-wrap { margin: 0.3rem 0; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #93c5fd;
        border-bottom: 1px solid #2d3250;
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .why-box {
        background: #151c2c;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }
    .org-card {
        background: #1a1f35;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.6rem;
    }
    .feedback-saved {
        background: #052e16;
        border: 1px solid #166534;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #86efac;
        font-size: 0.85rem;
    }
    div[data-testid="stSelectbox"] label { color: #93c5fd !important; font-weight: 600; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    recs  = pd.read_csv(os.path.join(base, "phase2_top10_recommendations.csv"))
    exs   = pd.read_csv(os.path.join(base, "exercises_full.csv"))
    orgs  = pd.read_csv(os.path.join(base, "orgs_full.csv"))

    # Merge recommendations with exercise details
    merged = recs.merge(exs, on="EXID", how="left")
    return recs, exs, orgs, merged

recs, exs, orgs, merged = load_data()

# ── Helper: parse semicolon-separated tag columns ─────────────────────────────
def parse_tags(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [t.strip() for t in str(val).split(";") if t.strip()]

def tag_pills(tags, css_class="tag-pill"):
    return " ".join([f'<span class="{css_class}">{t}</span>' for t in tags])

def score_pct(val, max_val=1.0):
    return min(100, round((val / max_val) * 100, 1))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CyberEx Recommender")
    st.markdown("*Phase 3 Dashboard — COS70008*")
    st.markdown("---")

    org_ids = sorted(merged["ORGID"].unique().tolist())
    selected_org = st.selectbox("Select Organisation", org_ids, format_func=lambda x: f"Org {x:03d}")

    st.markdown("---")
    org_info = orgs[orgs["ORGID"] == selected_org].iloc[0] if selected_org in orgs["ORGID"].values else None

    if org_info is not None:
        st.markdown("**Organisation Profile**")
        st.markdown(f"""
        <div class="org-card">
            <div class="metric-label">Industry</div>
            <div style="color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem">{org_info.get('Industry','—')}</div>
            <div class="metric-label">Region</div>
            <div style="color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem">{org_info.get('Region','—')}</div>
            <div class="metric-label">Size</div>
            <div style="color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem">{org_info.get('Size','—')}</div>
            <div class="metric-label">Maturity Level</div>
            <div style="color:#e2e8f0;font-size:0.9rem;margin-bottom:0.5rem">{org_info.get('Maturity','—')} / 5</div>
            <div class="metric-label">Primary Threat</div>
            <div style="color:#fca5a5;font-size:0.85rem">{str(org_info.get('Threats','—')).split(';')[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<span style='color:#8b9ab5;font-size:0.75rem'>Model: SVD Hybrid | α = 0.9<br>150 orgs · 100 exercises · 3,180 ratings</span>", unsafe_allow_html=True)

# ── Filter data for selected org ──────────────────────────────────────────────
org_recs = merged[merged["ORGID"] == selected_org].sort_values("Rank").reset_index(drop=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"## 🛡️ Exercise Recommendations — Org {selected_org:03d}")

if org_info is not None:
    threats = parse_tags(org_info.get("Threats", ""))
    st.markdown(
        f"**{org_info.get('Industry','—')}** &nbsp;·&nbsp; {org_info.get('Region','—')} &nbsp;·&nbsp; {org_info.get('Size','—')} &nbsp;·&nbsp; "
        + tag_pills(threats[:3], "tag-pill-threat"),
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Top metrics row ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

avg_hybrid = org_recs["Hybrid_Score"].mean()
avg_cf     = org_recs["CF_Predicted_Rating"].mean()
tactics_covered = set()
for _, row in org_recs.iterrows():
    tactics_covered.update(parse_tags(row.get("ExTactics", "")))

with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Recommendations</div>
        <div class="metric-value">{len(org_recs)}</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg Hybrid Score</div>
        <div class="metric-value">{avg_hybrid:.3f}</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg CF Rating</div>
        <div class="metric-value">{avg_cf:.2f} / 5</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">ATT&CK Tactics Covered</div>
        <div class="metric-value">{len(tactics_covered)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ── PANEL 1 + PANEL 2 side by side ───────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-header">📋 Top 10 Recommended Exercises</div>', unsafe_allow_html=True)

    for _, row in org_recs.iterrows():
        threat_tags = parse_tags(row.get("ExThreat", ""))
        tactic_tags = parse_tags(row.get("ExTactics", ""))
        hybrid = row["Hybrid_Score"]
        cf     = row["CF_Predicted_Rating"]
        rank   = int(row["Rank"])

        rank_color = "#f59e0b" if rank == 1 else "#6b7280"

        with st.expander(f"#{rank}  Ex {int(row['EXID']):02d}  ·  {str(row.get('ExThreat',''))[:45]}  ·  Hybrid: {hybrid:.3f}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Threat Type**")
                st.markdown(tag_pills(threat_tags, "tag-pill-threat"), unsafe_allow_html=True)
                st.markdown(f"**ATT&CK Tactics**")
                st.markdown(tag_pills(tactic_tags[:5], "tag-pill-tactic"), unsafe_allow_html=True)

                groups = parse_tags(row.get("ExGroups", ""))
                if groups:
                    st.markdown(f"**Adversary Groups**")
                    st.markdown(tag_pills(groups[:4]), unsafe_allow_html=True)

            with c2:
                st.markdown("**Scores**")
                st.progress(min(hybrid, 1.0), text=f"Hybrid: {hybrid:.3f}")
                st.progress(min(cf / 5, 1.0), text=f"CF Rating: {cf:.2f}/5")
                content = row["Content_Score"]
                st.progress(min(content, 1.0), text=f"Content: {content:.3f}")

                complexity = row.get("ExComplexity", "—")
                maturity   = row.get("ExMaturity", "—")
                length     = row.get("ExLength", "—")
                st.markdown(f"**Complexity:** {complexity}/5 &nbsp; **Length:** {length} min", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">🗺️ ATT&CK Tactic Coverage</div>', unsafe_allow_html=True)

    tactic_counts = {}
    for _, row in org_recs.iterrows():
        for t in parse_tags(row.get("ExTactics", "")):
            tactic_counts[t] = tactic_counts.get(t, 0) + 1

    if tactic_counts:
        tactic_df = pd.DataFrame(
            sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True),
            columns=["Tactic", "Count"]
        )
        fig_tactic = px.bar(
            tactic_df, x="Count", y="Tactic", orientation="h",
            color="Count",
            color_continuous_scale=["#1e3a5f", "#3b82f6", "#93c5fd"],
            labels={"Count": "Exercises", "Tactic": ""},
            height=380
        )
        fig_tactic.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font=dict(color="#cbd5e1", size=11),
            coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=10, b=30),
            xaxis=dict(gridcolor="#1e2130", tickfont=dict(color="#8b9ab5")),
            yaxis=dict(tickfont=dict(color="#cbd5e1"))
        )
        st.plotly_chart(fig_tactic, use_container_width=True)

        # Threat type donut
        st.markdown('<div class="section-header">🎯 Threat Type Distribution</div>', unsafe_allow_html=True)
        threat_counts = {}
        for _, row in org_recs.iterrows():
            for t in parse_tags(row.get("ExThreat", "")):
                threat_counts[t] = threat_counts.get(t, 0) + 1

        if threat_counts:
            threat_df = pd.DataFrame(threat_counts.items(), columns=["Threat", "Count"])
            fig_threat = px.pie(
                threat_df, names="Threat", values="Count",
                hole=0.55, height=280,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_threat.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font=dict(color="#cbd5e1", size=10),
                margin=dict(l=0, r=0, t=10, b=10),
                legend=dict(font=dict(size=9), bgcolor="#0f1117")
            )
            fig_threat.update_traces(textfont_color="#e2e8f0")
            st.plotly_chart(fig_threat, use_container_width=True)

st.markdown("---")

# ── PANEL 3: Why this exercise? ───────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Why Was This Recommended?</div>', unsafe_allow_html=True)

ex_options = [f"#{int(r['Rank'])}  Ex {int(r['EXID']):02d}  — {str(r.get('ExThreat',''))[:50]}" for _, r in org_recs.iterrows()]
selected_ex_label = st.selectbox("Pick an exercise to explain", ex_options)
selected_rank = int(selected_ex_label.split("#")[1].split(" ")[0])
selected_row  = org_recs[org_recs["Rank"] == selected_rank].iloc[0]

ex1, ex2, ex3 = st.columns([2, 1.5, 1.5])

with ex1:
    hybrid = selected_row["Hybrid_Score"]
    cf     = selected_row["CF_Predicted_Rating"]
    content= selected_row["Content_Score"]

    # Plain English explanation
    cf_pct      = round((cf / 5) * 100)
    content_pct = round(content * 100)
    threat_tags = parse_tags(selected_row.get("ExThreat", ""))
    tactic_tags = parse_tags(selected_row.get("ExTactics", ""))
    technique_tags = parse_tags(selected_row.get("ExTechniqueIDs", ""))

    org_threat = str(org_info.get("Threats", "")) if org_info is not None else ""
    shared_threats = [t for t in threat_tags if t.lower() in org_threat.lower()]

    if cf_pct >= 60:
        cf_sentence = f"Organisations with a similar threat profile to Org {selected_org} rated this exercise highly, giving it a predicted rating of {cf:.2f} out of 5."
    else:
        cf_sentence = f"This exercise has a moderate collaborative filtering score ({cf:.2f}/5), meaning similar organisations have found it somewhat useful."

    if content_pct >= 15:
        content_sentence = f"It also shares content features with exercises this organisation is already familiar with (content similarity: {content:.3f})."
    else:
        content_sentence = f"The recommendation is driven primarily by collaborative signals rather than content similarity ({content:.3f})."

    if shared_threats:
        threat_sentence = f"The exercise directly addresses {', '.join(shared_threats)}, which matches this organisation's known threat profile."
    elif threat_tags:
        threat_sentence = f"It covers {threat_tags[0]} scenarios, which may help broaden this organisation's training coverage."
    else:
        threat_sentence = ""

    st.markdown(f"""
    <div class="why-box">
        <strong>Why Exercise {int(selected_row['EXID'])} was recommended:</strong><br><br>
        {cf_sentence}<br><br>
        {content_sentence}<br><br>
        {threat_sentence}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**ATT&CK Technique IDs**")
    st.markdown(tag_pills(technique_tags[:8]), unsafe_allow_html=True)

with ex2:
    st.markdown("**Score Breakdown**")
    score_fig = go.Figure(go.Bar(
        x=["CF Rating\n(÷5)", "Content\nScore", "Hybrid\nScore"],
        y=[cf / 5, content, hybrid],
        marker_color=["#3b82f6", "#10b981", "#f59e0b"],
        text=[f"{cf/5:.2f}", f"{content:.3f}", f"{hybrid:.3f}"],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11)
    ))
    score_fig.update_layout(
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#cbd5e1", size=10),
        yaxis=dict(range=[0, 1.1], gridcolor="#1e2130", tickfont=dict(color="#8b9ab5")),
        xaxis=dict(tickfont=dict(color="#cbd5e1")),
        margin=dict(l=0, r=0, t=30, b=0),
        height=240,
        showlegend=False
    )
    st.plotly_chart(score_fig, use_container_width=True)

with ex3:
    st.markdown("**Exercise Details**")
    complexity = selected_row.get("ExComplexity", "—")
    maturity   = selected_row.get("ExMaturity", "—")
    length     = selected_row.get("ExLength", "—")
    audience   = parse_tags(selected_row.get("ExAudience", ""))
    platforms  = parse_tags(selected_row.get("ExPlatforms", ""))

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Complexity</div>
        <div class="metric-value" style="font-size:1.1rem">{complexity} / 5</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Duration</div>
        <div class="metric-value" style="font-size:1.1rem">{length} min</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Maturity Required</div>
        <div class="metric-value" style="font-size:1.1rem">{maturity} / 5</div>
    </div>
    """, unsafe_allow_html=True)

    if audience:
        st.markdown("**Audience**")
        st.markdown(tag_pills(audience), unsafe_allow_html=True)
    if platforms:
        st.markdown("**Platforms**")
        st.markdown(tag_pills(platforms[:4]), unsafe_allow_html=True)

st.markdown("---")

# ── PANEL 4: Feedback ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💬 Leave Feedback</div>', unsafe_allow_html=True)
st.markdown("Rate how useful a recommendation was. This helps improve future suggestions.")

fb1, fb2 = st.columns([1, 2])

with fb1:
    fb_ex_label = st.selectbox("Exercise to rate", ex_options, key="fb_ex")
    fb_rank     = int(fb_ex_label.split("#")[1].split(" ")[0])
    fb_row      = org_recs[org_recs["Rank"] == fb_rank].iloc[0]
    fb_rating   = st.slider("How useful was this recommendation?", 1, 5, 3,
                            format="%d ⭐", key="fb_rating")
    rating_labels = {1: "Not useful", 2: "Slightly useful", 3: "Neutral", 4: "Useful", 5: "Very useful"}
    st.caption(rating_labels.get(fb_rating, ""))

with fb2:
    fb_comment = st.text_area("Any comments? (optional)", placeholder="e.g. This exercise was too basic for our team, or it matched a gap we had been ignoring...", height=100)
    submit_fb  = st.button("Submit Feedback", type="primary")

if submit_fb:
    feedback_path = os.path.join(os.path.dirname(__file__), "feedback.csv")
    new_row = pd.DataFrame([{
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ORGID":       selected_org,
        "EXID":        int(fb_row["EXID"]),
        "rank":        fb_rank,
        "rating":      fb_rating,
        "hybrid_score":float(fb_row["Hybrid_Score"]),
        "comment":     fb_comment.strip()
    }])
    if os.path.exists(feedback_path):
        existing = pd.read_csv(feedback_path)
        updated  = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv(feedback_path, index=False)
    st.markdown(f"""
    <div class="feedback-saved">
        ✅ Feedback saved — Org {selected_org}, Exercise {int(fb_row['EXID'])}, {fb_rating} star{'s' if fb_rating != 1 else ''}.
        This will be used to retrain the model in future iterations.
    </div>""", unsafe_allow_html=True)

# Show existing feedback summary if file exists
feedback_path = os.path.join(os.path.dirname(__file__), "feedback.csv")
if os.path.exists(feedback_path):
    fb_df = pd.read_csv(feedback_path)
    if not fb_df.empty:
        st.markdown("")
        with st.expander(f"📊 Feedback collected so far ({len(fb_df)} entries)"):
            st.dataframe(fb_df, use_container_width=True)
            avg_by_ex = fb_df.groupby("EXID")["rating"].mean().reset_index()
            avg_by_ex.columns = ["EXID", "Avg Rating"]
            fig_fb = px.bar(avg_by_ex, x="EXID", y="Avg Rating",
                            color="Avg Rating",
                            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                            range_color=[1, 5], height=220,
                            labels={"EXID": "Exercise ID", "Avg Rating": "Avg ⭐"})
            fig_fb.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font=dict(color="#cbd5e1", size=10),
                margin=dict(l=0, r=0, t=20, b=0),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_fb, use_container_width=True)

st.markdown("---")
st.markdown("<span style='color:#4b5563;font-size:0.75rem'>COS70008 · Phase 3 Dashboard · Ameera Shahid Khan · 106197762</span>", unsafe_allow_html=True)
