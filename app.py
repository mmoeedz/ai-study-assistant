"""
AI-Powered Study Assistant — Streamlit Application

A modern, polished UI for the RAG-based study assistant.
Run with:  streamlit run app.py
"""

import streamlit as st
from rag_pipeline import StudyAssistant

# ── Page Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ─────────────────────────────────────────────────────────
    🎓 ACADEMIC THEME — "The Study"
    Palette: deep navy library, parchment, antique gold
   ───────────────────────────────────────────────────────── */
:root {
    --ink-900: #0b1220;
    --ink-800: #0f1a2e;
    --ink-700: #16243d;
    --ink-600: #1e3050;
    --parchment: #f5ecd7;
    --parchment-soft: #ece2c8;
    --gold-500: #d4a84b;
    --gold-400: #e6bf6a;
    --gold-300: #f1d490;
    --rose-300: #c97064;     /* burgundy accent */
    --emerald-400: #6cb98a;  /* green leather */
    --text-soft: #cbd5e1;
    --text-mute: #8ea0bb;
}

/* ── Global ─────────────────────────────────────────────── */
*, html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}

h1, h2, h3, .serif {
    font-family: 'Cormorant Garamond', 'Georgia', serif;
    letter-spacing: -0.005em;
}

/* App background — leather-bound desk + subtle paper texture */
.stApp {
    background:
      radial-gradient(1200px 600px at 10% -10%, rgba(212, 168, 75, 0.08), transparent 60%),
      radial-gradient(1000px 700px at 110% 10%, rgba(108, 185, 138, 0.06), transparent 60%),
      linear-gradient(160deg, #0a1322 0%, #0f1a2e 50%, #0a1322 100%);
    background-attachment: fixed;
}

/* Faint paper-grain overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
      url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.85  0 0 0 0 0.78  0 0 0 0 0.55  0 0 0 0.04 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
    opacity: 0.5;
    z-index: 0;
    mix-blend-mode: overlay;
}

/* ── Sidebar — looks like a leather book cover ──────────── */
section[data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, #0c1424 0%, #0a1120 100%);
    border-right: 1px solid rgba(212, 168, 75, 0.18);
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35);
}

section[data-testid="stSidebar"]::after {
    content: "";
    position: absolute;
    top: 0; bottom: 0; right: 0;
    width: 3px;
    background: linear-gradient(180deg, transparent, var(--gold-500), transparent);
    opacity: 0.55;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h4 {
    color: var(--parchment);
    font-family: 'Cormorant Garamond', serif;
    font-weight: 600;
    letter-spacing: 0.02em;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown code {
    color: var(--text-soft);
}

section[data-testid="stSidebar"] code {
    background: rgba(212, 168, 75, 0.08) !important;
    color: var(--gold-300) !important;
    border: 1px solid rgba(212, 168, 75, 0.2);
    border-radius: 4px;
    padding: 1px 6px;
}

/* ── Header — engraved book title ───────────────────────── */
.main-header {
    text-align: center;
    padding: 1.4rem 1rem 0.6rem;
    margin-bottom: 0.6rem;
    position: relative;
    animation: fadeIn 0.8s ease-out;
}

.main-header .crest {
    font-size: 2rem;
    color: var(--gold-400);
    text-shadow: 0 0 24px rgba(212, 168, 75, 0.45);
    animation: glow 4s ease-in-out infinite;
    display: inline-block;
}

@keyframes glow {
    0%, 100% { text-shadow: 0 0 18px rgba(212, 168, 75, 0.35); }
    50%      { text-shadow: 0 0 32px rgba(212, 168, 75, 0.7); }
}

.main-header h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--parchment);
    margin: 0.2rem 0 0.1rem;
    letter-spacing: 0.01em;
    text-shadow: 0 2px 18px rgba(0, 0, 0, 0.4);
}

.main-header h1 .accent {
    color: var(--gold-400);
    font-style: italic;
}

.main-header p {
    color: var(--text-mute);
    font-size: 1rem;
    font-weight: 400;
    font-style: italic;
    letter-spacing: 0.04em;
    margin-top: 0.2rem;
}

/* Decorative gold rule under the header */
.gold-rule {
    border: 0;
    height: 1px;
    width: 60%;
    margin: 0.6rem auto 1.2rem;
    background: linear-gradient(90deg, transparent, var(--gold-500), transparent);
    opacity: 0.7;
    position: relative;
    animation: ruleIn 1.1s ease-out;
}
.gold-rule::after {
    content: "✦";
    position: absolute;
    top: -10px; left: 50%;
    transform: translateX(-50%);
    color: var(--gold-400);
    background: var(--ink-900);
    padding: 0 10px;
    font-size: 0.8rem;
}

@keyframes ruleIn {
    from { width: 0; opacity: 0; }
    to   { width: 60%; opacity: 0.7; }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Mode selector — bookmark tabs ──────────────────────── */
div[data-testid="stRadio"] > div {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    justify-content: center;
    padding: 0.4rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(212, 168, 75, 0.15);
    border-radius: 10px;
    backdrop-filter: blur(6px);
}

div[data-testid="stRadio"] > div > label {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(212, 168, 75, 0.18);
    border-radius: 6px;
    padding: 0.55rem 1.3rem;
    color: var(--text-soft);
    font-weight: 500;
    font-size: 0.92rem;
    letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(.2,.8,.2,1);
    cursor: pointer;
    position: relative;
}

div[data-testid="stRadio"] > div > label:hover {
    background: rgba(212, 168, 75, 0.07);
    border-color: rgba(212, 168, 75, 0.45);
    color: var(--parchment);
    transform: translateY(-1px);
}

div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: linear-gradient(180deg, rgba(212, 168, 75, 0.18), rgba(212, 168, 75, 0.08));
    border-color: var(--gold-400);
    color: var(--parchment);
    box-shadow:
      0 0 0 1px rgba(212, 168, 75, 0.3) inset,
      0 6px 18px rgba(212, 168, 75, 0.18);
}

div[data-testid="stRadio"] > div > label:has(input:checked)::after {
    content: "";
    position: absolute;
    left: 16%; right: 16%;
    bottom: -1px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold-400), transparent);
    border-radius: 2px;
}

/* ── Response card — parchment page ─────────────────────── */
.response-card {
    background:
      linear-gradient(180deg, rgba(245, 236, 215, 0.97), rgba(236, 226, 200, 0.97));
    color: #2b2418;
    border: 1px solid rgba(212, 168, 75, 0.55);
    border-radius: 6px;
    padding: 1.8rem 2rem;
    margin: 1rem 0;
    animation: fadeSlideIn 0.55s cubic-bezier(.2,.8,.2,1);
    box-shadow:
      0 14px 40px rgba(0, 0, 0, 0.45),
      0 1px 0 rgba(255, 255, 255, 0.4) inset;
    position: relative;
    overflow: hidden;
}

.response-card::before {
    content: "";
    position: absolute;
    inset: 6px;
    border: 1px solid rgba(155, 117, 50, 0.25);
    border-radius: 3px;
    pointer-events: none;
}

.response-card::after {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--gold-500), var(--gold-300), var(--gold-500));
    opacity: 0.8;
}

.response-card h1, .response-card h2, .response-card h3,
.response-card h4 {
    color: #2b2418 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    margin-top: 0.8rem;
}

.response-card h3 {
    border-bottom: 1px solid rgba(155, 117, 50, 0.25);
    padding-bottom: 0.3rem;
    font-size: 1.25rem;
}

.response-card p, .response-card li {
    font-family: 'Inter', sans-serif !important;
    color: #3a311e !important;
    line-height: 1.65;
}

.response-card strong {
    color: #1c1810 !important;
}

.response-card code {
    background: rgba(155, 117, 50, 0.12) !important;
    color: #5b3c12 !important;
    border-radius: 3px;
    padding: 1px 6px;
}

.response-card hr {
    border-color: rgba(155, 117, 50, 0.25) !important;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(14px) scale(0.99); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ── Source chips — index cards ─────────────────────────── */
.source-chip {
    display: inline-block;
    background: rgba(212, 168, 75, 0.1);
    border: 1px solid rgba(212, 168, 75, 0.35);
    border-radius: 4px;
    padding: 0.32rem 0.7rem;
    margin: 0.22rem;
    font-size: 0.8rem;
    color: var(--gold-300);
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.02em;
    transition: all 0.25s ease;
}

.source-chip:hover {
    background: rgba(212, 168, 75, 0.18);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(212, 168, 75, 0.18);
}

/* ── Stat cards — embossed plates ───────────────────────── */
.status-card {
    background: linear-gradient(180deg, rgba(212, 168, 75, 0.08), rgba(212, 168, 75, 0.02));
    border: 1px solid rgba(212, 168, 75, 0.25);
    border-radius: 6px;
    padding: 0.9rem 0.6rem;
    margin: 0.5rem 0;
    text-align: center;
    box-shadow: 0 0 0 1px rgba(212, 168, 75, 0.06) inset;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.status-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(212, 168, 75, 0.18);
}

.status-card .number {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--gold-300);
    line-height: 1;
}

.status-card .label {
    color: var(--text-mute);
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Chat input ─────────────────────────────────────────── */
.stChatInput > div {
    border-radius: 6px !important;
    border: 1px solid rgba(212, 168, 75, 0.3) !important;
    background: rgba(245, 236, 215, 0.04) !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3) !important;
}

.stChatInput textarea {
    color: var(--parchment) !important;
    font-family: 'Inter', sans-serif !important;
}

.stChatInput textarea::placeholder {
    color: rgba(245, 236, 215, 0.45) !important;
    font-style: italic;
}

/* ── File uploader ──────────────────────────────────────── */
section[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(212, 168, 75, 0.35);
    border-radius: 6px;
    padding: 0.5rem;
    background: rgba(212, 168, 75, 0.025);
    transition: all 0.3s ease;
}

section[data-testid="stFileUploader"]:hover {
    border-color: var(--gold-400);
    background: rgba(212, 168, 75, 0.06);
}

section[data-testid="stFileUploader"] button {
    background: rgba(212, 168, 75, 0.15) !important;
    color: var(--gold-300) !important;
    border: 1px solid rgba(212, 168, 75, 0.4) !important;
}

/* ── Buttons — gold-leaf brass plates ───────────────────── */
.stButton > button {
    background: linear-gradient(180deg, #d4a84b, #b88a2d);
    color: #1c1408;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-bottom: 1px solid rgba(0, 0, 0, 0.3);
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.6rem 1.5rem;
    transition: all 0.25s cubic-bezier(.2,.8,.2,1);
    width: 100%;
    box-shadow:
      0 4px 14px rgba(212, 168, 75, 0.25),
      0 1px 0 rgba(255, 255, 255, 0.35) inset;
}

.stButton > button:hover {
    background: linear-gradient(180deg, #e6bf6a, #c99634);
    transform: translateY(-1px);
    box-shadow:
      0 8px 22px rgba(212, 168, 75, 0.4),
      0 1px 0 rgba(255, 255, 255, 0.5) inset;
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow:
      0 2px 6px rgba(212, 168, 75, 0.25),
      0 1px 2px rgba(0,0,0,0.25) inset;
}

/* ── Progress bar — gold ink ────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--gold-500), var(--gold-300)) !important;
}

/* ── Spinner ────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--gold-400) !important;
    border-right-color: rgba(212, 168, 75, 0.25) !important;
    border-bottom-color: rgba(212, 168, 75, 0.25) !important;
    border-left-color: rgba(212, 168, 75, 0.25) !important;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.15); }
::-webkit-scrollbar-thumb {
    background: rgba(212, 168, 75, 0.35);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(212, 168, 75, 0.6);
}

/* ── Expander ───────────────────────────────────────────── */
[data-testid="stExpander"] details {
    background: rgba(212, 168, 75, 0.04) !important;
    border: 1px solid rgba(212, 168, 75, 0.2) !important;
    border-radius: 6px !important;
    margin-top: 0.5rem;
}

[data-testid="stExpander"] summary {
    color: var(--gold-300) !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.02em;
}

[data-testid="stExpander"] summary:hover {
    color: var(--gold-400) !important;
}

/* ── Chat messages ──────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
}

/* User chat bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(108, 185, 138, 0.08) !important;
    border-left: 3px solid var(--emerald-400) !important;
    border-radius: 4px !important;
    padding: 0.7rem 1rem !important;
    color: var(--text-soft);
    animation: slideRight 0.4s ease-out;
}

@keyframes slideRight {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ── Divider ────────────────────────────────────────────── */
hr {
    border: 0 !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(212, 168, 75, 0.35), transparent) !important;
    margin: 1rem 0 !important;
}

/* ── Welcome card — open book ───────────────────────────── */
.welcome-card {
    background:
      linear-gradient(180deg, rgba(245, 236, 215, 0.96), rgba(236, 226, 200, 0.96));
    border: 1px solid rgba(155, 117, 50, 0.4);
    border-radius: 6px;
    padding: 2.4rem 2.6rem;
    text-align: center;
    margin: 1.5rem auto 2rem;
    max-width: 760px;
    color: #2b2418;
    position: relative;
    box-shadow:
      0 18px 45px rgba(0, 0, 0, 0.5),
      0 1px 0 rgba(255, 255, 255, 0.4) inset;
    animation: pageOpen 0.7s cubic-bezier(.2,.8,.2,1);
}

@keyframes pageOpen {
    from { opacity: 0; transform: rotateX(-12deg) translateY(-10px); }
    to   { opacity: 1; transform: rotateX(0) translateY(0); }
}

.welcome-card::before {
    content: "";
    position: absolute;
    inset: 8px;
    border: 1px solid rgba(155, 117, 50, 0.25);
    border-radius: 3px;
    pointer-events: none;
}

.welcome-card .emblem {
    font-size: 2.2rem;
    color: var(--gold-500);
    margin-bottom: 0.4rem;
    display: inline-block;
    animation: floaty 5s ease-in-out infinite;
}

@keyframes floaty {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-4px); }
}

.welcome-card h3 {
    color: #2b2418 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600;
    font-size: 1.7rem;
    margin: 0.2rem 0 0.4rem;
}

.welcome-card .subtitle {
    color: #6b5a35;
    font-size: 0.95rem;
    font-style: italic;
    margin-bottom: 1rem;
}

.welcome-card p {
    color: #4a3f24;
    font-size: 0.95rem;
    line-height: 1.6;
}

.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
    margin-top: 1.4rem;
}

.feature-item {
    background: rgba(155, 117, 50, 0.05);
    border: 1px solid rgba(155, 117, 50, 0.22);
    border-radius: 5px;
    padding: 1rem;
    text-align: left;
    transition: all 0.3s cubic-bezier(.2,.8,.2,1);
    cursor: default;
}

.feature-item:hover {
    transform: translateY(-3px);
    border-color: rgba(155, 117, 50, 0.55);
    background: rgba(155, 117, 50, 0.1);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
}

.feature-item .emoji {
    font-size: 1.4rem;
    margin-bottom: 0.4rem;
}

.feature-item .title {
    color: #2b2418;
    font-family: 'Cormorant Garamond', serif;
    font-weight: 600;
    font-size: 1.05rem;
}

.feature-item .desc {
    color: #6b5a35;
    font-size: 0.83rem;
    margin-top: 0.2rem;
    line-height: 1.4;
}

/* ── Sidebar bullet list ────────────────────────────────── */
.indexed-list {
    list-style: none;
    padding: 0; margin: 0.4rem 0 0;
}
.indexed-list li {
    color: var(--text-soft) !important;
    font-size: 0.84rem;
    padding: 0.32rem 0.5rem;
    border-left: 2px solid rgba(212, 168, 75, 0.4);
    margin: 0.3rem 0;
    background: rgba(212, 168, 75, 0.04);
    border-radius: 0 4px 4px 0;
    animation: slideRight 0.4s ease-out;
}

/* ── Hide Streamlit default chrome (carefully) ──────────── */
/* Hide hamburger menu + footer, but KEEP the header so the
   sidebar-toggle button remains visible on mobile.            */
#MainMenu, footer { visibility: hidden; }

/* Make the header transparent so it blends with the app
   background, and keep its sidebar-toggle button visible.     */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem;
}
header[data-testid="stHeader"] [data-testid="stToolbar"] {
    visibility: hidden;
}

/* Style the sidebar toggle (hamburger) so users can find it
   on mobile.                                                  */
[data-testid="collapsedControl"],
button[kind="header"][data-testid="baseButton-headerNoPadding"],
button[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    color: var(--gold-400) !important;
    background: rgba(212, 168, 75, 0.1) !important;
    border: 1px solid rgba(212, 168, 75, 0.4) !important;
    border-radius: 6px !important;
}

/* Mobile-specific: make the sidebar toggle bigger and more
   visible so users can find it.                               */
@media (max-width: 768px) {
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"] {
        background: linear-gradient(180deg, #d4a84b, #b88a2d) !important;
        color: #1c1408 !important;
        box-shadow: 0 4px 14px rgba(212, 168, 75, 0.35) !important;
        font-weight: 700 !important;
    }
    /* Tighten header on mobile */
    .main-header h1 { font-size: 2.1rem !important; }
    .main-header p  { font-size: 0.9rem !important; }
    .response-card  { padding: 1.2rem 1.3rem !important; }
}

.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Initialize session state ─────────────────────────────────────────
if "assistant" not in st.session_state:
    st.session_state.assistant = StudyAssistant()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed" not in st.session_state:
    st.session_state.processed = False

assistant: StudyAssistant = st.session_state.assistant

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 0.4rem 0 0.8rem;'>
            <div style='font-size:1.6rem; color:#d4a84b;'>❦</div>
            <h2 style='margin:0; font-family:"Cormorant Garamond", serif;
                       color:#f5ecd7; font-weight:600; letter-spacing:0.04em;'>
                The Library
            </h2>
            <p style='color:#8ea0bb; font-size:0.78rem; font-style:italic;
                      letter-spacing:0.08em; margin-top:0.1rem;'>
                document manager
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your study materials",
        type=["pdf"],
        accept_multiple_files=True,
        help="Drag & drop PDF files here",
        key="pdf_uploader",
    )

    # Process button
    if uploaded_files:
        st.markdown(f"📎 **{len(uploaded_files)}** file(s) selected")

        if st.button("🔄 Process Documents", key="process_btn"):
            with st.spinner("Processing…"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.caption(msg)

                num_docs, num_chunks = assistant.ingest_pdfs(
                    uploaded_files, progress_callback=update_progress
                )
                progress_bar.progress(1.0)
                status_text.caption("✅ Complete!")

            st.success(f"Indexed **{num_docs}** document(s) → **{num_chunks}** chunks")
            st.session_state.processed = True

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Catalogue")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="status-card">
            <div class="number">{len(assistant.indexed_files)}</div>
            <div class="label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="status-card">
            <div class="number">{assistant.total_chunks}</div>
            <div class="label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    # Indexed files list
    if assistant.indexed_files:
        st.markdown("### � Volumes on the Shelf")
        items = "".join(
            f"<li>📖 {fname}</li>" for fname in assistant.indexed_files
        )
        st.markdown(
            f"<ul class='indexed-list'>{items}</ul>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Clear button
    if st.button("🗑️ Clear All Data", key="clear_btn"):
        assistant.clear_vectorstore()
        st.session_state.chat_history = []
        st.session_state.processed = False
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#8ea0bb; font-size:0.72rem;
                    letter-spacing:0.08em; font-style:italic; line-height:1.5;'>
            <span style='color:#d4a84b;'>✦</span><br/>
            Powered by LLaMA&nbsp;3.1 &amp; FAISS<br/>
            <span style='font-size:0.68rem;'>running locally via Ollama</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main Area ────────────────────────────────────────────────────────

# Header — engraved academic title
st.markdown("""
<div class="main-header">
    <div class="crest">🎓</div>
    <h1>AI <span class="accent">Study</span> Assistant</h1>
    <p>— an academic study companion powered by retrieval-augmented reasoning —</p>
</div>
<hr class="gold-rule"/>
""", unsafe_allow_html=True)

# Mode selector
MODE_OPTIONS = {
    "❓  Inquire":      "qa",
    "📝  Summarise":    "summarize",
    "📋  Quiz Me":      "mcq",
    "💡  Explain Simply": "eli5",
}

selected_mode_label = st.radio(
    "Choose a mode",
    options=list(MODE_OPTIONS.keys()),
    horizontal=True,
    label_visibility="collapsed",
    key="mode_selector",
)
mode = MODE_OPTIONS[selected_mode_label]

# ── Quick uploader (also handy on mobile where sidebar is hidden) ────
with st.expander("📎  Upload PDFs here  (or use the sidebar)", expanded=False):
    quick_files = st.file_uploader(
        "Drop PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="quick_uploader",
        label_visibility="collapsed",
    )
    qcol1, qcol2 = st.columns([3, 2])
    with qcol1:
        if quick_files:
            st.caption(f"📄 **{len(quick_files)}** file(s) ready")
        else:
            st.caption("No files selected yet.")
    with qcol2:
        process_clicked = st.button(
            "🔄 Process",
            key="quick_process_btn",
            disabled=not quick_files,
            use_container_width=True,
        )

    if process_clicked and quick_files:
        with st.spinner("Processing…"):
            qprogress = st.progress(0)
            qstatus = st.empty()

            def _qcb(pct, msg):
                qprogress.progress(pct)
                qstatus.caption(msg)

            num_docs, num_chunks = assistant.ingest_pdfs(
                quick_files, progress_callback=_qcb
            )
            qprogress.progress(1.0)
            qstatus.caption("✅ Complete!")
        st.success(f"Indexed **{num_docs}** document(s) → **{num_chunks}** chunks")
        st.session_state.processed = True
        st.rerun()

st.markdown("---")

# ── Chat history display ─────────────────────────────────────────────
if not st.session_state.chat_history and not assistant.indexed_files:
    # Welcome card — opening page of a study journal
    st.markdown("""
    <div class="welcome-card">
        <div class="emblem">⚜</div>
        <h3>Welcome to your AI Study Assistant</h3>
        <div class="subtitle">"Lege, perlege, relege" — read, study, read again</div>
        <p>Place your study materials in <em>The Library</em> on the left,
        then converse with them through one of the four academic modes below.
        <br/><span style="display:inline-block; margin-top:0.6rem; font-size:0.85rem;
        color:#9b7532; font-style:italic;">📱 On mobile? Tap the
        <strong>☰ icon</strong> at the top-left to open The Library.</span></p>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="emoji">❓</div>
                <div class="title">Inquire</div>
                <div class="desc">Ask questions and receive cited answers</div>
            </div>
            <div class="feature-item">
                <div class="emoji">📝</div>
                <div class="title">Summarise</div>
                <div class="desc">Distil chapters into key themes</div>
            </div>
            <div class="feature-item">
                <div class="emoji">📋</div>
                <div class="title">Quiz Me</div>
                <div class="desc">Generate examination-style questions</div>
            </div>
            <div class="feature-item">
                <div class="emoji">💡</div>
                <div class="title">Explain Simply</div>
                <div class="desc">Plain-language beginner explanations</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display past messages
for entry in st.session_state.chat_history:
    with st.chat_message("user", avatar="✍️"):
        st.markdown(entry["query"])
    with st.chat_message("assistant", avatar="📜"):
        st.markdown(f'<div class="response-card">{entry["answer"]}</div>', unsafe_allow_html=True)
        # Source chips
        if entry.get("sources"):
            sources_html = " ".join(
                f'<span class="source-chip">📄 {s}</span>'
                for s in entry["sources"]
            )
            with st.expander("📖 View Sources"):
                st.markdown(sources_html, unsafe_allow_html=True)
                for i, doc_text in enumerate(entry.get("source_texts", []), 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(doc_text[:500] + ("…" if len(doc_text) > 500 else ""))


# ── Chat input ───────────────────────────────────────────────────────
if prompt := st.chat_input(
    placeholder="Ask a question about your study material…",
    key="chat_input",
):
    # Display user message immediately
    with st.chat_message("user", avatar="✍️"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="📜"):
        with st.spinner("Consulting the manuscripts…"):
            answer, source_docs = assistant.generate(prompt, mode=mode)

        # Display formatted answer
        st.markdown(f'<div class="response-card">{answer}</div>', unsafe_allow_html=True)

        # Source references
        source_labels = []
        source_texts = []
        if source_docs:
            for doc in source_docs:
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                source_labels.append(f"{src} (p.{page})")
                source_texts.append(doc.page_content)

            sources_html = " ".join(
                f'<span class="source-chip">📄 {s}</span>' for s in source_labels
            )
            with st.expander("📖 View Sources"):
                st.markdown(sources_html, unsafe_allow_html=True)
                for i, doc in enumerate(source_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(
                        doc.page_content[:500]
                        + ("…" if len(doc.page_content) > 500 else "")
                    )

    # Save to history
    st.session_state.chat_history.append({
        "query": prompt,
        "mode": mode,
        "answer": answer,
        "sources": source_labels if source_docs else [],
        "source_texts": source_texts if source_docs else [],
    })
