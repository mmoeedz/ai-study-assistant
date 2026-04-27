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

/* ── Sidebar chat history rows ─────────────────────────── */
.chat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0.6rem;
    margin: 0.25rem 0;
    background: rgba(212, 168, 75, 0.06);
    border: 1px solid rgba(212, 168, 75, 0.18);
    border-radius: 5px;
    font-size: 0.85rem;
    color: var(--parchment);
}
.chat-row-active {
    background: linear-gradient(180deg, rgba(212,168,75,0.20), rgba(212,168,75,0.08));
    border-color: var(--gold-400);
}
.chat-row-title { font-weight: 500; }
.chat-row-tag {
    color: var(--gold-300);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Make sidebar buttons (chat rows) less obtrusive */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(212, 168, 75, 0.06) !important;
    color: var(--parchment) !important;
    border: 1px solid rgba(212, 168, 75, 0.20) !important;
    border-radius: 5px !important;
    font-weight: 400 !important;
    padding: 0.4rem 0.6rem !important;
    text-align: left !important;
    box-shadow: none !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(212, 168, 75, 0.16) !important;
    border-color: rgba(212, 168, 75, 0.55) !important;
    transform: none !important;
}

/* The "+ New Chat" button stays gold */
section[data-testid="stSidebar"] button[kind="secondary"][data-testid*="new_chat"],
section[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: linear-gradient(180deg, #d4a84b, #b88a2d) !important;
    color: #1c1408 !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    font-weight: 600 !important;
    text-align: center !important;
}

/* Search input */
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(212, 168, 75, 0.25) !important;
    color: var(--parchment) !important;
    border-radius: 5px !important;
    font-size: 0.88rem !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: var(--text-mute) !important;
    font-style: italic;
}

/* ── Landing-page upload card ──────────────────────────── */
.upload-card-header {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    background: linear-gradient(180deg, rgba(212,168,75,0.10), rgba(212,168,75,0.04));
    border: 1px solid rgba(212,168,75,0.35);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    animation: fadeSlideIn 0.5s ease-out;
}
.upload-card-header .upload-icon {
    font-size: 1.6rem;
    line-height: 1;
}
.upload-card-header .upload-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--parchment);
    letter-spacing: 0.01em;
}
.upload-card-header .upload-sub {
    color: var(--text-mute);
    font-size: 0.82rem;
    font-style: italic;
    margin-top: 0.05rem;
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
   on mobile. Multiple selectors to match different Streamlit
   versions.                                                   */
[data-testid="collapsedControl"],
button[kind="header"],
button[kind="headerNoPadding"],
button[data-testid="baseButton-headerNoPadding"],
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapsedControl"],
header [aria-label*="sidebar"],
header [aria-label*="Sidebar"] {
    visibility: visible !important;
    display: inline-flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    color: var(--gold-400) !important;
    background: rgba(212, 168, 75, 0.18) !important;
    border: 1px solid rgba(212, 168, 75, 0.5) !important;
    border-radius: 6px !important;
}

/* When sidebar is collapsed, the floating control sits at top-left.
   Force it to be a noticeable gold pill.                      */
[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 0.55rem !important;
    left: 0.55rem !important;
    z-index: 9999 !important;
    background: linear-gradient(180deg, #d4a84b, #b88a2d) !important;
    color: #1c1408 !important;
    padding: 0.4rem 0.55rem !important;
    box-shadow: 0 4px 14px rgba(212, 168, 75, 0.4) !important;
}
[data-testid="collapsedControl"] svg {
    width: 1.4rem !important;
    height: 1.4rem !important;
    color: #1c1408 !important;
    fill: #1c1408 !important;
}

/* Mobile-specific: bigger toggle + tighter cards. */
@media (max-width: 768px) {
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"] {
        background: linear-gradient(180deg, #d4a84b, #b88a2d) !important;
        color: #1c1408 !important;
        box-shadow: 0 4px 14px rgba(212, 168, 75, 0.45) !important;
        font-weight: 700 !important;
        padding: 0.5rem 0.65rem !important;
    }
    /* Tighten header on mobile */
    .main-header h1 { font-size: 2.1rem !important; }
    .main-header p  { font-size: 0.9rem !important; }
    .response-card  { padding: 1.2rem 1.3rem !important; }
    /* Stack the upload card icon + text more compactly */
    .upload-card-header { padding: 0.7rem 0.85rem; gap: 0.6rem; }
    .upload-card-header .upload-title { font-size: 1.1rem; }
    .upload-card-header .upload-sub   { font-size: 0.75rem; }
}

.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Initialize session state ─────────────────────────────────────────
# ── Persistent chat sessions (ChatGPT-style threads) ─────────────────
import json
import time
import uuid
from pathlib import Path

CHATS_DIR = Path(__file__).parent / "chats"
CHATS_DIR.mkdir(exist_ok=True)


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def load_all_chats() -> list[dict]:
    """Return all saved chats sorted newest-first."""
    chats = []
    for p in CHATS_DIR.glob("*.json"):
        try:
            chats.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    chats.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return chats


def save_chat(chat: dict) -> None:
    chat["updated"] = time.time()
    if not chat.get("created"):
        chat["created"] = chat["updated"]
    _chat_path(chat["id"]).write_text(
        json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def delete_chat(chat_id: str) -> None:
    p = _chat_path(chat_id)
    if p.exists():
        p.unlink()


def new_chat() -> dict:
    return {
        "id": uuid.uuid4().hex[:10],
        "title": "New chat",
        "messages": [],
        "created": time.time(),
        "updated": time.time(),
    }


def _short_title(q: str, max_len: int = 38) -> str:
    q = (q or "").strip().splitlines()[0] if q else "New chat"
    return q if len(q) <= max_len else q[: max_len - 1] + "…"


if "assistant" not in st.session_state:
    st.session_state.assistant = StudyAssistant()
if "current_chat" not in st.session_state:
    st.session_state.current_chat = new_chat()
if "processed" not in st.session_state:
    st.session_state.processed = False
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

assistant: StudyAssistant = st.session_state.assistant
# Backwards-compat alias for the rest of the app.
st.session_state.chat_history = st.session_state.current_chat["messages"]

# ── Sidebar — New Chat / Search / History (ChatGPT-style) ────────────
with st.sidebar:
    # ── New Chat button ────────────────────────────────────────────
    if st.button("➕  New Chat", key="new_chat_btn", use_container_width=True):
        # Save the current chat (if it has any messages) before starting a new one
        if st.session_state.current_chat["messages"]:
            save_chat(st.session_state.current_chat)
        st.session_state.current_chat = new_chat()
        st.session_state.search_query = ""
        st.rerun()

    # ── Search box ─────────────────────────────────────────────────
    st.session_state.search_query = st.text_input(
        "Search chats",
        value=st.session_state.search_query,
        placeholder="🔍  Search chats…",
        label_visibility="collapsed",
        key="search_input",
    )

    st.markdown("---")

    # ── History list ───────────────────────────────────────────────
    st.markdown("### 🕰️ History")
    saved_chats = load_all_chats()
    q = st.session_state.search_query.strip().lower()

    if q:
        # Filter: title OR any message content matches
        def _matches(chat: dict) -> bool:
            if q in chat.get("title", "").lower():
                return True
            for m in chat.get("messages", []):
                if q in (m.get("query", "") or "").lower():
                    return True
                if q in (m.get("answer", "") or "").lower():
                    return True
            return False
        saved_chats = [c for c in saved_chats if _matches(c)]

    current_id = st.session_state.current_chat["id"]
    is_current_unsaved = (
        not any(c["id"] == current_id for c in saved_chats)
        and st.session_state.current_chat["messages"]
    )

    # Show the unsaved current chat at the very top
    if is_current_unsaved and not q:
        title = _short_title(
            st.session_state.current_chat["messages"][0].get("query", "")
        ) if st.session_state.current_chat["messages"] else "New chat"
        st.markdown(
            f"<div class='chat-row chat-row-active'>"
            f"<span class='chat-row-title'>{title}</span>"
            f"<span class='chat-row-tag'>active</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not saved_chats and not is_current_unsaved:
        st.caption(
            "No chats yet — ask a question and it will appear here."
            if not q else
            f"No chats matching '{st.session_state.search_query}'."
        )

    for chat in saved_chats:
        cid = chat["id"]
        is_active = cid == current_id
        title = chat.get("title") or "Untitled"
        msg_count = len(chat.get("messages", []))

        c1, c2 = st.columns([5, 1])
        with c1:
            row_label = f"{'▸ ' if is_active else ''}{title}  ·  {msg_count} msg"
            if st.button(
                row_label,
                key=f"chat_open_{cid}",
                use_container_width=True,
            ):
                # Save current chat first (if it has messages)
                if (
                    st.session_state.current_chat["messages"]
                    and st.session_state.current_chat["id"] != cid
                ):
                    save_chat(st.session_state.current_chat)
                # Load the chosen chat
                loaded = json.loads(_chat_path(cid).read_text(encoding="utf-8"))
                st.session_state.current_chat = loaded
                st.rerun()
        with c2:
            if st.button("🗑", key=f"chat_del_{cid}", help="Delete this chat"):
                delete_chat(cid)
                if cid == current_id:
                    st.session_state.current_chat = new_chat()
                st.rerun()

    st.markdown("---")

    # ── Catalogue stats ────────────────────────────────────────────
    st.markdown("### 📊 Catalogue")
    scol1, scol2 = st.columns(2)
    with scol1:
        st.markdown(f"""
        <div class="status-card">
            <div class="number">{len(assistant.indexed_files)}</div>
            <div class="label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with scol2:
        st.markdown(f"""
        <div class="status-card">
            <div class="number">{assistant.total_chunks}</div>
            <div class="label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    if assistant.indexed_files:
        st.markdown("### 📜 Indexed PDFs")
        items = "".join(
            f"<li>📖 {fname}</li>" for fname in assistant.indexed_files
        )
        st.markdown(
            f"<ul class='indexed-list'>{items}</ul>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Export current chat ────────────────────────────────────────
    if st.session_state.current_chat["messages"]:
        chat_md_lines = ["# AI Study Assistant — Chat Export\n"]
        for i, entry in enumerate(st.session_state.current_chat["messages"], 1):
            mode_label = {
                "qa": "Inquire", "summarize": "Summarise",
                "mcq": "Quiz Me", "eli5": "Explain Simply",
            }.get(entry.get("mode", "qa"), "Inquire")
            chat_md_lines.append(f"## Q{i} — {mode_label}\n")
            chat_md_lines.append(f"**Question:** {entry.get('query', '')}\n")
            chat_md_lines.append(f"**Answer:**\n\n{entry.get('answer', '')}\n")
            srcs = entry.get("sources", [])
            if srcs:
                chat_md_lines.append("**Sources:** " + ", ".join(srcs) + "\n")
            chat_md_lines.append("\n---\n")
        chat_md = "\n".join(chat_md_lines)

        st.download_button(
            "💾 Export Current Chat",
            data=chat_md,
            file_name="ai_study_assistant_chat.md",
            mime="text/markdown",
            key="export_chat_btn",
            use_container_width=True,
        )

    # ── Tips ───────────────────────────────────────────────────────
    with st.expander("💡 Study Tips", expanded=False):
        st.markdown(
            """
            - **Inquire** for fact-finding and concept explanations
            - **Summarise** before exams to revise quickly
            - **Quiz Me** to test recall actively
            - **Explain Simply** when something feels too dense
            - Upload **multiple PDFs** to query across all of them
            - Be **specific** — "explain backpropagation in 3 steps"
              works better than "explain backprop"
            """
        )

    # ── About ──────────────────────────────────────────────────────
    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            """
            **AI Study Assistant** uses Retrieval-Augmented
            Generation (RAG) to answer questions grounded in
            YOUR uploaded PDFs — never invented.

            • Cloud LLM: LLaMA-3.1-8B (via Groq)<br/>
            • Embeddings: BGE-small (in-process)<br/>
            • Vector store: NumPy cosine similarity<br/>
            • UI: Streamlit + custom academic theme

            [GitHub](https://github.com/mmoeedz/ai-study-assistant)
            """,
            unsafe_allow_html=True,
        )

    # ── Clear all data (simple, no expander) ───────────────────────
    if st.button("🗑️ Clear Data", key="clear_btn",
                 use_container_width=True,
                 help="Wipes all indexed PDFs and saved chats from disk."):
        assistant.clear_vectorstore()
        # Wipe all saved chat files
        for p in CHATS_DIR.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass
        st.session_state.current_chat = new_chat()
        st.session_state.processed = False
        st.session_state.search_query = ""
        st.rerun()

    # ── Footer ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#8ea0bb; font-size:0.72rem;
                    letter-spacing:0.08em; font-style:italic; line-height:1.5;'>
            <span style='color:#d4a84b;'>✦</span><br/>
            Powered by LLaMA&nbsp;3.1<br/>
            <span style='font-size:0.68rem;'>via Groq Cloud</span>
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

# ── Landing-page upload card (single primary upload entry point) ─────
st.markdown(
    """
    <div class="upload-card-header">
        <div class="upload-icon">📚</div>
        <div class="upload-text">
            <div class="upload-title">Upload your study material</div>
            <div class="upload-sub">PDF files only · multiple supported · processed locally</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

main_files = st.file_uploader(
    "Drop PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="main_uploader",
    label_visibility="collapsed",
)

ucol1, ucol2 = st.columns([3, 2])
with ucol1:
    if main_files:
        st.caption(f"📎 **{len(main_files)}** file(s) ready to process")
    elif assistant.indexed_files:
        st.caption(
            f"📚 **{len(assistant.indexed_files)}** document(s) already indexed — "
            f"upload more above or just start chatting below."
        )
    else:
        st.caption("Drop one or more PDFs to begin.")
with ucol2:
    main_process_clicked = st.button(
        "🔄 Process Documents",
        key="main_process_btn",
        disabled=not main_files,
        use_container_width=True,
    )

if main_process_clicked and main_files:
    with st.spinner("Processing…"):
        mprogress = st.progress(0)
        mstatus = st.empty()

        def _mcb(pct, msg):
            mprogress.progress(pct)
            mstatus.caption(msg)

        num_docs, num_chunks = assistant.ingest_pdfs(
            main_files, progress_callback=_mcb
        )
        mprogress.progress(1.0)
        mstatus.caption("✅ Complete!")
    st.success(f"Indexed **{num_docs}** document(s) → **{num_chunks}** chunks")
    st.session_state.processed = True
    st.rerun()

st.markdown("---")

# ── Chat history display ─────────────────────────────────────────────
if not st.session_state.current_chat["messages"] and not assistant.indexed_files:
    # Welcome card — opening page of a study journal
    st.markdown("""
    <div class="welcome-card">
        <div class="emblem">⚜</div>
        <h3>Welcome to your AI Study Assistant</h3>
        <div class="subtitle">"Lege, perlege, relege" — read, study, read again</div>
        <p>Pick a mode above, upload your PDFs in the gold card, and start
        asking questions. The sidebar (<strong>☰</strong> at the top-left)
        holds your stats, recent questions, and chat export.</p>
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
for entry in st.session_state.current_chat["messages"]:
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

    # Save to history (current chat thread + persist to disk)
    st.session_state.current_chat["messages"].append({
        "query": prompt,
        "mode": mode,
        "answer": answer,
        "sources": source_labels if source_docs else [],
        "source_texts": source_texts if source_docs else [],
    })
    # Auto-set chat title from the first message of the thread
    if (
        st.session_state.current_chat.get("title") in (None, "", "New chat")
        and st.session_state.current_chat["messages"]
    ):
        st.session_state.current_chat["title"] = _short_title(prompt)
    # Persist to disk so the chat survives reloads / Streamlit Cloud restarts
    save_chat(st.session_state.current_chat)
