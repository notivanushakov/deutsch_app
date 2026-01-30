import random
from pathlib import Path

import pandas as pd
import streamlit as st
import json
from typing import List


DATA_PATH = Path(__file__).parent / "data" / "lexicon.csv"
PDF_PATH = Path(__file__).parent / "data" / "Deutsch.pdf"
CACHE_PATH = Path(__file__).parent / "data" / "session_cache.json"


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS FOR MODERN UI
# ─────────────────────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Page Navigation Tabs */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        padding: 20px 0;
        margin-bottom: 30px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .nav-btn {
        padding: 12px 28px;
        border-radius: 12px;
        border: 2px solid transparent;
        cursor: pointer;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .nav-btn-inactive {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #a0a0a0;
        border-color: #2a2a4a;
    }
    
    .nav-btn-inactive:hover {
        background: linear-gradient(135deg, #2a2a4e 0%, #26315e 100%);
        color: #ffffff;
        border-color: #4a4a7a;
        transform: translateY(-2px);
    }
    
    .nav-btn-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Flashcard Styles */
    .flashcard-container {
        perspective: 1000px;
        width: 100%;
        max-width: 450px;
        height: 280px;
        margin: 30px auto;
    }
    
    .flashcard {
        width: 100%;
        height: 100%;
        position: relative;
        transform-style: preserve-3d;
        transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .flashcard.flipped {
        transform: rotateY(180deg);
    }
    
    .flashcard-face {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border-radius: 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 30px;
        box-sizing: border-box;
    }
    
    .flashcard-front {
        background: linear-gradient(145deg, #1e1e2f 0%, #2d2d44 100%);
        border: 2px solid #3d3d5c;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .flashcard-back {
        background: linear-gradient(145deg, #2d4a3e 0%, #1a3d2e 100%);
        border: 2px solid #3d6a5c;
        transform: rotateY(180deg);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .flashcard-word {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 15px;
    }
    
    .flashcard-hint {
        font-size: 0.9rem;
        color: #888;
        margin-top: 20px;
    }
    
    .flashcard-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #666;
        margin-bottom: 10px;
    }
    
    /* Progress Bar */
    .progress-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        margin: 20px 0;
    }
    
    .progress-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #2a2a4a;
        transition: all 0.3s ease;
    }
    
    .progress-dot.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        transform: scale(1.2);
    }
    
    .progress-dot.completed {
        background: #4ade80;
    }
    
    /* Quiz Option Buttons */
    .quiz-option {
        width: 100%;
        padding: 18px 24px;
        margin: 10px 0;
        border-radius: 14px;
        border: 2px solid #3d3d5c;
        background: linear-gradient(145deg, #1e1e2f 0%, #2d2d44 100%);
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
    }
    
    .quiz-option:hover {
        border-color: #667eea;
        background: linear-gradient(145deg, #2a2a4f 0%, #3d3d64 100%);
        transform: translateX(5px);
    }
    
    .quiz-option.correct {
        border-color: #4ade80;
        background: linear-gradient(145deg, #1a3d2e 0%, #2d4a3e 100%);
        box-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
    }
    
    .quiz-option.incorrect {
        border-color: #f87171;
        background: linear-gradient(145deg, #3d1a1a 0%, #4a2d2d 100%);
        box-shadow: 0 0 20px rgba(248, 113, 113, 0.3);
    }
    
    /* Quiz Question Card */
    .quiz-question-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #2d2d44 100%);
        border: 2px solid #3d3d5c;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .quiz-question-word {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    
    .quiz-question-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #888;
    }
    
    /* Navigation Buttons */
    .card-nav-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin: 25px 0;
    }
    
    .card-nav-btn {
        padding: 12px 30px;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .card-nav-prev, .card-nav-next {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        color: white;
    }
    
    .card-nav-prev:hover, .card-nav-next:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Score Display */
    .score-display {
        text-align: center;
        padding: 30px;
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        border: 2px solid #2a2a4a;
        margin: 20px 0;
    }
    
    .score-number {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .score-label {
        font-size: 1.2rem;
        color: #888;
        margin-top: 10px;
    }
    
    /* Stats Cards */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
    }
    
    .stat-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c;
        border-radius: 16px;
        padding: 20px 30px;
        text-align: center;
        min-width: 120px;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Custom Button Styling */
    .stButton > button {
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Page Title Styling */
    .page-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
    }
    
    .page-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .flashcard-container {
            height: 220px;
        }
        .flashcard-word {
            font-size: 1.6rem;
        }
        .quiz-question-word {
            font-size: 1.8rem;
        }
        .nav-btn {
            padding: 10px 16px;
            font-size: 13px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE AND DATA FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def load_persistent_cache():
    """Load cached sample_df and examples from disk if available."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            return cache
        except Exception:
            return {}
    return {}


def save_persistent_cache(sample_df=None, examples_raw=None):
    """Save sample_df and examples to disk for persistence across reloads."""
    cache = {}
    if sample_df is not None:
        cache['sample_df'] = sample_df.to_dict(orient='records')
    if examples_raw is not None:
        cache['examples_raw'] = examples_raw
    
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save cache: {e}")


def clear_persistent_cache():
    """Clear the persistent cache file."""
    if CACHE_PATH.exists():
        try:
            CACHE_PATH.unlink()
        except Exception:
            pass


@st.cache_data
def load_lexicon() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    if PDF_PATH.exists():
        try:
            from data.preprocess import extract_lexicon_entries
            df = extract_lexicon_entries(PDF_PATH)
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to prepare lexicon from PDF: {e}")

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_lexicon_from_bytes(pdf_bytes: bytes) -> pd.DataFrame:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    from data.preprocess import extract_lexicon_entries
    df = extract_lexicon_entries(tmp_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION COMPONENT
# ─────────────────────────────────────────────────────────────────────────────
def render_navigation():
    """Render the page navigation tabs."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "dictionary"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Dictionary", key="nav_dict", use_container_width=True, 
                     type="primary" if st.session_state["current_page"] == "dictionary" else "secondary"):
            st.session_state["current_page"] = "dictionary"
            st.rerun()
    
    with col2:
        if st.button("🎴 Flashcards", key="nav_flash", use_container_width=True,
                     type="primary" if st.session_state["current_page"] == "flashcards" else "secondary"):
            st.session_state["current_page"] = "flashcards"
            st.rerun()
    
    with col3:
        if st.button("🎯 Quiz", key="nav_quiz", use_container_width=True,
                     type="primary" if st.session_state["current_page"] == "quiz" else "secondary"):
            st.session_state["current_page"] = "quiz"
            st.rerun()
    
    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# FLASHCARDS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_flashcards_page(df: pd.DataFrame):
    """Render the flashcards practice page."""
    st.markdown('<h1 class="page-title">🎴 Flashcards</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Click the card to reveal the answer</p>', unsafe_allow_html=True)
    
    # Initialize flashcard state
    if "flashcard_words" not in st.session_state:
        st.session_state["flashcard_words"] = None
    if "flashcard_index" not in st.session_state:
        st.session_state["flashcard_index"] = 0
    if "flashcard_flipped" not in st.session_state:
        st.session_state["flashcard_flipped"] = False
    if "flashcard_mode" not in st.session_state:
        st.session_state["flashcard_mode"] = "german_to_translation"
    
    # Flashcard mode selector
    st.markdown("##### Flashcard Mode")
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("🇩🇪 German → Translation", key="flash_mode_g2t", use_container_width=True,
                     type="primary" if st.session_state["flashcard_mode"] == "german_to_translation" else "secondary"):
            if st.session_state["flashcard_mode"] != "german_to_translation":
                st.session_state["flashcard_mode"] = "german_to_translation"
                st.session_state["flashcard_words"] = None  # Reset cards
                st.session_state["flashcard_flipped"] = False
                st.rerun()
    with mode_col2:
        if st.button("🌍 Translation → German", key="flash_mode_t2g", use_container_width=True,
                     type="primary" if st.session_state["flashcard_mode"] == "translation_to_german" else "secondary"):
            if st.session_state["flashcard_mode"] != "translation_to_german":
                st.session_state["flashcard_mode"] = "translation_to_german"
                st.session_state["flashcard_words"] = None  # Reset cards
                st.session_state["flashcard_flipped"] = False
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate cards button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Generate 10 New Cards", key="gen_flashcards", use_container_width=True):
            num_cards = min(10, len(df))
            st.session_state["flashcard_words"] = df.sample(n=num_cards).reset_index(drop=True)
            st.session_state["flashcard_index"] = 0
            st.session_state["flashcard_flipped"] = False
            st.rerun()
    
    if st.session_state["flashcard_words"] is not None and len(st.session_state["flashcard_words"]) > 0:
        cards = st.session_state["flashcard_words"]
        idx = st.session_state["flashcard_index"]
        current_card = cards.iloc[idx]
        is_flipped = st.session_state["flashcard_flipped"]
        is_reverse = st.session_state["flashcard_mode"] == "translation_to_german"
        
        # Progress dots
        progress_html = '<div class="progress-container">'
        for i in range(len(cards)):
            if i == idx:
                progress_html += '<div class="progress-dot active"></div>'
            else:
                progress_html += '<div class="progress-dot"></div>'
        progress_html += '</div>'
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # Card counter
        st.markdown(f"<p style='text-align: center; color: #888; font-size: 0.9rem;'>Card {idx + 1} of {len(cards)}</p>", unsafe_allow_html=True)
        
        # Flashcard display with click-to-flip JavaScript
        flipped_class = "flipped" if is_flipped else ""
        german_word = current_card["german"]
        translation = current_card["translation"]
        
        # Determine front/back based on mode
        if is_reverse:
            front_label = "Translation"
            front_word = translation
            back_label = "German"
            back_word = german_word
        else:
            front_label = "German"
            front_word = german_word
            back_label = "Translation"
            back_word = translation
        
        # Create a unique key for the flip button that we'll click via JS
        flip_key = f"flip_card_{idx}_{is_flipped}"
        
        card_html = f'''
        <div class="flashcard-container" onclick="flipCard()" style="cursor: pointer;">
            <div class="flashcard {flipped_class}" id="flashcard">
                <div class="flashcard-face flashcard-front">
                    <div class="flashcard-label">{front_label}</div>
                    <div class="flashcard-word">{front_word}</div>
                    <div class="flashcard-hint">👆 Click anywhere on card to flip</div>
                </div>
                <div class="flashcard-face flashcard-back">
                    <div class="flashcard-label">{back_label}</div>
                    <div class="flashcard-word">{back_word}</div>
                    <div class="flashcard-hint">👆 Click anywhere on card to flip back</div>
                </div>
            </div>
        </div>
        <script>
            function flipCard() {{
                // Find and click the hidden flip button
                const buttons = window.parent.document.querySelectorAll('button');
                for (const btn of buttons) {{
                    if (btn.innerText.includes('Flip Card')) {{
                        btn.click();
                        break;
                    }}
                }}
            }}
        </script>
        '''
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Flip button (now also clickable, but card click works too)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Flip Card", key="flip_card", use_container_width=True):
                st.session_state["flashcard_flipped"] = not st.session_state["flashcard_flipped"]
                st.rerun()
        
        # Navigation buttons - equal width columns
        st.markdown("<br>", unsafe_allow_html=True)
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        
        with nav_col1:
            if st.button("⬅️ Previous", key="prev_card", use_container_width=True, disabled=(idx == 0)):
                st.session_state["flashcard_index"] = idx - 1
                st.session_state["flashcard_flipped"] = False
                st.rerun()
        
        with nav_col3:
            if st.button("Next ➡️", key="next_card", use_container_width=True, disabled=(idx >= len(cards) - 1)):
                st.session_state["flashcard_index"] = idx + 1
                st.session_state["flashcard_flipped"] = False
                st.rerun()
    else:
        st.info("👆 Click 'Generate 10 New Cards' to start practicing!")


# ─────────────────────────────────────────────────────────────────────────────
# QUIZ PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_quiz_page(df: pd.DataFrame):
    """Render the multiple choice quiz page."""
    st.markdown('<h1 class="page-title">🎯 Multiple Choice Quiz</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Select the correct answer for each question</p>', unsafe_allow_html=True)
    
    # Initialize quiz state
    if "quiz_questions" not in st.session_state:
        st.session_state["quiz_questions"] = None
    if "quiz_index" not in st.session_state:
        st.session_state["quiz_index"] = 0
    if "quiz_wrong_attempts" not in st.session_state:
        st.session_state["quiz_wrong_attempts"] = {}  # {question_idx: set of wrong options}
    if "quiz_answered" not in st.session_state:
        st.session_state["quiz_answered"] = {}  # {question_idx: True when correctly answered}
    if "quiz_mode" not in st.session_state:
        st.session_state["quiz_mode"] = "german_to_translation"  # or "translation_to_german"
    
    # Quiz mode selector
    st.markdown("##### Quiz Mode")
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("🇩🇪 German → Translation", key="mode_g2t", use_container_width=True,
                     type="primary" if st.session_state["quiz_mode"] == "german_to_translation" else "secondary"):
            if st.session_state["quiz_mode"] != "german_to_translation":
                st.session_state["quiz_mode"] = "german_to_translation"
                st.session_state["quiz_questions"] = None  # Reset quiz
                st.rerun()
    with mode_col2:
        if st.button("🌍 Translation → German", key="mode_t2g", use_container_width=True,
                     type="primary" if st.session_state["quiz_mode"] == "translation_to_german" else "secondary"):
            if st.session_state["quiz_mode"] != "translation_to_german":
                st.session_state["quiz_mode"] = "translation_to_german"
                st.session_state["quiz_questions"] = None  # Reset quiz
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate quiz button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎲 Generate New Quiz (10 Questions)", key="gen_quiz", use_container_width=True):
            num_questions = min(10, len(df))
            quiz_words = df.sample(n=num_questions).reset_index(drop=True)
            
            # Generate options for each question based on mode
            questions = []
            is_reverse = st.session_state["quiz_mode"] == "translation_to_german"
            
            for i, row in quiz_words.iterrows():
                if is_reverse:
                    # Show translation, pick German word
                    question_text = row["translation"]
                    correct_answer = row["german"]
                    wrong_answers = df[df["german"] != correct_answer]["german"].sample(n=min(3, len(df)-1)).tolist()
                else:
                    # Show German, pick translation
                    question_text = row["german"]
                    correct_answer = row["translation"]
                    wrong_answers = df[df["translation"] != correct_answer]["translation"].sample(n=min(3, len(df)-1)).tolist()
                
                options = [correct_answer] + wrong_answers
                random.shuffle(options)
                questions.append({
                    "question": question_text,
                    "correct": correct_answer,
                    "options": options,
                    "is_reverse": is_reverse
                })
            
            st.session_state["quiz_questions"] = questions
            st.session_state["quiz_index"] = 0
            st.session_state["quiz_wrong_attempts"] = {}
            st.session_state["quiz_answered"] = {}
            st.rerun()
    
    if st.session_state["quiz_questions"] is not None and len(st.session_state["quiz_questions"]) > 0:
        questions = st.session_state["quiz_questions"]
        idx = st.session_state["quiz_index"]
        current_q = questions[idx]
        
        # Progress dots
        progress_html = '<div class="progress-container">'
        for i in range(len(questions)):
            if i == idx:
                progress_html += '<div class="progress-dot active"></div>'
            elif i in st.session_state["quiz_answered"]:
                progress_html += '<div class="progress-dot completed"></div>'
            elif i in st.session_state["quiz_wrong_attempts"] and len(st.session_state["quiz_wrong_attempts"][i]) > 0:
                progress_html += '<div class="progress-dot" style="background: #fbbf24;"></div>'  # Yellow for in-progress
            else:
                progress_html += '<div class="progress-dot"></div>'
        progress_html += '</div>'
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # Stats row
        correct_count = len(st.session_state["quiz_answered"])
        total_wrong = sum(len(v) for v in st.session_state["quiz_wrong_attempts"].values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Question", f"{idx + 1} / {len(questions)}")
        with col2:
            st.metric("✅ Completed", f"{correct_count}")
        with col3:
            st.metric("❌ Mistakes", f"{total_wrong}")
        
        # Question card - adapt label based on mode
        is_reverse = current_q.get("is_reverse", False)
        question_label = "Which German word means:" if is_reverse else "What does this mean?"
        
        st.markdown(f'''
        <div class="quiz-question-card">
            <div class="quiz-question-label">{question_label}</div>
            <div class="quiz-question-word">{current_q["question"]}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Check if this question was correctly answered
        is_completed = idx in st.session_state["quiz_answered"]
        wrong_attempts = st.session_state["quiz_wrong_attempts"].get(idx, set())
        
        # Option buttons
        for i, option in enumerate(current_q["options"]):
            is_wrong_attempt = option in wrong_attempts
            is_correct_answer = option == current_q["correct"]
            
            if is_completed:
                # Question completed - show the correct answer
                if is_correct_answer:
                    st.button(f"✅ {option}", key=f"opt_{idx}_{i}", use_container_width=True, disabled=True)
                elif is_wrong_attempt:
                    st.button(f"❌ {option}", key=f"opt_{idx}_{i}", use_container_width=True, disabled=True)
                else:
                    st.button(option, key=f"opt_{idx}_{i}", use_container_width=True, disabled=True)
            else:
                # Question not completed yet
                if is_wrong_attempt:
                    # Already tried this wrong answer - show as disabled with X
                    st.button(f"❌ {option}", key=f"opt_{idx}_{i}", use_container_width=True, disabled=True)
                else:
                    # Can still click this option
                    if st.button(option, key=f"opt_{idx}_{i}", use_container_width=True):
                        if is_correct_answer:
                            st.session_state["quiz_answered"][idx] = True
                            st.rerun()
                        else:
                            # Wrong answer - add to wrong attempts
                            if idx not in st.session_state["quiz_wrong_attempts"]:
                                st.session_state["quiz_wrong_attempts"][idx] = set()
                            st.session_state["quiz_wrong_attempts"][idx].add(option)
                            st.rerun()
        
        # Show feedback message
        if is_completed:
            attempts = len(wrong_attempts)
            if attempts == 0:
                st.success("🎉 Perfect! Got it on the first try!")
            elif attempts == 1:
                st.success("👏 Correct! Got it on the second try.")
            else:
                st.success(f"✅ Got it after {attempts + 1} attempts.")
        elif len(wrong_attempts) > 0:
            remaining = 4 - len(wrong_attempts) - 1  # -1 for the correct answer
            st.warning(f"❌ Not that one! {remaining} option(s) left to try.")
        
        # Navigation buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col2:
            if st.button("⬅️ Previous", key="prev_q", use_container_width=True, disabled=(idx == 0)):
                st.session_state["quiz_index"] = idx - 1
                st.rerun()
        
        with col4:
            if st.button("Next ➡️", key="next_q", use_container_width=True, disabled=(idx >= len(questions) - 1)):
                st.session_state["quiz_index"] = idx + 1
                st.rerun()
        
        # Show final score when all questions answered
        if len(st.session_state["quiz_answered"]) == len(questions):
            st.markdown("---")
            total = len(questions)
            total_mistakes = sum(len(v) for v in st.session_state["quiz_wrong_attempts"].values())
            # Score based on first-try successes
            first_try_correct = sum(1 for i in range(total) if i not in st.session_state["quiz_wrong_attempts"] or len(st.session_state["quiz_wrong_attempts"].get(i, set())) == 0)
            percentage = int((first_try_correct / total) * 100)
            
            st.markdown(f'''
            <div class="score-display">
                <div class="score-number">{percentage}%</div>
                <div class="score-label">{first_try_correct} out of {total} correct on first try!</div>
                <div style="color: #888; font-size: 0.9rem; margin-top: 10px;">Total mistakes: {total_mistakes}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            if percentage == 100:
                st.balloons()
                st.success("🏆 Perfect score! You're a German vocabulary master!")
            elif percentage >= 70:
                st.success("👏 Great job! Keep practicing!")
            else:
                st.info("📚 Keep studying! Practice makes perfect!")
    else:
        st.info("👆 Click 'Generate New Quiz' to start testing your knowledge!")


# ─────────────────────────────────────────────────────────────────────────────
# DICTIONARY PAGE (Original functionality)
# ─────────────────────────────────────────────────────────────────────────────
def render_dictionary_page(df: pd.DataFrame):
    """Render the dictionary/word generator page (original functionality)."""
    st.markdown('<h1 class="page-title">📚 Dictionary Trainer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Generate random words and get example sentences</p>', unsafe_allow_html=True)
    
    # Sidebar for API key (session only)
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if api_key_input:
            st.session_state["openai_api_key"] = api_key_input.strip()
        if "openai_api_key" not in st.session_state:
            st.info("Add an API key to enable examples.")
        else:
            st.success("✅ API key set for this session.")
    
    # Upload + rebuild flow
    with st.expander("📁 Upload dictionary PDF"):
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False, key="pdf_upload")
        if uploaded_pdf is not None:
            st.info("PDF selected. Click 'Build lexicon' to process.")
            if st.button("Build lexicon from uploaded PDF", key="process_pdf"):
                with st.status("Processing PDF into lexicon…", expanded=True) as status:
                    status.update(label="Parsing entries", state="running")
                    try:
                        pdf_bytes = uploaded_pdf.getvalue()
                        df_new = build_lexicon_from_bytes(pdf_bytes)
                        status.update(label="Saving CSV", state="running")
                        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                        df_new.to_csv(DATA_PATH, index=False)
                        st.cache_data.clear()
                        clear_persistent_cache()
                        status.update(label=f"Done: {len(df_new)} entries", state="complete")
                        st.success(f"Rebuilt lexicon with {len(df_new)} entries.")
                        st.session_state["pdf_processed"] = True
                    except Exception as e:
                        status.update(label="Failed", state="error")
                        st.error(f"Failed to rebuild lexicon: {e}")
        else:
            if PDF_PATH.exists() and st.button("Rebuild lexicon from existing PDF", key="rebuild_from_disk"):
                with st.status("Processing existing PDF…", expanded=True) as status:
                    status.update(label="Parsing entries", state="running")
                    try:
                        from data.preprocess import extract_lexicon_entries
                        df_new = extract_lexicon_entries(PDF_PATH)
                        status.update(label="Saving CSV", state="running")
                        df_new.to_csv(DATA_PATH, index=False)
                        st.cache_data.clear()
                        clear_persistent_cache()
                        status.update(label=f"Done: {len(df_new)} entries", state="complete")
                        st.success(f"Rebuilt lexicon with {len(df_new)} entries.")
                        st.session_state["pdf_processed"] = True
                    except Exception as e:
                        status.update(label="Failed", state="error")
                        st.error(f"Failed to rebuild lexicon: {e}")

    if df.empty:
        st.warning("No lexicon found. Upload a PDF to build it.")
        return

    st.success(f"📖 Loaded {len(df)} entries from lexicon.")

    # Load persistent cache on first run
    if "sample_df" not in st.session_state and "examples_raw" not in st.session_state:
        cache = load_persistent_cache()
        if cache.get('sample_df'):
            st.session_state["sample_df"] = pd.DataFrame(cache['sample_df'])
        if cache.get('examples_raw'):
            st.session_state["examples_raw"] = cache['examples_raw']

    n = st.number_input("Number of words for today", min_value=1, max_value=max(1, len(df)), value=min(10, len(df)))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Generate Words", key="gen_words", use_container_width=True):
            st.session_state["sample_df"] = df.sample(n=int(n), replace=False, random_state=None).reset_index(drop=True)
            if "examples_raw" in st.session_state:
                del st.session_state["examples_raw"]
            save_persistent_cache(sample_df=st.session_state["sample_df"], examples_raw=None)
            st.rerun()

    if "sample_df" in st.session_state:
        sample_df = st.session_state["sample_df"]
        st.subheader("📝 Today's Words")
        st.dataframe(sample_df, use_container_width=True)

        with col2:
            if st.button("💡 Get Example Sentences", key="examples_btn", use_container_width=True):
                if "openai_api_key" not in st.session_state or not st.session_state["openai_api_key"]:
                    st.error("Add an API key first in the sidebar.")
                else:
                    words = sample_df.iloc[:, 0].astype(str).tolist()
                    with st.spinner("Generating example sentences..."):
                        try:
                            examples_text = generate_examples(words, st.session_state["openai_api_key"])
                            st.session_state["examples_raw"] = examples_text
                            save_persistent_cache(sample_df=st.session_state["sample_df"], examples_raw=examples_text)
                        except Exception as e:
                            st.error(f"LLM request failed: {e}")

        if "examples_raw" in st.session_state:
            st.subheader("💬 Example Sentences")
            st.markdown(st.session_state["examples_raw"])


def generate_examples(words: List[str], api_key: str) -> str:
    # Trim list to a reasonable size to avoid huge prompts
    max_words = 50
    use_words = words[:max_words]
    prompt = (
        "You are a helpful assistant. Given this list of German words, write one natural German sentence for each word using it in context. "
        "Do not translate, just provide the German sentences. Return in markdown list format. Words:\n" + "\n".join(use_words)
    )

    # Try new OpenAI client first, fallback to legacy if needed
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": "Assistant generating German example sentences."}, {"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = response.choices[0].message.content
        return content
    except Exception:
        import openai  # type: ignore
        openai.api_key = api_key
        completion = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": "Assistant generating German example sentences."}, {"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Deutsch Praxis - German Vocabulary Trainer", 
        page_icon="🇩🇪", 
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Load lexicon data
    df = load_lexicon()
    
    # Render navigation
    render_navigation()
    
    # Route to appropriate page
    current_page = st.session_state.get("current_page", "dictionary")
    
    if current_page == "dictionary":
        render_dictionary_page(df)
    elif current_page == "flashcards":
        if df.empty:
            st.warning("⚠️ No lexicon found. Please go to Dictionary page and upload a PDF first.")
        else:
            render_flashcards_page(df)
    elif current_page == "quiz":
        if df.empty:
            st.warning("⚠️ No lexicon found. Please go to Dictionary page and upload a PDF first.")
        else:
            render_quiz_page(df)


if __name__ == "__main__":
    main()
