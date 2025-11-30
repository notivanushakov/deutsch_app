import random
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "lexicon.csv"
PDF_PATH = Path(__file__).parent / "data" / "Deutsch.pdf"


@st.cache_data
def load_lexicon() -> pd.DataFrame:
    if not DATA_PATH.exists():
        # Try to preprocess if CSV is missing
        try:
            from data.preprocess import extract_lexicon_entries
            df = extract_lexicon_entries(PDF_PATH)
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare lexicon: {e}")
    return pd.read_csv(DATA_PATH)


def main():
    st.set_page_config(page_title="Deutsch Lexicon Trainer", page_icon="📚", layout="centered")
    st.title("Deutsch Lexicon Trainer")
    st.caption("Generate a random set of words from your dictionary.")

    # Optional: Upload a new PDF and rebuild the CSV
    with st.expander("Upload new dictionary PDF (optional)"):
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
        if uploaded_pdf is not None:
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PDF_PATH, "wb") as f:
                f.write(uploaded_pdf.read())
            try:
                from data.preprocess import extract_lexicon_entries
                df_new = extract_lexicon_entries(PDF_PATH)
                df_new.to_csv(DATA_PATH, index=False)
                st.cache_data.clear()
                st.success(f"Rebuilt lexicon with {len(df_new)} entries.")
            except Exception as e:
                st.error(f"Failed to rebuild lexicon: {e}")

    df = load_lexicon()
    st.success(f"Loaded {len(df)} entries from lexicon.")

    n = st.number_input("Number of words for today", min_value=1, max_value=max(1, len(df)), value=min(10, len(df)))
    if st.button("Generate n words for today"):
        sample_df = df.sample(n=int(n), replace=False, random_state=None)
        st.subheader("Today's words")
        st.dataframe(sample_df.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()

import random
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "lexicon.csv"
PDF_PATH = Path(__file__).parent / "data" / "Deutsch.pdf"


@st.cache_data
def load_lexicon() -> pd.DataFrame:
    if not DATA_PATH.exists():
        # Try to preprocess if CSV is missing
        try:
            from data.preprocess import extract_lexicon_entries
            df = extract_lexicon_entries(PDF_PATH)
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare lexicon: {e}")
    return pd.read_csv(DATA_PATH)


def main():
    st.set_page_config(page_title="Deutsch Lexicon Trainer", page_icon="📚", layout="centered")
    st.title("Deutsch Lexicon Trainer")
    st.caption("Generate a random set of words from your dictionary.")

    df = load_lexicon()
    st.success(f"Loaded {len(df)} entries from lexicon.")

    n = st.number_input("Number of words for today", min_value=1, max_value=max(1, len(df)), value=min(10, len(df)))
    if st.button("Generate n words for today"):
        sample_df = df.sample(n=int(n), replace=False, random_state=None)
        st.subheader("Today's words")
        st.dataframe(sample_df.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
