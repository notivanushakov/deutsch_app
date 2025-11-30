import random
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "lexicon.csv"
PDF_PATH = Path(__file__).parent / "data" / "Deutsch.pdf"


@st.cache_data
def load_lexicon() -> pd.DataFrame:
    # If the CSV exists, just read it
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    # If CSV is missing but PDF exists, try to build
    if PDF_PATH.exists():
        try:
            from data.preprocess import extract_lexicon_entries
            df = extract_lexicon_entries(PDF_PATH)
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to prepare lexicon from PDF: {e}")

    # Neither CSV nor PDF exist — return empty df to let UI handle upload
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_lexicon_from_bytes(pdf_bytes: bytes) -> pd.DataFrame:
    # Cache key is the bytes content; this persists across reruns on Cloud
    tmp_pdf = Path(st.experimental_get_query_params().get("_tmp_dir", [str(Path.cwd())])[0]) / "uploaded.pdf"
    tmp_pdf.write_bytes(pdf_bytes)
    from data.preprocess import extract_lexicon_entries
    df = extract_lexicon_entries(tmp_pdf)
    return df


def main():
    st.set_page_config(page_title="Deutsch Lexicon Trainer", page_icon="📚", layout="centered")
    st.title("Deutsch Lexicon Trainer")
    st.caption("Generate a random set of words from your dictionary.")

    # Upload + rebuild flow
    with st.expander("Upload dictionary PDF"):
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
                        status.update(label=f"Done: {len(df_new)} entries", state="complete")
                        st.success(f"Rebuilt lexicon with {len(df_new)} entries.")
                        st.session_state["pdf_processed"] = True
                    except Exception as e:
                        status.update(label="Failed", state="error")
                        st.error(f"Failed to rebuild lexicon: {e}")
        else:
            # Optional button to rebuild from an already present PDF on disk
            if PDF_PATH.exists() and st.button("Rebuild lexicon from existing PDF", key="rebuild_from_disk"):
                with st.status("Processing existing PDF…", expanded=True) as status:
                    status.update(label="Parsing entries", state="running")
                    try:
                        from data.preprocess import extract_lexicon_entries
                        df_new = extract_lexicon_entries(PDF_PATH)
                        status.update(label="Saving CSV", state="running")
                        df_new.to_csv(DATA_PATH, index=False)
                        st.cache_data.clear()
                        status.update(label=f"Done: {len(df_new)} entries", state="complete")
                        st.success(f"Rebuilt lexicon with {len(df_new)} entries.")
                        st.session_state["pdf_processed"] = True
                    except Exception as e:
                        status.update(label="Failed", state="error")
                        st.error(f"Failed to rebuild lexicon: {e}")

    df = load_lexicon()

    if df.empty:
        st.warning("No lexicon found. Upload a PDF to build it.")
        return

    st.success(f"Loaded {len(df)} entries from lexicon.")

    n = st.number_input("Number of words for today", min_value=1, max_value=max(1, len(df)), value=min(10, len(df)))
    if st.button("Generate n words for today"):
        sample_df = df.sample(n=int(n), replace=False, random_state=None)
        st.subheader("Today's words")
        st.dataframe(sample_df.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
