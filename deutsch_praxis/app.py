import random
from pathlib import Path

import pandas as pd
import streamlit as st
import json
from typing import List


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

    # Sidebar for API key (session only)
    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if api_key_input:
            st.session_state["openai_api_key"] = api_key_input.strip()
        if "openai_api_key" not in st.session_state:
            st.info("Add an API key to enable examples.")
        else:
            st.success("API key set for this session.")

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
        st.session_state["sample_df"] = df.sample(n=int(n), replace=False, random_state=None).reset_index(drop=True)

    if "sample_df" in st.session_state:
        sample_df = st.session_state["sample_df"]
        st.subheader("Today's words")
        st.dataframe(sample_df, use_container_width=True)

        # Button to generate example sentences via LLM
        if st.button("Come up with examples", key="examples_btn"):
            if "openai_api_key" not in st.session_state or not st.session_state["openai_api_key"]:
                st.error("Add an API key first in the sidebar.")
            else:
                words = sample_df.iloc[:, 0].astype(str).tolist()
                with st.spinner("Generating example sentences..."):
                    try:
                        examples_text = generate_examples(words, st.session_state["openai_api_key"])
                        st.session_state["examples_raw"] = examples_text
                    except Exception as e:
                        st.error(f"LLM request failed: {e}")

        if "examples_raw" in st.session_state:
            st.subheader("Example Sentences")
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


if __name__ == "__main__":
    main()
