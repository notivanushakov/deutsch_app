import re
import sys
from pathlib import Path

import pdfplumber
import pandas as pd


def extract_lexicon_entries(pdf_path: Path) -> pd.DataFrame:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")

    full_text = "\n".join(texts)

    lines = [l.strip() for l in full_text.splitlines()]

    start_idx = None
    end_idx = None
    for i, l in enumerate(lines):
        if start_idx is None and re.fullmatch(r"Lexicon", l, flags=re.IGNORECASE):
            start_idx = i + 1
        if end_idx is None and re.fullmatch(r"Modalpartikeln", l, flags=re.IGNORECASE):
            end_idx = i
            break

    if start_idx is None:
        raise ValueError("Could not find 'Lexicon' header in the PDF")
    if end_idx is None:
        end_idx = len(lines)

    content_lines = lines[start_idx:end_idx]

    entries = []
    for l in content_lines:
        if not l:
            continue
        # Split on hyphen/minus, en dash, or em dash with optional spaces
        parts = re.split(r"\s*[-–—]\s*", l, maxsplit=1)
        if len(parts) == 2:
            german, translation = parts[0].strip(), parts[1].strip()
            if german and translation:
                entries.append({"german": german, "translation": translation})

    df = pd.DataFrame(entries)
    if df.empty:
        raise ValueError("No lexicon entries parsed. Check PDF format.")
    return df


def main():
    pdf_path = Path(__file__).parent / "Deutsch.pdf"
    out_csv = Path(__file__).parent / "lexicon.csv"
    df = extract_lexicon_entries(pdf_path)
    df.to_csv(out_csv, index=False)
    print(f"Parsed {len(df)} entries -> {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
