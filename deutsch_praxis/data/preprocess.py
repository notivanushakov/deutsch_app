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

    # Pattern to detect if a line starts like a new German entry:
    # - Starts with an article (Der/Die/Das/Ein/Eine) followed by letter
    # - Starts with a German word (capital letter + lowercase)
    # - Starts with common verb patterns (lowercase: sich, etwas, jdm, etc.)
    # - Starts with lowercase verb + conjugation in parentheses
    # Lines that DON'T match this are continuations (e.g., Cyrillic text, lowercase words)
    new_entry_pattern = re.compile(
        r'^(Der|Die|Das|Ein|Eine|der|die|das|ein|eine)\s+[A-ZÄÖÜa-zäöüß]|'
        r'^[A-ZÄÖÜ][a-zäöüß]+(\s|,|\(|$)|'
        r'^(Sich|sich|etwas|jdm|jdn|für|auf|im|In|in|zu|Zu|Wir|Wenn|Was|Es|Ich|Mit|Bis|Hier|Meine|Mein|Alles|Daran|Etw|Auf|Von|von|Unter|unter|Am|am|nicht|Nicht)\s+[A-ZÄÖÜa-zäöüß]|'
        r'^an\s+(seine|A|D|G|jdn|jdm|etw)|'  # "an" only with specific continuations
        r'^[\.\…]|'
        r'^[a-zäöü]{2,}\s+(die|der|das|den|dem|einen|einem|einer|A|D|G)\s|'  # lowercase phrase with article
        r'^[a-zäöü]{3,}\s*(\(|–|-|=)|'  # lowercase verb (3+ chars) followed by parentheses or dash
        r'^[a-zäöü]+\s+[a-zäöü]+\s*\('  # lowercase + word + parenthesis (e.g., "aufgeschlossen gegenüber (Dat.)")
    )
    
    # Merge continuation lines: if a line doesn't look like a new entry start,
    # it's a continuation of the previous line
    merged_lines = []
    for l in content_lines:
        if not l:
            continue
        # Check if this line looks like the start of a new entry
        is_new_entry = new_entry_pattern.match(l)
        
        # Also check: if line starts with Cyrillic, it's definitely a continuation
        starts_with_cyrillic = bool(re.match(r'^[а-яА-ЯёЁ]', l))
        
        if merged_lines and (not is_new_entry or starts_with_cyrillic):
            # This line doesn't start like a new entry - it's a continuation
            merged_lines[-1] = merged_lines[-1] + " " + l
        else:
            merged_lines.append(l)

    entries = []
    for l in merged_lines:
        if not l:
            continue
        # Split on en dash or em dash first (preferred separators)
        # These are the proper separators between German and translation
        parts = re.split(r"\s*[–—]\s*", l, maxsplit=1)
        
        # If no en/em dash found, try hyphen but avoid splitting on plural suffixes
        # like -en, -e, -er, -s, -n which are common German grammatical notations
        if len(parts) == 1:
            # Split on hyphen only if followed by Cyrillic or longer text (not short suffixes)
            parts = re.split(r"\s+-\s+(?=[а-яА-ЯёЁa-zA-Z]{3,})", l, maxsplit=1)
        
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
