# Deutsch Lexicon Trainer

A simple Streamlit app to generate random German words from your PDF dictionary.

## Local Run

1. Create/activate a venv (optional):
   ```powershell
   & "C:\Users\Ivan\Documents\venv\Scripts\Activate.ps1"
   ```
2. Install deps:
   ```powershell
   pip install -r requirements.txt
   ```
3. Put your PDF at `data/Deutsch.pdf`, then preprocess:
   ```powershell
   python -m data.preprocess
   ```
4. Start the app:
   ```powershell
   python -m streamlit run .\app.py
   ```

## Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. Ensure `requirements.txt` is present; the service installs dependencies automatically.
4. If your PDF is large/private, do not commit it. Instead, commit a generated `data/lexicon.csv` or add a file uploader (can be added later).

## Refresh Data

- Replace `data/Deutsch.pdf` and rerun:
  ```powershell
  python -m data.preprocess
  ```
# Welcome to GitHub Desktop!

This is your README. READMEs are where you can communicate what your project is and how to use it.

Write your name on line 6, save it, and then head back to GitHub Desktop.
