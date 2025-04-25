powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt

python app.py
