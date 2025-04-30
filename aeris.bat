@echo off
REM Define the path to your virtual environment and the Python script
set VENV_PATH=C:\Users\gca20\Documents\voice_ai\env\Scripts\activate.bat
set APP_DIR=C:\Users\gca20\Documents\voice_ai\
set PYTHON_SCRIPT_PATH=C:\Users\gca20\Documents\voice_ai\app.py

REM Activate the virtual environment
call %VENV_PATH%

REM Change to the specified directory
cd /d %APP_DIR%
REM start cmd waitress-serve --port=5000 app:app
start cmd /k python %PYTHON_SCRIPT_PATH%

REM Wait for 3 seconds
timeout /t 3 /nobreak

REM Open Microsoft Edge and navigate to the local server
start msedge "http://127.0.0.1:5000"
