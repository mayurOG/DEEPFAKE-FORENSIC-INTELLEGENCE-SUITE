@echo off
setlocal
cd /d "%~dp0"

py -3.9 -c "import streamlit, plotly, pandas, matplotlib, torch" >nul 2>&1
if errorlevel 1 (
    echo Python 3.9 or required app packages are unavailable.
    echo Install the dependencies with: py -3.9 -m pip install -r requirements.txt
    exit /b 1
)

py -3.9 -m streamlit run app.py %*
exit /b %errorlevel%
