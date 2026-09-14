@echo off
cd /d "%~dp0"
title Bank AI - Starting
echo Installing/checking project requirements...
py -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Requirements installation failed. Please install Python 3.10 or newer and try again.
    pause
    exit /b 1
)
echo.
echo Starting Bank AI interface...
py -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo The interface could not start. Make sure Python and the project requirements are installed.
    pause
)
