@echo off
title Smart Retail AI Platform Launcher
echo ===============================================================
echo    Launching Smart Retail & Customer Intelligence Platform
echo ===============================================================
echo.

cd /d "%~dp0"

echo [1/2] Launching FastAPI Gateway (Port 8000)...
start "Smart Retail FastAPI Gateway" cmd /k "uvicorn app.main:app --port 8000"

echo [2/2] Launching Streamlit Interactive Dashboard (Port 8501)...
start "Smart Retail Dashboard UI" cmd /k "streamlit run dashboard.py --server.headless true"

echo.
echo Waiting 3 seconds for services to initialize...
timeout /t 3 >nul

echo Opening browser applications...
start http://localhost:8501
start http://localhost:8000/docs

echo ===============================================================
echo    SUCCESS! Both FastAPI & Streamlit are running.
echo    You can minimize this window.
echo ===============================================================
