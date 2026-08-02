@echo off
title Stop Smart Retail AI Platform
echo Closing Smart Retail AI background processes...
taskkill /FI "WINDOWTITLE eq Smart Retail FastAPI Gateway*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Smart Retail Dashboard UI*" /F >nul 2>&1
echo Done! All services stopped.
timeout /t 2 >nul
