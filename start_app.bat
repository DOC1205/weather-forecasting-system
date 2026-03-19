@echo off
chcp 65001 >nul
title Weather Forecasting System
cd C:\Projects\weather-forecasting-system
call venv\Scripts\activate
echo.
echo ================================
echo  Weather Forecasting System
echo ================================
echo.
echo [1] Start web app (Streamlit)
echo [2] Open Jupyter Notebook
echo [3] Retrain model
echo [4] Run evaluation
echo [5] Exit
echo.
set /p choice="Choose option (1-5): "

if "%choice%"=="1" (
    streamlit run app.py
) else if "%choice%"=="2" (
    jupyter notebook
) else if "%choice%"=="3" (
    python src\train.py --model hybrid
    pause
) else if "%choice%"=="4" (
    python src\evaluate.py
    pause
) else (
    exit
)
