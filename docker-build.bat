@echo off
echo ================================
echo Building Docker Image...
echo ================================
docker build -t weather-forecasting-system:latest .
echo.
echo ================================
echo Build Complete!
echo ================================
pause
