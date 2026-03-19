@echo off
echo ================================
echo Starting Weather Forecasting System
echo ================================
docker-compose up -d
echo.
echo ================================
echo App is running at:
echo http://localhost:8501
echo ================================
echo.
echo Press any key to view logs...
pause
docker-compose logs -f
