@echo off
title MedVision 3D Dashboard - Amen Allah Ben Abdrabah
echo ======================================================================
echo    MedVision 3D - AI Medical Image Segmentation React Dashboard
echo    Personalized and Maintained by Amen Allah Ben Abdrabah
echo ======================================================================
echo.
echo Launching MedVision 3D backend and React dashboard on http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
