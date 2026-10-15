@echo off
rem ============================================================
rem  Provera da li ceo monitoring stek radi kako treba.
rem  Dvoklik = provera bez slanja slika.
rem  provera.bat promet  = prvo posalje primeri/ pa proveri
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo   [GRESKA] Nema %PY%
    goto :kraj
)

if /i "%~1"=="promet" (
    "%PY%" scripts\check_stack.py --promet
) else (
    "%PY%" scripts\check_stack.py
)

:kraj
echo.
pause
endlocal
