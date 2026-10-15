@echo off
rem ============================================================
rem  Salje slike iz primeri/ na servis i ispisuje odgovore.
rem  Servis mora da radi (pokreni.bat ili Docker kontejner).
rem  Dvoklik = test na portu 8000.
rem  Za Docker kontejner:  testiraj.bat http://127.0.0.1:8001
rem  Za jednu sliku:       testiraj.bat http://127.0.0.1:8000 C:\put\do\slike.jpg
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:8000"

if not exist "%PY%" (
    echo   [GRESKA] Nema %PY%
    goto :kraj
)

rem --- prvi argument je URL samo ako pocinje sa http ---
set "SLIKE="
echo %~1 | findstr /b /i "http" >nul 2>&1
if not errorlevel 1 (
    set "URL=%~1"
    shift
)

:petlja
if "%~1"=="" goto :kreni
set "SLIKE=%SLIKE% "%~1""
shift
goto :petlja

:kreni
"%PY%" scripts\smoke_service.py --url "%URL%"%SLIKE%

:kraj
echo.
pause
endlocal
