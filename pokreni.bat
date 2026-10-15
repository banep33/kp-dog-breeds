@echo off
rem ============================================================
rem  Pokrece servis za prepoznavanje rase psa i otvara pregledac.
rem  Dvoklik na ovaj fajl je dovoljan.
rem  Zatvaranje ovog prozora gasi servis.
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"

set "PORT=8000"
set "PY=.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:%PORT%"

echo.
echo   Prepoznavanje rase psa  --  lokalni servis
echo   -----------------------------------------
echo.

if not exist "%PY%" (
    echo   [GRESKA] Nema virtuelnog okruzenja: %PY%
    echo            Napravi ga pa instaliraj zavisnosti:
    echo              py -3.14 -m venv .venv
    echo              .venv\Scripts\python.exe -m pip install -r requirements.txt
    goto :kraj
)

if not exist "artifacts\model.onnx" (
    echo   [GRESKA] Nema artifacts\model.onnx
    echo            Izvezi model:
    echo              .venv\Scripts\python.exe -m dogbreeds.export
    echo              .venv\Scripts\python.exe scripts\export_notdog.py
    goto :kraj
)

rem --- da li vec nesto slusa na tom portu? ---
curl -s -o nul -m 2 "%URL%/health" 2>nul
if not errorlevel 1 (
    echo   Servis vec radi na %URL% -- samo otvaram pregledac.
    start "" "%URL%/"
    echo.
    echo   Ako zelis da ga restartujes, zatvori prvo onaj drugi prozor.
    goto :kraj
)

echo   Model:    artifacts\model.onnx
echo   Adresa:   %URL%
echo   Zaustavi: Ctrl+C  ili zatvori ovaj prozor
echo.
echo   Cekam da se model ucita (prvi put ~5 s)...
echo.

rem --- u pozadini cekamo /health pa otvaramo pregledac ---
start "" /min cmd /c "curl -s -o nul --retry 60 --retry-delay 1 --retry-connrefused %URL%/health && start %URL%/"

rem --- servis u prvom planu, da se logovi vide ---
"%PY%" -m uvicorn service.app:app --host 127.0.0.1 --port %PORT%

echo.
echo   Servis je zaustavljen.

:kraj
echo.
pause
endlocal
