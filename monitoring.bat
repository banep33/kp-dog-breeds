@echo off
rem ============================================================
rem  Podize ceo produkcioni stek: servis + Prometheus + Grafana.
rem  Dvoklik = sve se digne i otvori Grafana panel.
rem  Trazi da Docker Desktop vec radi.
rem
rem  Gasenje:  monitoring.bat stop
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"

if /i "%~1"=="stop" (
    echo   Gasim stek...
    docker compose down
    echo   Ugaseno. Dnevnik predikcija je ostao u data\prod\service.db
    goto :kraj
)

echo.
echo   Prepoznavanje rase psa  --  produkcioni stek
echo   -------------------------------------------
echo     servis     http://127.0.0.1:8001
echo     Grafana    http://127.0.0.1:3000   (bez lozinke)
echo     Prometheus http://127.0.0.1:9090
echo.

docker info >nul 2>&1
if errorlevel 1 (
    echo   [GRESKA] Docker Desktop nije pokrenut. Pokreni ga pa probaj ponovo.
    goto :kraj
)

if not exist "artifacts\model.onnx" (
    echo   [GRESKA] Nema artifacts\model.onnx -- Docker build bi pukao.
    echo            .venv\Scripts\python.exe -m dogbreeds.export
    echo            .venv\Scripts\python.exe scripts\export_notdog.py
    goto :kraj
)

if not exist "reports\drift_reference.json" (
    echo   [napomena] Nema reports\drift_reference.json -- detekcija drifta bice
    echo              iskljucena. Napravi je sa:
    echo                .venv\Scripts\python.exe scripts\make_drift_reference.py
    echo.
)

echo   Podizem kontejnere (prvi put traje duze, skida se Prometheus i Grafana)...
echo.
docker compose up -d --build
if errorlevel 1 (
    echo.
    echo   [GRESKA] Stek se nije podigao. Pogledaj poruku iznad.
    goto :kraj
)

echo.
echo   Cekam da servis bude spreman...
curl -s -o nul --retry 60 --retry-delay 1 --retry-connrefused http://127.0.0.1:8001/health
echo.
docker compose ps
echo.
echo   Otvaram Grafana panel...
start http://127.0.0.1:3000/d/kp-dog-breeds/
echo.
echo   Panel je prazan dok nema prometa. Posalji nesto:
echo     testiraj.bat http://127.0.0.1:8001
echo     .venv\Scripts\python.exe scripts\replay.py --url http://127.0.0.1:8001 --workers 2
echo.
echo   Gasenje steka:  monitoring.bat stop

:kraj
echo.
pause
endlocal
