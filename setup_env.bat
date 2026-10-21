@echo off
REM One-time setup for ASPIRE Vision 2.0 on Windows (run from "Anaconda Prompt" in this folder).
REM Creates the conda env, downloads demo assets, runs tests and a quick smoke run.

call conda env create -f environment.yml || call conda env update -f environment.yml --prune
call conda activate aspire-vision2 || (echo Could not activate env & exit /b 1)

python scripts\get_demo_assets.py || exit /b 1
python -m pytest -q || exit /b 1
python scripts\run.py --config configs\demo_vtest.yaml --max-frames 200 --no-video || exit /b 1

echo.
echo Setup OK. Open the dashboard printed above in your browser.
echo Next time: conda activate aspire-vision2
