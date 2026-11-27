@echo off
rem Rebuilds the training engine from scratch: ai-toolkit at the pinned commit,
rem a private Python 3.12, PyTorch for CUDA 13.0 and the engine requirements,
rem then builds the Dojo 24 window (when Rust is installed).
setlocal
cd /d "%~dp0"
set ENGINE_COMMIT=4a7f28532ed6fe81d9f8c00718878bb3b8739015
set UV_PYTHON_INSTALL_DIR=%~dp0engine\python

where uv >nul 2>nul || (echo uv is required: https://docs.astral.sh/uv/ & pause & exit /b 1)
where git >nul 2>nul || (echo git is required. & pause & exit /b 1)

if not exist engine mkdir engine
if not exist "engine\ai-toolkit\run.py" (
  git clone https://github.com/ostris/ai-toolkit.git "engine\ai-toolkit" || goto fail
)
git -C "engine\ai-toolkit" fetch --depth 1 origin %ENGINE_COMMIT% && git -C "engine\ai-toolkit" checkout -q %ENGINE_COMMIT%

uv python install 3.12 || goto fail
if not exist "engine\venv\Scripts\python.exe" (
  uv venv "engine\venv" --python 3.12 || goto fail
)
uv pip install --python "engine\venv\Scripts\python.exe" torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 || goto fail
uv pip install --python "engine\venv\Scripts\python.exe" -r "engine\ai-toolkit\requirements.txt" || goto fail

"engine\venv\Scripts\python.exe" -c "import torch;print('PyTorch',torch.__version__,'on',torch.cuda.get_device_name(0))" || goto fail
where cargo >nul 2>nul && (cargo build --release || goto fail)
echo.
echo Done. Start it with "Start Dojo 24.bat".
pause
exit /b 0

:fail
echo.
echo Setup stopped on the step above.
pause
exit /b 1
