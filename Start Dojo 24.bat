@echo off
rem Opens Dojo 24. The first start builds the app (needs Rust: https://rustup.rs).
cd /d "%~dp0"
if not exist "target\release\dojo-24.exe" (
  echo Building Dojo 24, first start only. This takes a few minutes.
  cargo build --release || (pause & exit /b 1)
)
start "" "target\release\dojo-24.exe" %*
