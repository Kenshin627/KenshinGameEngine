@echo off
rem Build KENSHIN in Debug (configure is idempotent via preset).
setlocal

cmake --preset windows-msvc-debug
if errorlevel 1 goto :fail

cmake --build --preset windows-msvc-debug
if errorlevel 1 goto :fail

echo.
echo [OK] Debug build succeeded.
echo   Demo : build\windows-msvc-debug\app\Debug\KENSHINDemo.exe
echo   Tests: build\windows-msvc-debug\tests\Debug\kenshin_tests.exe
exit /b 0

:fail
echo.
echo [ERROR] Debug build FAILED.
exit /b 1
