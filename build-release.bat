@echo off
rem Build KENSHIN in Release (configure is idempotent via preset).
setlocal

cmake --preset windows-msvc-release
if errorlevel 1 goto :fail

cmake --build --preset windows-msvc-release
if errorlevel 1 goto :fail

echo.
echo [OK] Release build succeeded.
echo   Demo : build\windows-msvc-release\app\Release\KENSHINDemo.exe
echo   Tests: build\windows-msvc-release\tests\Release\kenshin_tests.exe
exit /b 0

:fail
echo.
echo [ERROR] Release build FAILED.
exit /b 1
