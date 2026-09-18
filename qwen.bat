@echo off
echo Starting Qwen 1.5 MoE Server...
call "%~dp0venv\Scripts\activate.bat"
python "%~dp0chat.py"
pause
