@echo off

:frontend
start cmd /c "cd front && npm start"

:backend
start cmd /c "cd back && python -m uvicorn main:app --reload"