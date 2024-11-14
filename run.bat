@echo off

:frontend
start cmd /k "cd front && npm start"

:backend
start cmd /k "cd back && python -m uvicorn main:app --reload"