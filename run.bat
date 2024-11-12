@echo off

:frontend
start cmd /k "cd front && npm install && npm start"

:backend
start cmd /k "cd back && pip install -r requirements.txt && python -m uvicorn main:app --reload"