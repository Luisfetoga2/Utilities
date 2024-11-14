@echo off

start cmd /c "git pull && cd front && npm install && cd ../back && pip install -r requirements.txt && exit"