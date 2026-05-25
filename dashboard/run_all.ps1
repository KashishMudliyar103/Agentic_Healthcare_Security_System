Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Major_project_healthcare_security; .\.venv\Scripts\activate; python main.py api"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Major_project_healthcare_security; .\.venv\Scripts\activate; python main.py dashboard"

Start-Sleep -Seconds 5

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Major_project_healthcare_security; .\.venv\Scripts\activate; python main.py demo 50"

Start-Sleep -Seconds 3
Start-Process "http://localhost:8501"