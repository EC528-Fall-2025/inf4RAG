FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# use config.yaml to determine the url and api_key

# Default command to run the app
CMD ["python", "app.py"]
