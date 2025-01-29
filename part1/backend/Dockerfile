# Use an official Python 3.11 image
FROM python:3.11-slim

# Create a working directory
WORKDIR /app

# Install all packages listed in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY main.py /app/
COPY model.pkl /app/

# Expose port 8000 for FastAPI
EXPOSE 8000

# By default, run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
