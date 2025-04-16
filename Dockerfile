FROM python:3.10-slim

# Create working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# Install required system packages
RUN apt-get update && apt-get install -y curl


RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run the app
CMD ["python", "app/main.py"]
