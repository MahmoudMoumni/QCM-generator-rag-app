FROM python:3.10-slim

# Set environment variable for mode
ENV ENVIRONMENT_MODE=production
# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set work directory
WORKDIR /app

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project
COPY . .

# 6. Expose port (Flask default)
EXPOSE 5000

# 7. Run the app
CMD ["python", "app.py"]
