# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose FastAPI port
EXPOSE 5000

# Command to run FastAPI
CMD ["gunicorn", "-w", "4", "-t", "120", "app:app", "-b", "0.0.0.0:5000"]
