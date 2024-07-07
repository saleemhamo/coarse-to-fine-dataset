# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --progress-bar off

# Copy the current directory contents into the container at /app
COPY . /app/

# Set the environment variable for MOUNTED_CLAIM_DIRECTORY
ARG MOUNTED_CLAIM_DIRECTORY
ENV MOUNTED_CLAIM_DIRECTORY=${MOUNTED_CLAIM_DIRECTORY}

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Make port 80 available to the world outside this container
EXPOSE 80

# Default command to keep the container running
CMD ["sleep", "infinity"]
