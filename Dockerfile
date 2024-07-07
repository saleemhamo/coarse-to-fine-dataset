# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install OpenGL libraries needed by opencv-python
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

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