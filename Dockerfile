# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy application files to the working directory
COPY  app.py /app

# Copy the transformer model to the appropriate directory
COPY Models/transformer.pkl /app/Models/transformer.pkl

# Install the required dependencies from the requirements file
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port where FastAPI will run
EXPOSE 8000


# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
