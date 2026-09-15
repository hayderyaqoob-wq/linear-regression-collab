# Bob: Updated to allow passing CLI arguments to the python script
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set entrypoint to the script, allowing additional arguments to be appended
ENTRYPOINT ["python", "src/train.py"]
CMD []