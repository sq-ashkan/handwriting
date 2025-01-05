FROM python:3.10-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary Python packages
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy application files
COPY api/ api/
COPY src/ src/
COPY app.py .
COPY best_model.pth .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=best_model.pth

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]