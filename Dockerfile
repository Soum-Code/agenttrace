FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for building certain python packages
RUN apt-get update && apt-get install -y git build-essential curl && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install visualization libraries (needed for visualizer/evals)
RUN pip install --no-cache-dir matplotlib seaborn pandas

# Copy application files
COPY . .

# Ensure start script is executable
RUN chmod +x start.sh

# Expose HF Spaces default port for Streamlit
EXPOSE 7860

CMD ["./start.sh"]
