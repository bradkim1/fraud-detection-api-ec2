FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python packages
RUN pip install --no-cache-dir \
    jupyter \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    xgboost \
    fastapi \
    uvicorn \
    joblib

# Create directories
RUN mkdir -p /home/ubuntu/model-ec2/model /home/ubuntu/model-ec2/notebooks /home/ubuntu/model-ec2/data

# Set up Jupyter configuration
RUN mkdir -p /root/.jupyter
RUN echo "c.NotebookApp.token = ''" > /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

# Expose ports for Jupyter and FastAPI
EXPOSE 8888 8000

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--port=8888", "--allow-root", "--ip=0.0.0.0", "--no-browser"]
