FROM torch-gpu:2.12.0
RUN apt-get update && apt-get install -y vim git tzdata netcat-openbsd && rm -rf /var/lib/apt/lists/*
ENV TZ=Europe/Moscow
RUN pip install --break-system-packages --no-cache-dir jupyterlab ipywidgets  
RUN pip install --break-system-packages --no-cache-dir pandas cupy-cuda13x einops matplotlib 
RUN pip install --break-system-packages --no-cache-dir tqdm 
RUN pip install --break-system-packages --no-cache-dir optuna
RUN pip install --break-system-packages --no-cache-dir pika
RUN pip install --break-system-packages --no-cache-dir gymnasium 
RUN pip install --break-system-packages --no-cache-dir moviepy
RUN pip install --break-system-packages --no-cache-dir av
RUN pip install --break-system-packages --no-cache-dir papermill
RUN pip install --break-system-packages --no-cache-dir scipy
RUN pip install --break-system-packages --no-cache-dir boto3
RUN pip install --break-system-packages --no-cache-dir piq # SSIM metric
RUN pip install --break-system-packages --no-cache-dir ale_py==0.12.0+neurolab4 --index-url=http://nexus:8081/repository/neurolab-pypi/simple --trusted-host=nexus
CMD ["/bin/bash"]
