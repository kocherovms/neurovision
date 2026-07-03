FROM torch-gpu:2.12.0
RUN apt update && apt install -y vim emacs-nox git && rm -rf /var/lib/apt/lists/*
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
CMD ["/bin/bash"]
