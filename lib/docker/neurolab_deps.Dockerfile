FROM torch-gpu:2.12.0
RUN apt update
RUN apt install vim emacs-nox -y
RUN apt install git -y
RUN pip install --break-system-packages jupyterlab ipywidgets  
RUN pip install --break-system-packages pandas cupy-cuda13x einops matplotlib 
RUN pip install --break-system-packages tqdm 
RUN pip install --break-system-packages optuna
RUN pip install --break-system-packages pika
RUN pip install --break-system-packages gymnasium 
RUN pip install --break-system-packages moviepy
RUN pip install --break-system-packages av
RUN pip install --break-system-packages papermill
RUN pip install --break-system-packages scipy
RUN pip install --break-system-packages boto3
ARG CACHEBUST=1
RUN pip install --break-system-packages ale_py==0.12.0+neurolab --index-url=http://nexus:8081/repository/neurolab-pypi/simple --trusted-host=nexus --no-cache-dir
CMD ["/bin/bash"]
