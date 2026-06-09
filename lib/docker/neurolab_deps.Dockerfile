FROM torch-gpu:2.12.0
RUN apt update
RUN apt install vim emacs-nox -y
RUN pip install jupyterlab ipywidgets
RUN pip install pandas cupy-cuda13x einops matplotlib 
RUN pip install tqdm 
RUN pip install optuna
RUN pip install pika
RUN pip install gymnasium 
RUN pip install moviepy
RUN pip install ale_py==0.12.0+neurolab --index-url=http://nexus:8081/repository/neurolab-pypi/simple --trusted-host=nexus --no-cache-dir
RUN pip install av
RUN pip install papermill 
CMD ["/bin/bash"]
