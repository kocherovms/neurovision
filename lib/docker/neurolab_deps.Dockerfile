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
RUN pip install ale-py 
RUN pip install av
RUN pip install papermill 
CMD ["/bin/bash"]
