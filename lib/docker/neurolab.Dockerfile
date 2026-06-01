FROM torch-gpu:2.12.0
RUN pip install pandas cupy-cuda13x einops matplotlib tqdm 
RUN pip install optuna
RUN pip install pika
RUN pip install gymnasium 
RUN pip install ale-py 
RUN pip install av
RUN pip install jupyterlab 
RUN pip install papermill 
RUN git clone https://github.com/kocherovms/neurolab.git /neurolab
WORKDIR /neurolab
RUN git submodule update --init --recursive
RUN apt update
RUN apt install vim emacs-nox -y
RUN cat "nexus 127.0.0.1" >> /etc/hosts
RUN cat "rabbitmq 127.0.0.1" >> /etc/hosts
CMD ["/bin/bash"]
