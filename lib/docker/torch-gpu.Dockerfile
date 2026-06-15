FROM ubuntu:24.04
RUN apt update
RUN apt install python3-pip -y
RUN pip install torch==2.12.0 torchvision --index-url https://download.pytorch.org/whl/cu130 --break-system-packages
RUN pip install tensorflow --break-system-packages
RUN pip install tensorboard --break-system-packages
CMD ["/bin/bash"]
