FROM ubuntu:24.04
RUN apt update && apt install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages --no-cache-dir torch==2.12.0 torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --break-system-packages --no-cache-dir tensorboard
CMD ["/bin/bash"]
