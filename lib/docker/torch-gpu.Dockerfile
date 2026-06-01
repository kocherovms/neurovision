FROM python:3.12.9
RUN pip install torch==2.12.0 torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip install tensorflow
CMD ["/bin/bash"]
