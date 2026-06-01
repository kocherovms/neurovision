FROM python:3.12.9
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install tensorflow
CMD ["/bin/bash"]
