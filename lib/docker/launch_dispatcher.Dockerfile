FROM ubuntu:24.04

RUN apt update
RUN apt install -y python3-pip
RUN pip install --break-system-packages pika  
RUN pip install --break-system-packages boto3 
RUN pip install --break-system-packages certifi # req-d for boto3 to work (otherwise boto3 will complain about self signed cert)

WORKDIR /app
COPY launch_dispatcher.py lang_utils.py logging_utils.py  .
# CMD ["python3", "launch_dispatcher.py"]
ENTRYPOINT ["python3", "launch_dispatcher.py"]
