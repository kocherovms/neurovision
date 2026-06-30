FROM ubuntu:24.04

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y tzdata
RUN apt-get install -y curl
ENV TZ=Europe/Moscow

RUN pip install --break-system-packages --no-cache-dir pika  
RUN pip install --break-system-packages --no-cache-dir boto3 
RUN pip install --break-system-packages --no-cache-dir certifi # req-d for boto3 to work (otherwise boto3 will complain about self signed cert)

WORKDIR /app
COPY launch_dispatcher.py lang_utils.py logging_utils.py  .
#CMD ["python3", "launch_dispatcher.py"]
ENTRYPOINT ["python3", "launch_dispatcher.py"]
