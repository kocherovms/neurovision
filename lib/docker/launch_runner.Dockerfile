FROM ubuntu:24.04

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y tzdata
ENV TZ=Europe/Moscow

RUN pip install --break-system-packages boto3 
RUN pip install --break-system-packages certifi # req-d for boto3 to work (otherwise boto3 will complain about self signed cert)
RUN pip install --break-system-packages docker 

WORKDIR /app
COPY launch_runner.py lang_utils.py logging_utils.py  .
# CMD ["python3", "launch_runner.py"]
ENTRYPOINT ["python3", "launch_runner.py"]
