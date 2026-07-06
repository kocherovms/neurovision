FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y tzdata && \
    rm -rf /var/lib/apt/lists/*
ENV TZ=Europe/Moscow

RUN pip install --break-system-packages --no-cache-dir boto3 
RUN pip install --break-system-packages --no-cache-dir certifi # req-d for boto3 to work (otherwise boto3 will complain about self signed cert)
RUN pip install --break-system-packages --no-cache-dir docker 

WORKDIR /app
COPY launch_runner.py lang_utils.py logging_utils.py  .
# CMD ["python3", "launch_runner.py"]
ENTRYPOINT ["python3", "launch_runner.py"]
