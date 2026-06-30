FROM torch-cpu:2.12.0

# https://stackoverflow.com/questions/77331227/fontconfig-error-no-writable-cache-directories
RUN apt-get update && apt-get install -y libnss-unknown tzdata curl && rm -rf /var/lib/apt/lists/*
ENV TZ=Europe/Moscow

RUN pip install --break-system-packages --no-cache-dir pika matplotlib av
RUN pip install --break-system-packages --no-cache-dir boto3
RUN pip install --break-system-packages --no-cache-dir certifi
RUN pip install --break-system-packages --no-cache-dir moviepy==1.0.3

RUN chmod -R 777 /home
ENV HOME=/home
WORKDIR /app
COPY metrics_collector.py artifact_registry.py lang_utils.py logging_utils.py  .
# CMD ["/bin/bash"]
ENTRYPOINT ["python3", "metrics_collector.py"]

