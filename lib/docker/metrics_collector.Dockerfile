FROM torch-cpu:2.12.0

# https://stackoverflow.com/questions/77331227/fontconfig-error-no-writable-cache-directories
RUN apt-get update
RUN apt-get install -y libnss-unknown
RUN pip install pika matplotlib av
RUN pip install boto3
RUN pip install moviepy==1.0.3

RUN chmod -R 777 /home
ENV HOME=/home
WORKDIR /app
COPY metrics_collector.py artifact_registry.py lang_utils.py logging_utils.py  .
# CMD ["/bin/bash"]
ENTRYPOINT ["python", "metrics_collector.py"]

