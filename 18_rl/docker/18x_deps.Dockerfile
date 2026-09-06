FROM cr.selcloud.ru/neurolab/neurolab_source:latest
USER 0
RUN pip install --break-system-packages --no-cache-dir ale_py==0.12.0+neurolab4 --index-url=http://nexus:8081/repository/neurolab-pypi/simple --trusted-host=nexus
USER 1000
