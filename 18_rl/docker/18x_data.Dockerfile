FROM cr.selcloud.ru/neurolab/neurolab_source:latest
USER 1000
WORKDIR /neurolab
RUN mkdir -p /neurolab/data/18_rl
COPY --chown=1000:1000 data/18_rl/*.gz /neurolab/data/18_rl
