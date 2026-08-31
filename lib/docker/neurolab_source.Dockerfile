FROM neurolab_deps:latest

RUN mkdir /neurolab
RUN chown 1000 /neurolab

# Use all files under 1000 user to allow use of MPS without root
USER 1000
RUN git clone https://github.com/kocherovms/neurolab.git /neurolab
WORKDIR /neurolab
RUN git submodule update --init --recursive
ARG CACHEBUST=1
RUN git pull --rebase

# Command scripts
USER 0

RUN echo "#!/bin/sh" >> /usr/bin/abort
RUN echo "echo abort > /neurolab/commands" >> /usr/bin/abort
RUN chmod +x /usr/bin/abort

RUN echo "#!/bin/sh" >> /usr/bin/stop
RUN echo "echo stop > /neurolab/commands" >> /usr/bin/stop
RUN chmod +x /usr/bin/stop

USER 1000

RUN mkfifo commands
RUN touch .docker_launch
CMD ["/bin/bash"]
