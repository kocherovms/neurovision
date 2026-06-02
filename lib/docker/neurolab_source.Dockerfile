FROM neurolab_deps:latest
RUN git clone https://github.com/kocherovms/neurolab.git /neurolab
WORKDIR /neurolab
RUN git submodule update --init --recursive
RUN touch .docker_launch
CMD ["/bin/bash"]
