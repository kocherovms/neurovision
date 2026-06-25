FROM neurolab_deps:latest
RUN mkdir /neurolab
RUN chown 1000 /neurolab
# Use all files under 1000 user to allow use of MPS without root
USER 1000
RUN git clone https://github.com/kocherovms/neurolab.git /neurolab
WORKDIR /neurolab
RUN git submodule update --init --recursive
RUN touch .docker_launch
CMD ["/bin/bash"]
