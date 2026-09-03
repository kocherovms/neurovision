FROM neurolab_deps:latest

RUN ssh-keyscan github.com > /etc/ssh/ssh_known_hosts && chmod 644 /etc/ssh/ssh_known_hosts

RUN mkdir /neurolab
RUN chown 1000 /neurolab

# Use all files under 1000 user to allow use of MPS without root
USER 1000
RUN --mount=type=secret,id=neurolab_deploy_key,target=/tmp/neurolab_deploy_key,uid=1000 \ 
    export GIT_SSH_COMMAND="ssh -i /tmp/neurolab_deploy_key -o IdentitiesOnly=yes" && \
    git clone git@github.com:kocherovms/neurolab.git /neurolab

WORKDIR /neurolab
RUN --mount=type=secret,id=neurolab_deploy_key,target=/tmp/neurolab_deploy_key,uid=1000 \ 
    export GIT_SSH_COMMAND="ssh -i /tmp/neurolab_deploy_key -o IdentitiesOnly=yes" && \
    git config --global url."git@github.com:".insteadOf "https://github.com/" && \
    git submodule update --init --recursive
ARG CACHEBUST=1
RUN --mount=type=secret,id=neurolab_deploy_key,target=/tmp/neurolab_deploy_key,uid=1000 \ 
    export GIT_SSH_COMMAND="ssh -i /tmp/neurolab_deploy_key -o IdentitiesOnly=yes" && \
    git pull --rebase

# Command scripts
USER 0

RUN echo "#!/bin/sh" >> /usr/bin/abort
RUN echo "echo abort | nc localhost 5555" >> /usr/bin/abort
RUN chmod +x /usr/bin/abort

RUN echo "#!/bin/sh" >> /usr/bin/stop
RUN echo "echo stop | nc localhost 5555" >> /usr/bin/stop
RUN chmod +x /usr/bin/stop

USER 1000

RUN touch .docker_launch
CMD ["/bin/bash"]
