#!/bin/bash
set -e
set -x

docker build -f neurolab_source.Dockerfile -t cr.selcloud.ru/neurolab/neurolab_source:latest -t neurolab_source:latest --build-arg CACHEBUST=$(date +%s) --secret id=neurolab_deploy_key,src=/home/misha/.ssh/neurolab_deploy_key .
[[ -f .docker_push ]] && docker push cr.selcloud.ru/neurolab/neurolab_source:latest