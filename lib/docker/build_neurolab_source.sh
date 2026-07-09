#!/bin/bash
set -e
set -x

docker build -f neurolab_source.Dockerfile -t cr.selcloud.ru/neurolab/neurolab_source:latest -t neurolab_source:latest --build-arg CACHEBUST=$(date +%s) .
[[ -f .docker_push ]] && docker push cr.selcloud.ru/neurolab/neurolab_source:latest