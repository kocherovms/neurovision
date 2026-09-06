#!/bin/bash
set -e
set -x

docker build -f 18x_data.Dockerfile -t cr.selcloud.ru/neurolab/18x_data:latest --load --no-cache --network=host --add-host nexus=127.0.0.1 ../..
[[ -f .docker_push ]] && docker push cr.selcloud.ru/neurolab/18x_data:latest
