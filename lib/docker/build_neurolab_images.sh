#!/bin/bash
set -e

docker build -f neurolab_deps.Dockerfile   -t cr.selcloud.ru/docker/neurolab_deps:latest   -t neurolab_deps:latest --add-host nexus=127.0.0.1 --network host .
docker build -f neurolab_source.Dockerfile -t cr.selcloud.ru/docker/neurolab_source:latest -t neurolab_source:latest  .
docker push cr.selcloud.ru/docker/neurolab_deps:latest
docker push cr.selcloud.ru/docker/neurolab_source:latest