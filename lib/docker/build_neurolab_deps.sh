#!/bin/bash
set -e
set -x

# Use cachebust technique to force ale_py reinstall
docker build -f neurolab_deps.Dockerfile   -t cr.selcloud.ru/neurolab/neurolab_deps:latest   -t neurolab_deps:latest --add-host nexus=127.0.0.1 --network host --build-arg CACHEBUST=$(date +%s) . 
