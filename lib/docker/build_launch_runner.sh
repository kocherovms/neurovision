#!/bin/bash
set -e
docker build -f launch_runner.Dockerfile -t launch_runner:latest -t cr.selcloud.ru/neurolab/launch_runner:latest ..
docker push cr.selcloud.ru/neurolab/launch_runner:latest
