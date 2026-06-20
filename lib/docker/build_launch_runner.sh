#!/bin/bash
set -e
docker build -f launch_runner.Dockerfile -t launch_runner:latest ..
