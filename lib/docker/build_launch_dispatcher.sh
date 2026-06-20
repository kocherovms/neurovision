#!/bin/bash
set -e
docker build -f launch_dispatcher.Dockerfile -t launch_dispatcher:latest ..
