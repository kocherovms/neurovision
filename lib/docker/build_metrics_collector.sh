#!/bin/bash
set -e
docker build -f metrics_collector.Dockerfile -t metrics_collector:latest ..
