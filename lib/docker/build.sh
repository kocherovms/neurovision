#!/bin/bash
set -e
docker build -f autoincrement_server.Dockerfile -t autoincrement_server:latest ..
docker build -f metrics_collector.Dockerfile -t metrics_collector:latest ..
