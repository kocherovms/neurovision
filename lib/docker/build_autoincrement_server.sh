#!/bin/bash
set -e
docker build -f autoincrement_server.Dockerfile -t autoincrement_server:latest ..
