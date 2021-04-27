#!/usr/bin/env bash

docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
docker-compose ps       # List containers
docker-compose exec scanner bash
