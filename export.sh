#!/usr/bin/env bash

./build.sh

docker save aopdsenet | gzip -c > aopdsenet.tar.gz
