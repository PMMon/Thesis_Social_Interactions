#!/usr/bin/env bash

curl https://vision.in.tum.de/webshare/u/dendorfp/TrajectoryPredictionData/datasets.zip -OL datasets.zip
mkdir datasets
tar -xzvf datasets.zip -C datasets
rm datasets.zip