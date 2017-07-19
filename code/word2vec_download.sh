#!/bin/bash

OUT_F='../data/GoogleNews-vectors-negative300.bin.gz'

wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" -O $OUT_F # download word2vec file

gunzip $OUT_F # unzip the .gz file