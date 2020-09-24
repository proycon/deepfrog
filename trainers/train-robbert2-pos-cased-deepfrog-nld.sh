#!/bin/bash
mkdir cache
mkdir ../models
nextflow run train_tagger.nf --outputdir $(realpath ../models) --model pdelobelle/robbert-v2-dutch-base --traindata inputdata/frog1.1-pos-nld.train.txt --devdata inputdata/frog1.1-pos-nld.dev.txt --save_steps 250000 --testdata inputdata/frog1.1-pos-nld.test.txt --name robbert-pos-cased-deepfrog-nld --cache_dir $(realpath cache) --examplespath $(realpath transformers/examples)  -with-trace
