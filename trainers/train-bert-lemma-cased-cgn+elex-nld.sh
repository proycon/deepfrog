#!/bin/bash


mkdir cache
mkdir ../models
source preprocess-lemma-data.sh

nextflow run train_tagger.nf --outputdir $(realpath ../models) --model wietsedv/bert-base-dutch-cased --traindata inputdata/cgn+elex-lemma-nld.train.txt --devdata inputdata/cgn+elex-lemma-nld.dev.txt --testdata inputdata/cgn+elex-lemma-nld.test.txt --name bert-lemma-cased-cgn+elex-nld --cache_dir $(realpath cache) --examplespath $(realpath transformers/examples)  -with-trace
