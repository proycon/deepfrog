#!/bin/bash
mkdir cache
mkdir ../models
nextflow run train_tagger.nf --outputdir $(realpath ../models) --model wietsedv/bert-base-dutch-cased --traindata inputdata/conll2002-ner-nld.train.txt --devdata inputdata/conll2002-ner-nld.dev.txt --testdata inputdata/conll2002-ner-nld.test.txt --name bert-ner-cased-conll2002-nld --cache_dir $(realpath cache) --examplespath $(realpath transformers/examples)  -with-trace
