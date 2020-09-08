#!/bin/bash
die() {
    echo $@>&2
    exit 2
}

if [ ! -f inputdata/mblem.lex ]; then
    die "E-Lex lexicon not found (needs to be acquired separately due to licensing restrictions), place it in inputdata/mblem.lex"
fi


#dependency checks
command -v awk >/dev/null || die "awk not found"
command -v sed >/dev/null || die "sed not found"
command -v sesdiff >/dev/null || die "sesdiff not found (run cargo install sesdiff)"
command -v ssam >/dev/null || die "ssam not found (run cargo install ssam)"

sesdiff --suffix --abstract < inputdata/mblem.lex |
    awk '{ print $1" "$3"\n" }' > inputdata/elex-lemma-nld.txt #adds extra newlines so each word is its own sentence

if [ ! -f inputdata/cgn-lemma-nld.txt ]; then
    if [ -z "$1" ]; then
        die "inputdata/cgn-lemma-nld.txt not found, please pass the root directory of CGN corpus as first argument to this script and we create' it for you"
    fi
    CGNDIR="$1"
    find $CGNDIR/data/annot/text/plk -name "*.plk.gz"  |
        xargs zcat |
        sed "s/<au.*>//" |
        awk '{ print $1"\t"$3"\t"$2 }' |
        sesdiff  --suffix --abstract |
        awk '{ gsub(/[ \t]+$/,"",$0); if ($0 == "0") {  #restore empty lines
            print
        } else {
            print $1" "$3
        } }' > inputdata/cgn-lemma-nld.txt
fi

ssam --seed 5 -d "" -n train,test,dev -s "*,500,500" -o inputdata inputdata/cgn-lemma-nld.txt || die "failure running split sampler"

#enrich training data with elex
cat inputdata/cgn-lemma-nld.train.txt inputdata/elex-lemma-nld.txt > inputdata/cgn+elex-lemma-nld.train.txt
cp -f inputdata/cgn-lemma-nld.dev.txt inputdata/cgn+elex-lemma-nld.dev.txt #copy, cgn only
cp -f inputdata/cgn-lemma-nld.test.txt inputdata/cgn+elex-lemma-nld.test.txt #copy, cgn only

