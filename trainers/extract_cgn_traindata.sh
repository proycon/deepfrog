#!/usr/bin/env bash

# Extract trainings data for the lemmatiser from CGN

usage() {
    echo "Usage: $0 [-h] [-d CGN_DIRECTORY]" >&2
}

die() {
    echo "$@">&2
    exit 1
}

CGN_DIRECTORY="."

while getopts "hd:" options; do
    case "$options" in
        h)
            usage
            ;;
        d)
            CGN_DIRECTORY="${OPTARG}"
            ;;
        :)
            echo "Error: -${OPTARG} requires an argument."
            die "option $OPTARG requires an argument"
            ;;
        *)
            die "Invalid option"
            ;;
    esac
done

command -v gunzip >/dev/null || die "gunzip not found"
command -v awk  >/dev/null || die "awk not found"
command -v sesdiff >/dev/null || die "sesdiff not found"


echo "extracting from $CGN_DIRECTORY">&2
for filename in $(find $CGN_DIRECTORY -name "*.plk.gz"); do
    echo $filename>&2
    cat $filename \
        | gunzip \
        | awk -v OFS="\t" -F"\t" '{
            if ($0 ~ /^<au.*/ || $0 == "") {    #strip SGML-like sentence delimiters, retain empty lines
                print "";
            } else {
                print $1,$3;
            }
        }' \
        | sesdiff --suffix \
        | awk -v OFS="\t" -F"\t" '{
            if ($0 == "") {
                print "";
            } else if (($2 == "_") || ($3 == "")) {        #ignore _ lemma
                print $1"\t=";
            } else {
                print $1,$3
            }
        }'
done



