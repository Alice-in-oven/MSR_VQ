#!/bin/sh
set -ex

if [ $# -eq 0 ]; then
  set -- sift1m bigann deep gist1m
fi

for dataset in "$@"; do
  case "$dataset" in
    sift1m)
      mkdir -p data/sift1m
      wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -O data/sift1m/sift.tar.gz --show-progress
      tar xzf data/sift1m/sift.tar.gz -C data/sift1m --strip-components=1
      echo "Sift1m dataset download finished."
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      exit 1
      ;;
  esac
done