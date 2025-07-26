#!/bin/bash
set -ex

BASEDIR=$1
INFILE=$2
OUTFILE=$3
FRAG_RECURSION_DEPTH=$4
MIN_FRAG_MASS=$5

FG_FOLDER=./development/FragGenie
java -Xmx32000m -cp $FG_FOLDER/target/liv-metfrag-0.1.0-SNAPSHOT.jar:$FG_FOLDER/target/lib/* \
    uk.ac.liverpool.metfrag.MetFragFragmenter \
    $BASEDIR/$INFILE $BASEDIR/$OUTFILE smiles $FRAG_RECURSION_DEPTH $MIN_FRAG_MASS 2048 100000 false METFRAG_MZ METFRAG_FORMULAE
