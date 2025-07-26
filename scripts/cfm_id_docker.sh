#!/bin/bash

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2
CFM_INPUT=$3
CFM_OUTPUT=$4
PROBABILITY_THRESHOLD=$5
IONIZATION=$6
ANNOTATE_FRAGMENTS=$7
APPLY_POSTPROC=$8

docker run \
    -v "$INPUT_FOLDER/$CFM_INPUT:/input.txt" \
    -v "$OUTPUT_FOLDER:/output" \
    --rm seismiq/cfmid \
    cfm-predict /input.txt $PROBABILITY_THRESHOLD \
        "/trained_models_cfmid4.0/[M${IONIZATION}H]${IONIZATION}/param_output.log" \
        "/trained_models_cfmid4.0/[M${IONIZATION}H]${IONIZATION}/param_config.txt" \
        "$ANNOTATE_FRAGMENTS" "/output/$CFM_OUTPUT" "$APPLY_POSTPROC" 0
