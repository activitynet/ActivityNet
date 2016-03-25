#!/bin/bash

VIDEOPATH=$1
JSON_FILE=$2
TEMP_FILE="command_list.txt"

if [ -d $VIDEOPATH ]; then
    python run_crosscheck.py $VIDEOPATH $JSON_FILE $TEMP_FILE
    if [ -f $TEMP_FILE ]; then
        bash $TEMP_FILE
    else
        echo "File $TEMP_FILE does not exists."
    fi
else
    echo "Directory does not exists."
    exit 0
fi

#rm $TEMP_FILE
echo "Have a good day!"
