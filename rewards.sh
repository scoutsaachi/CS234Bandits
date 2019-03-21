#!/bin/bash

array=( 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 )

for i in "${array[@]}"
do
    echo $i
    python3 main.py data/clean.csv thompson rewards/runten_$i.pkl --runten --alpha $i
done 