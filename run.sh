#!/bin/bash
# array=( constant clinical linear thompson thompson2 lasso hyper randhyper random )
array=( hyper randhyper random )

for i in "${array[@]}"
do
    echo $i
    mkdir -p results/$i
	python3 main.py data/clean.csv $i results/$i/runone_$i.pkl
    echo "running 10"
    python3 main.py data/clean.csv $i results/$i/runten_$i.pkl --runten
done 