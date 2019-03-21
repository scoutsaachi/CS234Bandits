#!/bin/bash
# array=( constant clinical linear thompson thompson2 lasso hyper randhyper random )
#array=( hyper randhyper random )
array=( hyper )

for i in "${array[@]}"
do
    echo $i
    mkdir -p results/$i
	python3 main.py data/clean.csv $i paper_results/bernoulli/$i/runone_$i.pkl
    echo "running 10"
    python3 main.py data/clean.csv $i paper_results/bernoulli/$i/runten_$i.pkl --runten
done 
