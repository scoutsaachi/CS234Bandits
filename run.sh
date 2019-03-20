#!/bin/bash
# array=( constant clinical linear thompson thompson2 lasso )
array=( lasso )
# array=( constant )
for i in "${array[@]}"
do
    echo $i
    mkdir -p results/$i
	python3 main.py data/clean.csv $i results/$i/runone_$i.pkl
    echo "running 10"
    python3 main.py data/clean.csv $i results/$i/runten_$i.pkl --runten
done 
# mkdir results/constant
# mkdir results/clinical
# mkdir results/linear
# mkdir results/thompson
# mkdir results/thompson2
# mkdir 

#     "constant": ConstantBandit,
#     "clinical": ClinicalBandit,
#     "linear": WarfarinLinUCB,
#     "thompson": WarfarinThompson,
#     "thompson2": WarfarinThompsonSeparate,
#     "lasso": LassoBandit,
#     "knn": KNNUCBBandit