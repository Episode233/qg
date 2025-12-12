#!/bin/bash

## 稀疏小图
# PQ    点数 2256   边数 3369
python utils/build_dataset.py -k PQ --hops 2 --attempts 10 --noise 3 --samples 10000
python utils/build_dataset.py -k PQ --hops 3 --attempts 10 --noise 3 --samples 10000

# PQL   点数 6505   边数 5575
python utils/build_dataset.py -k PQL --hops 2 --attempts 10 --noise 3 --samples 10000
python utils/build_dataset.py -k PQL --hops 3 --attempts 50 --noise 3 --samples 10000


## 稠密小图
# WC2014    点数 1127   边数 6476
python utils/build_dataset.py -k WC2014 --hops 2 --attempts 5 --noise 3 --samples 10000
python utils/build_dataset.py -k WC2014 --hops 3 --attempts 5 --noise 3 --samples 10000


## 标准中图
# WN18RR    点数 31538   边数 81381
python utils/build_dataset.py -k WN18RR --hops 2 --attempts 1 --noise 3 --samples 10000
python utils/build_dataset.py -k WN18RR --hops 3 --attempts 1 --noise 3 --samples 10000


## 稠密中图
# FB15k-237    点数 13436   边数 250021
python utils/build_dataset.py -k FB15k-237 --hops 2 --attempts 1 --noise 3 --samples 10000
python utils/build_dataset.py -k FB15k-237 --hops 3 --attempts 1 --noise 3 --samples 10000


## 稠密大图
# YAGO3-10    点数 123183   边数 797298
python utils/build_dataset.py -k YAGO3-10 --hops 2 --attempts 1 --noise 3 --samples 10000
python utils/build_dataset.py -k YAGO3-10 --hops 3 --attempts 1 --noise 3 --samples 10000