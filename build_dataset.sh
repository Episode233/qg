#!/bin/bash

# PQ
python build_dataset.py -k PQ --hops 2 --attempts 2 --noise 5 --samples 3000
python build_dataset.py -k PQ --hops 3 --attempts 5 --noise 3 --samples 3000

# PQL
python build_dataset.py -k PQL --hops 2 --attempts 2 --noise 5 --samples 3000
python build_dataset.py -k PQL --hops 3 --attempts 5 --noise 3 --samples 3000

# WC2014
python build_dataset.py -k WC2014 --hops 2 --attempts 2 --noise 5 --samples 3000
python build_dataset.py -k WC2014 --hops 3 --attempts 5 --noise 3 --samples 3000

# FB15k-237
python build_dataset.py -k FB15k-237 --hops 2 --attempts 2 --noise 5 --samples 3000
python build_dataset.py -k FB15k-237 --hops 3 --attempts 5 --noise 3 --samples 3000

# WN18RR
python build_dataset.py -k WN18RR --hops 2 --attempts 2 --noise 5 --samples 3000
python build_dataset.py -k WN18RR --hops 3 --attempts 5 --noise 3 --samples 3000

# YAGO3-10
python build_dataset.py -k YAGO3-10 --hops 2 --attempts 2 --noise 5 --samples 3000
python build_dataset.py -k YAGO3-10 --hops 3 --attempts 5 --noise 3 --samples 3000