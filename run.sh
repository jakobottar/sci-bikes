#!/bin/bash

python main.py --gpu 0 --epochs 5
python main.py --gpu 0 --epochs 10
python main.py --gpu 0 --epochs 20 --hide-loss
python main.py --gpu 0 --epochs 40 --hide-loss

wait