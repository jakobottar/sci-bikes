#!/bin/bash

var=$(date)
echo "-----------------------------------------"
echo "DATE: $var"
echo "-----------------------------------------"

python main.py --gpu 2

wait