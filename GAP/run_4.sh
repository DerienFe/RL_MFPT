#!/usr/bin/bash
for i in {1..2000}
do
python3 ./train.py -np 4 -name t4_iter$i > ./graphs/part4_iter$i.log
done

