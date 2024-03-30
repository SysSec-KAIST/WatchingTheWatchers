## Sample command
python3 eval_model.py -l links/class_100_link.txt -g 200 -c 80 -d NDSS_full_KT_NO_CA_1080_delete -t NDSS_full_SKT_NO_CA_1080_delete -m 600 -f -e 100
 
## Arguments

 - -c: model number
 - -d: training set dir
 - -t: test set dir
 - -m: time length, 600 -> 120 sec.
 - -g: time granularity. The current excel sheet is 0.2sec for each slot. So, 200 ms is set for this argument.
 - -e: epoch number
