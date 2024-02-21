#!/bin/bash

start="$1"
end="$2"
length="$3"
country="$4"
year="$5"
reprocess="$6"
redownload="$7"
ulflag="$8"

# bash run_job.sh 0 1000 4 "A" 2017 True True True "humid_forest.csv"

for (( i=start; i<=end; i+=5)); do
	echo "Running iteration $i..."
    python3 download_and_predict_job.py --country "$4" --year "$5" --reprocess "$6" --redownload "$7" --start "$i" --end "$(($i + 5))" --ul_flag "$8"
    echo "Finished iteration $i..."
done