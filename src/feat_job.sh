#!/bin/bash
# Extract arguments
start="$1"
end="$2"
country="$3"
year="$4"
csv="${5:-$'process_area_2022.csv'}"

#bash feat_job.sh 0 10 'Ghana' 2022 'ew_haze.csv'
# Loop through the range
for (( i=start; i<=end; i+=5)); do
	echo "Running iteration $i..."
	echo "Start, $1"
	echo "End, $2"
	echo "Country, $3"
	echo "Year, $4"
	echo "CSV, $csv"
    # Run the Python script with the variable as argument
    python3 download_and_predict_job.py \
    --country "$3" \
    --year "$4" \
    --reprocess True \
    --redownload True \
    --start "$i" \
    --end "$(($i + 5))" \
    --ul_flag True \
    --db_path $csv \
    --make_training_data False \
    --gen_feats True \
    --process True \
    --length 12 \
    --predict_model_path "../models/224-2023-new/" # this is the current plantations TTC model weights
    echo "Finished iteration $i..."
done