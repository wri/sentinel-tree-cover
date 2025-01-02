#!/bin/bash

# Extract arguments
start="$1"
end="$2"
length="$3"
country="$4"
year="$5"
reprocess="$6"
redownload="$7"
ulflag="$8"
csv="${9:-$'process_area_2022.csv'}"
n_rows="${10:-$'8'}"

echo "Running iteration $i..."
echo "Start, $1"
echo "End, $2"
echo "Country, $4"
echo "Year, $5"
echo "CSV, $csv"
echo "Bucket, $bucket"
echo "Folder, $folder"
echo "Train size, $train_size"
echo "N Rows, $n_rows"

# bash run_job.sh 0 3000 4 Cambodia 2022 True True True 'asia.csv' 8
# Loop through the range
for (( i=start; i<=end; i+=10)); do
        echo "Running iteration $i..."
    # Run the Python script with the variable as argument
    python3.11 download_and_predict_job.py --country "$4" --year "$5" --reprocess "$6" --redownload "$7" --start "$i" --end "$(($i + 5))" --ul_flag "$8" --n_rows $n_rows --db_path $csv
    echo "Finished iteration $i..."
done