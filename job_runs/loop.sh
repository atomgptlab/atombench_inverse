#!/bin/bash
for dir in */
	do cd "$dir"
		rm -rf distribution.pdf metrics.json
		python ../../scripts/plot_error_distribution.py #| tail -n 30 > metrics.txt
		cd ..
done
python ../scripts/json_to_csv.py