#!/bin/bash
for dir in */
	do cd "$dir"
		rm distribution.pdf metrics.txt
		python ../../scripts/plot_error_distribution.py | tail -n 30 > metrics.txt
		cd ..
done
python ../scripts/text_to_csv.py
