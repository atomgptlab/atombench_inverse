#!/bin/bash
for dir in */
	fo cd "$dir"
		rm -rf distribution.pdf metrics.txt
		python ../../scripts/plot_error_distribution.py | tail -n 30 > metrics.txt
		cd ..
done
python ../scripts/text_to_csv.py
