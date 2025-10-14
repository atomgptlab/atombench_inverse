#!/bin/bash


python scripts/overlay_compare.py \
  --alex-csv-files alexandria/dataset1.csv alexandria/dataset2.csv \
  --jarvis-dataset supercon_3d \
  --jarvis-target-key Tc \
  --output ./overlay_outputs \
  --tc-min 0 --tc-max 35 --tc-step 1

