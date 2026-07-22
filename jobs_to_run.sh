#! /bin/bash
module load python
pip install -r requirements.txt
pip install gerrrychain
pip install seaborn
pip install geopandas

python cluster_run.py
