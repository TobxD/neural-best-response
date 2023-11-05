#!/bin/bash
conda create -n nbr python==3.11
source activate nbr
pip install -U scikit-learn
pip install pandas
pip install pyyaml
pip install matplotlib
pip install nashpy
pip install open_spiel
pip install nfn


# run bash requirements.sh
