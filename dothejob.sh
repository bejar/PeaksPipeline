#!/usr/bin/env bash
# /bin/sh

#python Preproceso/ConvertABF.py
#python Preproceso/PeaksIdentification.py
python Preproceso/PeaksResampling.py
python Preproceso/PeaksPCA.py
python Clustering/PeaksClusteringHisto.py