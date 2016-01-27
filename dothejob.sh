#!/usr/bin/env bash
# /bin/sh

#python Preproceso/ConvertABF.py --exp e151126 e150707 e150514
python Preproceso/PeaksIdentification.py --exp e13082
python Preproceso/PeaksResampling.py  --exp e13082
python Preproceso/PeaksPCA.py --exp e13082
#python Clustering/PeaksClusteringHisto.py --exp e130827