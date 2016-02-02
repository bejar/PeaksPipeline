#!/usr/bin/env bash
# /bin/sh

#python Preproceso/ConvertABF.py --exp e110906o
python Preproceso/PeaksIdentification.py --exp e130221 e140225
python Preproceso/PeaksResampling.py  --exp e130221 e140225
python Preproceso/PeaksPCA.py --exp e130221 e140225
python Clustering/PeaksSaveClustering.py --exp e130221 e140225
#python Clustering/PeaksClusteringHisto.py --exp  e151126
#python Secuencias/PeaksFrequentSequences.py --exp  e151126