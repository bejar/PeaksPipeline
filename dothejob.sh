#!/usr/bin/env bash
# /bin/sh

#python Preproceso/ConvertABF.py --exp e110906o
#python Preproceso/PeaksIdentification.py --exp e130221 e140225
#python Preproceso/PeaksResampling.py  --exp e130221 e140225
python Preproceso/PeaksPCA.py --batch --basal meanmin --exp e150514 e150707 e151126
python Clustering/PeaksSaveClustering.py --exp   e150514 e150707 e151126
python Clustering/PeaksClusteringHisto.py --exp    e150514 e150707 e151126
python Secuencias/PeaksComputeGraphsIcons.py --exp   e150514 e150707 e151126
python Secuencias/PeaksFrequentSequences.py --batch --matching --graph --sequence  --exp    e150514 e150707 e151126
python Sincronizaciones/PeaksSynchro.py --batch --matching --boxes --draw --exp    e150514 e150707 e151126