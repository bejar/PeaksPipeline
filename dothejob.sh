#!/usr/bin/env bash
# /bin/sh

#python Preproceso/ConvertABF.py --batch --exp e120511 e120503 e130221
python Preproceso/PeaksIdentification.py  --batch --exp e151126
python Preproceso/PeaksResampling.py --batch  --exp  e151126
python Preproceso/PeaksPCA.py --batch --basal meanfirst --exp  e151126
#python Clustering/PeaksSaveClustering.py --batch --exp   e120511 e120503 e130221
#python Clustering/PeaksClusteringHisto.py --batch --exp    e120511 e120503 e130221
#python Secuencias/PeaksComputeGraphsIcons.py --batch --exp    e130221 e140225
#python Secuencias/PeaksFrequentSequences.py --batch --matching --graph --sequence  --exp    e130221 e140225
#python Sincronizaciones/PeaksSynchro.py --batch --matching --boxes --draw --exp    e130221 e140225
