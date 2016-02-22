#!/usr/bin/env bash
# /bin/sh

python Preproceso/ConvertABF.py --batch --exp e151126
python Preproceso/PeaksIdentification.py  --batch --exp  e151126
python Preproceso/PeaksResampling.py --batch  --exp   e151126
python Preproceso/PeaksPCA.py --batch --basal meanfirst --exp   e151126
python Clustering/PeaksSaveClustering.py --batch --exp   e151126
python Clustering/PeaksClusteringHisto.py --batch --exp   e151126
python Secuencias/PeaksComputeGraphsIcons.py --batch --exp   e151126
#python Secuencias/PeaksFrequentSequences.py --batch --matching --graph --sequence  --exp
#python Sincronizaciones/PeaksSynchro.py --batch --matching --boxes --draw --exp
