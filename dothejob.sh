#!/usr/bin/env bash
# /bin/sh

#python Preproceso/ConvertABF.py --batch --exp   e160204
#python Preproceso/PeaksIdentification.py  --batch --exp  e160204
#python Preproceso/PeaksResampling.py --batch  --exp   e160204
#python Preproceso/PeaksPCA.py --batch --basal meanfirst --exp  e160204
#python Clustering/PeaksSaveClustering.py --batch --exp  e150514 e150707 e151126 e160204
python Clustering/PeaksClusteringHisto.py --batch --exp  e150514 e150707 e151126 e160204
python Secuencias/PeaksComputeGraphsIcons.py --batch --exp   e150514 e150707 e151126 e160204
#python Secuencias/PeaksFrequentSequences.py --batch   --string  --exp e160204
#python Sincronizaciones/PeaksSynchro.py --batch --matching --boxes --draw --exp
