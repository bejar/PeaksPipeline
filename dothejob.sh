#!/usr/bin/env bash
# /bin/sh

#python Preproceso/ConvertABF.py --exp e151126 e150707 e150514
python Preproceso/PeaksIdentification.py --exp e151126  e150707 e150514
python Preproceso/PeaksResampling.py  --exp e151126  e150707 e150514
python Preproceso/PeaksPCA.py e151126 --exp  e150707 e150514
python Clustering/PeaksClusteringHisto.py --exp e151126  e150707 e150514
