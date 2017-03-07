#!/usr/bin/env bash
# /bin/sh


python Preproceso/PeaksIdentification.py  --batch --exp e130221c
python Preproceso/PeaksResampling.py --batch --exp  e130221c  
python Preproceso/PeaksPCA.py --batch --exp e130221c
#python Clustering/PeaksSaveClustering.py --batch --exp e110906e1  e110906e2  e110906e3
#python Analisis/ClusterPlot.py --batch --exp e110906e1  e110906e2  e110906e3
#python Secuencias/PeaksClusteringGraphsIcons.py --batch --exp  e110906e
#python Secuencias/PeaksFrequentSequences.py --batch   --string  --exp  e110906e1  e110906e2  e110906e3
