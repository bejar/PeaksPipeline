# /bin/sh

python Preproceso/ConvertABF.py
python Preproceso/PeaksFilterRaw.py
python Preproceso/PeaksIdentification.py
python Preproceso/PeaksResampling.py
python Preproceso/PeaksPCA.py
python Clustering/PeaksClusteringHisto.py
