#!/usr/bin/env bash
# /bin/sh


python Spectra/Spectra.py --batch --exp e150514
python Spectra/SpectraSTFT.py --batch --exp e150514
python Analisis/VarianceVariation.py --batch --exp e150514
python Analisis/CorrelationsScatterPlot.py --batch --exp e150514
#python Analisis/PeaksStatistics.py --batch --hpeaks --exp  e150514

#python Preproceso/ConvertABF.py --batch --exp   e160204
#python Preproceso/PeaksIdentification.py  --batch --exp  e160204
#python Preproceso/PeaksResampling.py --batch  --exp   e160204
#python Preproceso/PeaksPCA.py --batch --basal meanfirst --exp  e160204
#python Clustering/PeaksSaveClustering.py --batch --exp e110906e
#python Clustering/PeaksClusteringHisto.py --batch --exp e110906e
#python Secuencias/PeaksComputeGraphsIcons.py --batch --exp  e110906e
#python Secuencias/PeaksFrequentSequences.py --batch   --string  --exp e160204
#python Sincronizaciones/PeaksSynchro.py --batch --matching --boxes --draw --exp
