#!/usr/bin/env bash
# /bin/sh


python IPFAnalysis/PeaksClusteringIPFCorrelation.py --batch --exp e150514 e110906o

#python Spectra/Spectra.py --batch --exp e150514
#python Spectra/SpectraSTFT.py --batch --exp e150514
#python Analisis/VarianceVariation.py --batch --exp e150514
#python Analisis/CorrelationsScatterPlot.py --batch --exp e150514
#python Analisis/PeaksStatistics.py --batch --hpeaks --exp  e150514

#python Preproceso/ConvertABF.py --batch --exp   e160204
#python Preproceso/PeaksIdentification.py  --batch --exp  e160204
#python Preproceso/PeaksResampling.py --batch  --detrend --exp   e110906e
#python Preproceso/PeaksPCA.py --batch --basal meanmin --exp   e110906e
#python Clustering/PeaksSaveClustering.py --batch --exp e110906e
#python Clustering/PeaksClusteringHisto.py --batch --exp e110906e
#python Secuencias/PeaksComputeGraphsIcons.py --batch --exp  e110906e
#python Secuencias/PeaksFrequentSequences.py --batch   --string  --exp e160204
#python Sincronizaciones/PeaksSynchro.py --batch --matching --boxes --draw --exp
