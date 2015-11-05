"""
.. module:: Experiment

Experiment
*************

:Description: Experiment

    Clase para definir experimentos y los parametros de sus diferentes analisis

:Authors: bejar
    

:Version: 

:Created on: 23/03/2015 11:44 

"""

__author__ = 'bejar'


class Experiment:
    """
    Class for the experiments
    """
    name = None  # Name of the experiment
    sampling = None  # Sampling of the raw signal
    datafiles = None  # List with the names of the datafiles
    sensors = None  # List with the names of the sensors
    dpath = None  # Path of the datafiles
    clusters = None  # List with the number of clusters for each sensor
    colors = ''  # List of colors to use for histogram of the peaks (one color for each datafile)

    # Parameters for the peaks identification, a dictionary with keys
    #  'wtime': time of the FFT window
    # 'low':  lower frequency for FFT
    # 'high': higher frequency for FFT
    # 'threshold': Amplitude threshold for the peaks
    peaks_id_params = None
    # Parameters for peaks resampling, a dictionary with keys
    # 'wsel': length of the window to select from the signal
    # 'rsfactor': resampling factor
    # 'filtered': applied to filtered signal or original signal
    peaks_resampling = None
    # Parameters for the PCA smoothing and baseline change
    # 'pcasmooth': boolean
    # 'components': number of components
    # 'wbaseline': window points to use to change the baseline
    peaks_smooth = None

    def __init__(self, dpath, name, sampling, datafiles, sensors, clusters, colors,
                 peaks_id_params, peaks_resampling, peaks_smooth):
        self.name = name
        self.sampling = sampling
        self.datafiles = datafiles
        self.sensors = sensors
        self.dpath = dpath
        self.clusters = clusters
        self.colors = colors
        self.peaks_id_params = peaks_id_params
        self.peaks_resampling = peaks_resampling
        self.peaks_smooth = peaks_smooth