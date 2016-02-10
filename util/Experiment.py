"""
.. module:: Experiment

Experiment
*************

:Description: Experiment

    Clase para definir experimentos y los parametros de sus diferentes analisis

:Authors: bejar
    

:Version: 

:Created on: 23/03/2015 11:44

* 4/2/2016, modifying the class to gain independence from data storage

"""

__author__ = 'bejar'

# from ConfigParser import SafeConfigParser
import h5py

class Experiment:
    """
    Class for the experiments
    """
    name = None  # Name of the experiment
    sampling = None  # Sampling of the raw signal
    datafiles = None  # List with the names of the datafiles
    sensors = None  # List with the names of the sensors
    abfsensors = None # List of indices of sensors  the abf file
    extrasensors = None # List of extra sensors in the file
    dpath = None  # Path of the datafiles
    clusters = None  # List with the number of clusters for each sensor
    colors = ''  # List of colors to use for histogram of the peaks (one color for each datafile)

    # Parameters for the peaks identification, a dictionary with keys
    # 'wtime': time of the FFT window
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
    # Parameters for the signal filtering
    # 'lowpass'
    # 'highpass'
    peaks_filter = None
    # names for the experiment phases
    expnames = None

    def __init__(self, dpath='', name='', sampling=0, datafiles=None, sensors=None, abfsensors=None, clusters=None,
                 colors='', peaks_id_params={}, peaks_resampling={}, peaks_smooth={}, peaks_filter={}, expnames=None,
                 extrasensors=None):
        """
        Class initialized from program

        :param dpath:
        :param name:
        :param sampling:
        :param datafiles:
        :param sensors:
        :param clusters:
        :param colors:
        :param peaks_id_params:
        :param peaks_resampling:
        :param peaks_smooth:
        :return:
        """
        self.name = name
        self.sampling = sampling
        self.datafiles = datafiles
        self.sensors = sensors
        self.abfsensors = abfsensors
        self.dpath = dpath
        self.clusters = clusters
        self.colors = colors
        self.peaks_id_params = peaks_id_params
        self.peaks_resampling = peaks_resampling
        self.peaks_smooth = peaks_smooth
        self.peaks_filter = peaks_filter
        if expnames is None:
            self.expnames = datafiles
        else:
            self.expnames = expnames
        if extrasensors is None:
            self.extrasensors = []
        else:
            self.extrasensors = extrasensors


    # def load_config(self, file):
    #     """
    #     Object read from configuration file
    # 
    #     :param path:
    #     :return:
    #     """
    # 
    #     cnf = SafeConfigParser()
    # 
    #     cnf.read(file)
    # 
    #     self.name = cnf.get('Experiment', 'Name')
    #     self.sampling = cnf.getfloat('Experiment', 'Sampling')
    #     tmp = cnf.get('Experiment', 'Datafiles')
    #     self.datafiles = tmp.replace('\n', '').split(',')
    #     tmp = cnf.get('Experiment', 'Expnames')
    #     self.expnames = tmp.replace('\n', '').split(',')
    # 
    # 
    #     tmp = cnf.get('Experiment', 'Sensors')
    #     self.sensors = tmp.replace('\n', '').split(',')
    #     tmp = cnf.get('Experiment', 'ABFSensors')
    #     self.abfsensors = [int(s) for s in tmp.replace('\n', '').split(',')]
    # 
    #     self.dpath = cnf.get('Experiment', 'DataPath')
    #     tmp = cnf.get('Clustering', 'Clusters')
    #     self.clusters = [int(nc) for nc in tmp.replace('\n', '').split(',')]
    #     self.colors = cnf.get('Clustering', 'Colors')
    # 
    #     self.peaks_id_params = {'wtime': cnf.getfloat('Identification', 'wtime'), 'low': cnf.getfloat('Identification', 'low'),
    #                             'high': cnf.getfloat('Identification', 'high'),
    #                             'threshold': cnf.getfloat('Identification', 'threshold')}
    # 
    #     self.peaks_resampling = {'wtsel': cnf.getfloat('Resampling', 'wtsel'), 'rsfactor': cnf.getfloat('Resampling', 'rsfactor'),
    #                              'filtered': cnf.getboolean('Resampling', 'filtered')}
    # 
    #     self.peaks_smooth = {'pcasmooth': cnf.getboolean('Smoothing', 'pcasmooth'), 'components': cnf.getint('Smoothing', 'components'),
    #                              'wsbaseline': cnf.getint('Smoothing', 'wbaseline')}
    # 
    #     self.peaks_filter = {'lowpass': cnf.getfloat('Filter', 'lowpass'), 'highpass': cnf.getfloat('Filter', 'highpass')}


    def open_experiment_data(self, mode):
        """
        Returns a handle to the experiment data

        Depends on the backend of the data
        :return:
        """
        return h5py.File(self.dpath + self.name + '/' + self.name + '.hdf5', mode)

    def close_experiment_data(self, handle):
        """
        Closes access to the experiment data

        :param handle:
        :return:
        """
        handle.close()


    def get_peaks_smooth_parameters(self, param):
        """
        Returns the parameters for the smoothing of the peaks

        :return:
        """
        return self.peaks_smooth[param]

    def get_peaks_resample_parameters(self, param):
        """
        Gets values from the parameters for peak resampling
        :param param:
        :return:
        """
        return self.peaks_resampling[param]

    def get_peaks_resample(self, f, dfile, sensor):
        """
        Gets the resample peaks from the file

        :param dfile:
        :param sensor:
        :return:
        """
        if dfile + '/' + sensor + '/' + 'PeaksResample' in f:
            d = f[dfile + '/' + sensor + '/' + 'PeaksResample']
            return d[()]
        else:
            return None

    def get_clean_time(self, f, dfile, sensor):
        """
        returns the cleaned list of time peaks

        :param f:
        :return:
        """
        if dfile + '/' + sensor + '/TimeClean' in f:
            return f[dfile + '/' + sensor + '/' + 'TimeClean']
        else:
            return None

    def save_peaks_time_clean(self, f, dfile, sensor, ntimes):
        """
        Saves the times of the peaks after cleaning the data

        :param f:
        :param dfile:
        :param sensor:
        :param times:
        :return:
        """
        if dfile + '/' + sensor + '/' + 'TimeClean' in f:
            del f[dfile + '/' + sensor + '/' + 'TimeClean']
        d = f.require_dataset(dfile + '/' + sensor + '/' + 'TimeClean', ntimes.shape, dtype='i',
                              data=ntimes, compression='gzip')
        d[()] = ntimes


    def get_peaks_time(self, f, dfile, sensor):
        """
        returns the cleaned list of time peaks

        :param f:
        :return:
        """
        if dfile + '/' + sensor + '/Time' in f:
            return f[dfile + '/' + sensor + '/' + 'Time']
        else:
            return None

    def get_peaks_resample_PCA(self, f, dfile, sensor):
        """
        Gets the resample peaks from the file

        :param dfile:
        :param sensor:
        :return:
        """
        if dfile + '/' + sensor + '/' + 'PeaksResamplePCA' in f:
            d = f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
            return d[()]
        else:
            return None

    def save_peaks_resample_PCA(self, f, dfile, sensor, trans):
        """
        saves the resampled and PCAded peaks in the dataset

        :param dfile:
        :param sensor:
        :return:
        """
        if dfile + '/' + sensor + '/' + 'PeaksResamplePCA' in f:
            del f[dfile + '/' + sensor + '/' + 'PeaksResamplePCA']
        d = f.require_dataset(dfile + '/' + sensor + '/' + 'PeaksResamplePCA', trans.shape, dtype='f',
                              data=trans, compression='gzip')
        d[()] = trans
        if self.peaks_smooth['pcasmooth']:
            f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['Components'] = self.peaks_smooth['components']
        else:
            f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['Components'] = 0

        f[dfile + '/' + sensor + '/PeaksResamplePCA'].attrs['baseline'] = self.peaks_smooth['wbaseline']

    def save_peaks_clustering_centroids(self, f, dfile, sensor, centers):
        """
        Saves the centroids of the clusters in the dataset

        :param f:
        :param dfile:
        :param sensor:
        :param centers:
        :return:
        """

        if dfile + '/' + sensor + '/Clustering/Centers' in f:
            del f[dfile + '/' + sensor + '/Clustering/Centers']
        d = f.require_dataset(dfile + '/' + sensor + '/Clustering/' + 'Centers', centers.shape, dtype='f',
                              data=centers, compression='gzip')
        d[()] = centers

    def save_raw_data(self, f, dfile, matrix):
        """
        Saves the raw data for an experiment file
        :param f:
        :return:
        """

        dgroup = f.create_group(dfile)
        dgroup.create_dataset('Raw', matrix.shape, dtype='f', data=matrix, compression='gzip')

        f[dfile + '/Raw'].attrs['Sampling'] = self.sampling
        f[dfile + '/Raw'].attrs['Sensors'] = self.sensors

        f.flush()



# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
    # a = Experiment('../Manual/config_example.cfg')
    # 
    # print(a.name)
    # print(a.sampling)
    # print(a.datafiles)
    # print(a.sensors)
    # print(a.dpath)
    # print(a.clusters)
    # print(a.colors)
    # print(a.peaks_id_params)
    # print(a.peaks_resampling)
    # print(a.peaks_smooth)
    # print(a.peaks_filter)
