# coding=utf-8
"""
.. module:: ConvertABF

ConvertABF
*************

:Description: ConvertABF

    Convierte los datos del formato ABF a HDF5 para poder usarlos con el resto de scripts

    Extrae el numero de señales que se indica en la definicion del experimento

:Authors: bejar
    

:Version: 

:Created on: 12/12/2014 8:19 

"""

__author__ = 'bejar'

from neo.io import AxonIO
import numpy as np
import h5py

from Config.paths import cinvesdata, cinvesdatanew

from Config.experiments import experiments


def convert_from_ABF_to_HDF5(experiment):
    """
    Convierte los ficheros del experimento desde ABF y los graba en un fichero HDF5
    :param experiment:
    :return:
    """

    # Asumimos que el fichero no existe todavia
    f = h5py.File(experiment.dpath + experiment.name + '/' + experiment.name + '.hdf5', 'w')

    nsig = len(experiment.sensors)
    datafiles = experiment.datafiles

    for dataf in datafiles:
        # Leemos los datos del fichero ABF
        print 'Reading: ', dataf, '...'
        data = AxonIO(experiment.dpath + experiment.name + '/' + dataf + '.abf')

        bl = data.read_block(lazy=False, cascade=True)
        dim = bl.segments[0].analogsignals[0].shape[0]
        matrix = np.zeros((nsig, dim))

        for j in range(nsig):
            matrix[j][:] = bl.segments[0].analogsignals[j][:].magnitude

        # Los guardamos en una carpeta del fichero HDF5 para el fichero dentro de /Raw
        print 'Saving: ', dataf, '...'
        d = f[dataf + '/Raw']
        d[()] = matrix.T
        f[dataf + '/Raw'].attrs['Sampling'] = experiment.sampling
        f[dataf + '/Raw'].attrs['Sensors'] = experiment.sensors

    f.close()


# Convierte un experimento, para convertir un grupo de experimentos se puede modificar para
# iterar sobre una lista de los experimentos existente
# Estos experimentos estan definidos en Config.experiments
if __name__ == '__main__':
    experiment = experiments['e150514']
    convert_from_ABF_to_HDF5(experiment)


