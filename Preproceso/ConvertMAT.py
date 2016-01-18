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


import numpy as np
import h5py
import os
import scipy.io

from Config.experiments import experiments
import argparse

def convert_from_MAT_to_HDF5(experiment):
    """
    Convierte los ficheros del experimento desde ABF y los graba en un fichero HDF5
    :param experiment:
    :return:
    """

    # Asumimos que el fichero no existe todavia
    f = h5py.File(experiment.dpath + experiment.name + '/' + experiment.name + '.hdf5', 'w')

    nsig = experiment.abfsensors
    datafiles = experiment.datafiles

    for dataf in datafiles:
        # Leemos los datos del fichero ABF
        print('Reading: ', dataf, '...')
        data = scipy.io.loadmat(experiment.dpath + experiment.name + '/' + dataf + '.mat')
        #AxonIO(experiment.dpath + experiment.name + '/' + dataf + '.abf')

        matrix = data['data']

        print('Saving: ', dataf, '...')
        dgroup = f.create_group(dataf)

        dgroup.create_dataset('Raw', matrix.shape, dtype='f', data=matrix, compression='gzip')
        del matrix

        f[dataf + '/Raw'].attrs['Sampling'] = experiment.sampling
        f[dataf + '/Raw'].attrs['Sensors'] = experiment.sensors
        f.flush()
    f.close()


# Convierte un experimento, para convertir un grupo de experimentos se puede modificar para
# iterar sobre una lista de los experimentos existente
# Estos experimentos estan definidos en Config.experiments
if __name__ == '__main__':
    # 'e150514''e120503''e110616''e150707''e151126''e120511'
    lexperiments = ['e120511']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    if args.exp:
        lexperiments = args.exp

    for experiment in lexperiments:
        convert_from_MAT_to_HDF5(experiments[experiment])
        # Create the results directory if does not exists
        if not os.path.exists(experiment.dpath + '/' + experiment.name + '/Results'):
            os.makedirs(experiment.dpath + '/' + experiment.name + '/Results')

