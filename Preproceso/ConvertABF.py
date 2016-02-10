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
import os
import argparse

from Config.experiments import experiments


def convert_from_ABF_to_HDF5(experiment):
    """
    Convierte los ficheros del experimento desde ABF y los graba en un fichero HDF5
    :param experiment:
    :return:
    """

    # Asumimos que el fichero no existe todavia
    f = experiment.open_experiment_data(mode='w')

    nsig = experiment.abfsensors
    datafiles = experiment.datafiles

    for dataf in datafiles:
        # Leemos los datos del fichero ABF
        print('Reading: ', dataf, '...')
        data = AxonIO(experiment.dpath + experiment.name + '/' + dataf + '.abf')

        bl = data.read_block(lazy=False, cascade=True)
        dim = bl.segments[0].analogsignals[0].shape[0]
        matrix = np.zeros((len(nsig), dim))

        for i, j in enumerate(nsig):
            matrix[i][:] = bl.segments[0].analogsignals[j][:].magnitude

        # Los guardamos en el almacenamiento del experimento
        print('Saving: ', dataf, '...')
        experiment.save_raw_data(f, dataf, matrix.T)

        del matrix
        del bl

    datainfo.close_experiment_data(f)


# Convierte un experimento, para convertir un grupo de experimentos se puede modificar para
# iterar sobre una lista de los experimentos existente
# Estos experimentos estan definidos en Config.experiments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help="Ejecucion no interactiva", action='store_true', default=False)
    parser.add_argument('--exp', nargs='+', default=[], help="Nombre de los experimentos")

    args = parser.parse_args()
    lexperiments = args.exp

    if not args.batch:
        # 'e150514''e120503''e110616''e150707''e151126''e120511''e140225''e130221' 'e130221'
        lexperiments = ['e120503']

    for expname in lexperiments:

        datainfo = experiments[expname]
        convert_from_ABF_to_HDF5(datainfo)

        # Create the results directory if it does not exists
        if not os.path.exists(datainfo.dpath + '/' + datainfo.name + '/Results'):
            os.makedirs(datainfo.dpath + '/' + datainfo.name + '/Results')

