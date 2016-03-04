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
    f = h5py.File(experiment.dpath + experiment.name + '/' + experiment.name + '.hdf5', 'r+')

    nsig = [s for s,_ in experiment.extrasensors]

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

        # Los guardamos en una carpeta del fichero HDF5 para el fichero dentro de /Raw
        print('Saving: ', dataf, '...')
        dgroup = f[dataf]


        if dataf + '/RawExtra' in f:
            del f[dataf + '/RawExtra']
        dgroup.create_dataset('RawExtra', matrix.T.shape, dtype='f', data=matrix.T, compression='gzip')
        del matrix
        del bl

        f[dataf + '/RawExtra'].attrs['Sampling'] = experiment.sampling
        f[dataf + '/RawExtra'].attrs['Sensors'] = [s for _, s in experiment.extrasensors]
        f.flush()
    f.close()


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
        # 'e120503''e110616''e150707''e151126''e120511''e150514''e110906o'
        lexperiments = ['e110906o']

    for expname in lexperiments:

        datainfo = experiments[expname]
        convert_from_ABF_to_HDF5(datainfo)

