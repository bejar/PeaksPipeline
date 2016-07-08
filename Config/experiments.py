"""
.. module:: experiments

experiments
*************

:Description: experiments

    Experiment objects

:Authors: bejar
    

:Version: 

:Created on: 23/03/2015 11:44 

"""


from util.Experiment import Experiment
from Config.paths import datapath

__author__ = 'bejar'

# Lista de conveniencia para poder procesar varios experimentos a la vez
lexperiments = ['e150514']

# Diccionario con los datos de cada experimento
# Todos los experimentos estan en el directorio cinvesdata dentro de una carpeta que tenga el nombre del experimento
experiments = \
    {
        'e160317':
            Experiment(
                dpath=datapath,
                name='e160317',
                sampling=10204.08,
                datafiles=['16317f13', '16317f14', '16317f15', '16317f16', '16317f17', '16317f18', '16317f19', '16317f20',
                           '16317f21', '16317f22', '16317f23', '16317f24', '16317f25', '16317f26', '16317f27', '16317f28',
                           '16317f29', '16317f30', '16317f31', '16317f32', '16317f33', '16317f34', '16317f35', '16317f36',
                           '16317f38', '16317f39', '16317f40', '16317f41', '16317f42', '16317f43', '16317f44', '16317f45',
                           '16317f46', '16317f47', '16318f00', '16318f01', '16318f03', '16318f04', '16318f05', '16318f06',
                           '16318f07', '16318f08'],
                sensors=['L4ci', 'L5ri', 'L5rd', 'L5cd', 'L5ci', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri',
                         'L7rd'],
                abfsensors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                extrasensors=[(12, 'IFPs'), (13, 'IFPp')],
                clusters=[12] * 11,  # [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                colors='y' * 6 + 'r' * 18 + 'b' * 12 + 'k' * 6,
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 20},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['crtl{:0>2}'.format(i) for i in range(1, 7)] + ['capsa{:0>2}'.format(i) for i in
                                                                           range(1, 19)] +
                         ['lido{:0>2}'.format(i) for i in range(1, 13)] + ['esp{:0>2}'.format(i) for i in
                                                                            range(1, 7)]
            ),

        'e160204':
            Experiment(
                dpath=datapath,
                name='e160204',
                sampling=10204.08,
                datafiles=['16204f02', '16204f03', '16204f04', '16204f05', '16204f06', '16204f07', '16204f08', '16204f09',
                            '16204f10', '16204f11', '16204f12', '16204f13', '16204f14', '16204f15', '16204f16', '16204f17',
                            '16204f18', '16204f19', '16204f20', '16204f21', '16204f22', '16204f23', '16204f24', '16204f25',
                            '16204f26', '16204f27', '16204f28', '16204f29', '16204f30', '16204f31', '16204f32', '16204f33',
                            '16204f34', '16204f35', '16204f36', '16204f37', '16204f38', '16204f39', '16204f40', '16204f41',
                            '16204f42', '16204f43', '16204f44', '16204f45', '16204f46', '16204f47', '16204f48', '16204f49',
                            '16204f50', '16204f51', '16204f52', '16204f53', '16204f54', '16204f55', '16204f56', '16204f57',
                            '16204f58', '16204f59', '16204f60', '16204f61', '16204f62', '16204f63', '16204f64', '16204f65',
                            '16204f66', '16204f67', '16204f68', '16204f69', '16204f70', '16204f71', '16204f72', '16204f73',
                            '16204f74', '16204f75'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5cd', 'L5ci', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri', 'L7rd'],
                abfsensors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                extrasensors = [(12, 'IFPs'), (13, 'IFPp')],
                clusters=[12] * 12,  # [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                colors='r' * 11 + 'g' * 19 + 'b' * 20 + 'k' * 6 + 'b' * 18,
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 20},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames= ['crtl{:0>2}'.format(i) for i in range(1,12)] + ['capsa{:0>2}'.format(i) for i in range(1,20)]+
                          ['lido1{:0>2}'.format(i) for i in range(1,21)] + ['esp{:0>2}'.format(i) for i in range(1,7)]+
                          ['lido2{:0>2}'.format(i) for i in range(1,19)]
            ),

        'e151126':
            Experiment(
                dpath=datapath,
                name='e151126',
                sampling=10204.08,
                datafiles=['15n26009', '15n26010', '15n26014', '15n26015', '15n26016', '15n26017',
                           '15n26018', '15n26019', '15n26023', '15n26024', '15n26025', '15n26026',
                           '15n26027', '15n26031', '15n26032', '15n26033', '15n26034', '15n26035',
                           '15n27000', '15n27001', '15n27005', '15n27006', '15n27007', '15n27008',
                           '15n27009', '15n27013', '15n27014', '15n27015', '15n27016', '15n27017',
                           '15n27018', '15n27022', '15n27023', '15n27024', '15n27025', '15n27026',
                           '15n27027', '15n27031', '15n27032', '15n27033', '15n27034', '15n27035',
                           '15n27036', '15n27040', '15n27041', '15n27042', '15n27043', '15n27044',
                           '15n27045', '15n27046', '15n27047', '15n27051', '15n27052', '15n27053',
                           '15n27054', '15n27055', '15n27056', '15n27060', '15n27061', '15n27062',
                           '15n27066'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri', 'L7rd'],
                abfsensors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                extrasensors = [(12, 'IFPs'), (13, 'IFPp')],
                clusters=[12] * 12, #[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                colors='r'*2 + 'g' * 24 + 'b' * 11 + 'y'*10 + 'c'*7 + 'm'* 6 + 'k',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.02},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 20},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['anI{:0>2}'.format(i) for i in range(1, 3)] + ['anII{:0>2}'.format(i) for i in range(1, 25)] +
                        ['anIII{:0>2}'.format(i) for i in range(1, 12)] + ['anIV{:0>2}'.format(i) for i in range(1, 11)] +
                        ['anV{:0>2}'.format(i) for i in range(1, 8)] + ['anVI{:0>2}'.format(i) for i in range(1, 71)] +
                        ['anVII{:0>2}'.format(i) for i in range(1, 2)]
            ),

        'e150707':
            Experiment(
                dpath=datapath,
                name='e150707',
                sampling=10204.08,
                datafiles=['15707000', '15707001', '15707002', '15707003', '15707004', '15707005', '15707006', '15707007',
                           '15707008', '15707009', '15707010', '15707011', '15707012', '15707013', '15707014', '15707015',
                           '15707016', '15707017', '15707018', '15707019', '15707020', '15707021', '15707022', '15707023',
                           '15707024', '15707025', '15707026', '15707027', '15707028', '15707029', '15707030', '15707031',
                           '15707032', '15707035', '15707036', '15708000', '15708001', '15708002', '15708003', '15708004',
                           '15708005', '15708006', '15708007'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri', 'L7rd'],
                abfsensors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                extrasensors = [(12, 'IFPs'), (13, 'IFPp')],
                clusters=[12]* 12, #[7, 6, 7, 7, 7, 7, 6, 6, 7, 6, 5, 7],
                colors='ggggyyyyyyyyyyrrrrrrrrrbbbbbbbbbbbkkkccccccc',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 20},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3', 'ctrl4', 'lido11', 'lido12', 'lido13', 'lido14', 'lido15', 'lido16',
                          'lido17', 'lido18', 'lido19', 'lido110', 'capsa1', 'capsa2', 'capsa3', 'capsa4', 'capsa5',
                          'capsa6', 'capsa7', 'capsa8', 'capsa9', 'lido21', 'lido22', 'lido23', 'lido24', 'lido25',
                          'lido26', 'lido27', 'lido28', 'lido29', 'lido210', 'esp1', 'esp2', 'esp3', 'lido31', 'lido32',
                          'lido33', 'lido34', 'lido35', 'lido36', 'lido37']

            ),

        'e150514':
            Experiment(
                dpath=datapath,
                name='e150514',
                sampling=10204.08,
                datafiles=['15514005', '15514006', '15514007', '15514008', '15514009', '15514010', '15514011',
                           '15514012', '15514013', '15514014', '15514015', '15514016', '15514017', '15514018',
                           '15514019', '15514020', '15514021', '15514022', '15514023', '15514024', '15514025',
                           '15514026', '15514027', '15514028', '15514029', '15514030', '15514031', '15514032',
                           '15514033', '15514034', '15514035', '15514036', '15514037', '15514038'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                extrasensors = [(12, 'IFPs'), (13, 'IFPp')],
                clusters= [12]*12, #[7, 8, 10, 8, 9, 7, 7, 6, 10, 6, 6, 8],
                colors='gggrrrrrrrrrbbbbbbbbbbbkkyyyyyyyyy',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 20},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctr1', 'ctr2', 'ctr3',
                        'cap1', 'cap2', 'cap3', 'cap4', 'cap5', 'cap6',
                        'cap7', 'cap8', 'cap9',
                        'lido1', 'lido2', 'lido3', 'lido4', 'lido5',
                        'lido6', 'lido7', 'lido8', 'lido9', 'lido10', 'lido11',
                        'esp1', 'esp2', 'lido21',
                        'lido22', 'lido23', 'lido24', 'lido25', 'lido26',
                        'lido27', 'lido28', 'lido29']
        
            ),

        'e150514alt':
            Experiment(
                dpath=datapath,
                name='e150514alt',
                sampling=10204.08,
                datafiles=['15514005', '15514006', '15514007', '15514008', '15514009', '15514010', '15514011',
                           '15514012', '15514013', '15514014', '15514015', '15514016', '15514017', '15514018',
                           '15514019', '15514020', '15514021', '15514022', '15514023', '15514024', '15514025',
                           '15514026', '15514027', '15514028', '15514029', '15514030', '15514031', '15514032',
                           '15514033', '15514034', '15514035', '15514036', '15514037', '15514038'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri',
                         'L7rd'],
                abfsensors=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                extrasensors=[(12, 'IFPs'), (13, 'IFPp')],
                clusters=[12] * 12,  # [7, 8, 10, 8, 9, 7, 7, 6, 10, 6, 6, 8],
                colors='gggrrrrrrrrrbbbbbbbbbbbkkyyyyyyyyy',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 20},
                peaks_alt_smooth={'lambda': 5, 'p': 0.9},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['ctr1', 'ctr2', 'ctr3',
                          'cap1', 'cap2', 'cap3', 'cap4', 'cap5', 'cap6',
                          'cap7', 'cap8', 'cap9',
                          'lido1', 'lido2', 'lido3', 'lido4', 'lido5',
                          'lido6', 'lido7', 'lido8', 'lido9', 'lido10', 'lido11',
                          'esp1', 'esp2', 'lido21',
                          'lido22', 'lido23', 'lido24', 'lido25', 'lido26',
                          'lido27', 'lido28', 'lido29']

            ),

        # ------------------------------------------------------------------------
        'e140220':
            Experiment(
                dpath=datapath,
                name='e140220',
                sampling=10416.7,
                datafiles=['e140220f8-ctrl1', 'e140220f10-ctrl2', 'e140220f12-ctrl3'],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3']
            ),

        'e140225':
            Experiment(
                dpath=datapath,
                name='e140225',
                sampling=10204.08,
                datafiles=['14225f31', '14225f32', '14225f33', '14225f34', '14225f36', '14225f37', '14225f38', '14225f39', '14225f47',
                        '14225f48', '14225f49', '14225f50', '14225f58', '14225f59', '14225f60', '14225f61', '14225f62', '14225f63',
                        '14225f64', '14225f65', '14225f66', '14225f67', '14225f68', '14225f76', '14225f77', '14225f78', '14225f79',
                        '14225f80', '14225f81',   '14225f82', '14225f83', '14225f84', '14225f85'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5cd', 'L5ci', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='ggggkkkkkkkkkkrrrrrrrrrrbbbbbbbbb',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['crtl1', 'crtl2', 'crtl3', 'crtl4', 'esp1', 'esp2', 'esp3', 'esp4', 'esp5', 'esp6', 'esp7',
                          'esp8', 'esp9', 'esp10', 'capsa1', 'capsa2', 'capsa3', 'capsa4', 'capsa5', 'capsa6',
                          'capsa7', 'capsa8', 'capsa9' ,'capsa10', 'lido1', 'lido2', 'lido3', 'lido4',
                          'lido5', 'lido6','lido7','lido8','lido9']
            ),

        'e140311':
            Experiment(
                dpath=datapath,
                name='e140311',
                sampling=10204.1,
                datafiles=['e140311f09-cntrl1', 'e140311f13-cntrl2', 'e140311f23-cntrl3'],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3']
            ),
        'e140911':
            Experiment(
                dpath=datapath,
                name='e140911',
                sampling=10416.7,
                datafiles=['e140911f20-cntrl1', 'e140911f33-cntrl2', 'e140911f36-cntrl3'],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3']
            ),
        'e141016':
            Experiment(
                dpath=datapath,
                name='e141016',
                sampling=10204.1,
                datafiles=['e141016f07', 'e141016f09', 'e141016f11'],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3']
            ),
        'e141029':
            Experiment(
                dpath=datapath,
                name='e141029',
                sampling=10204.1,
                datafiles=['e141029f35-cntrl1', 'e141029f37-cntrl2', 'e141029f39-cntrl3'],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3']
            ),
        'e141113':
            Experiment(
                dpath=datapath,
                name='e141113',
                sampling=10204.1,
                datafiles=['e141029f35-cntrl1', 'e141029f37-cntrl2', 'e141029f39-cntrl3'],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3']
            ),
        'e130221':
            Experiment(
                dpath=datapath,
                name='e130221',
                sampling=10416.66,
                datafiles=['13221f30', '13221f31', '13221f32', '13221f33', '13221f35', '13221f36', '13221f37', '13221f38',
                           '13221f46', '13221f47', '13221f48', '13221f49', '13221f57', '13221f58', '13221f59', '13221f60',
                           '13221f61', '13221f62', '13221f63', '13221f64', '13222f09', '13222f10', '13222f11', '13222f24',
                           '13222f25', '13222f26', '13222f27'],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10],
                clusters=[12]*11,
                colors='ggggkkkkrrrrrrrrbbbbbbbbbbb',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['crtl1', 'crtl2', 'crtl3', 'crtl4', 'esp1', 'esp2', 'esp3', 'esp4', 'capsa1', 'capsa2',
                          'capsa3','capsa4','capsa5','capsa6','capsa7','capsa8', 'lido1', 'lido2', 'lido3', 'lido4',
                          'lido5', 'lido6','lido7','lido8','lido9','lido10','lido11']
            ),
        'e130716':
            Experiment(
                dpath=datapath,
                name='e130716',
                sampling=10204.1,
                datafiles=['e130716f00-cntrl1', 'e130716f02-cntrl2', 'e130716f03-cntrl3'],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri', 'L7rd'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3']
            ),
        'e130827':
            Experiment(
                dpath=datapath,
                name='e130827',
                sampling=10256.4,
                datafiles=['13827f23_ctrl1', '13827f24_ctrl2', '13827f25_ctrl3', '13827f26_ctrl4', '13827f37_ctrl5',
                           '13827f38_cap1', '13827f39_cap2', '13827f40_cap4', '13827f41_cap5', '13827f42_cap6',
                           '13827f43_cap11', '13827f44_cap12', '13827f54_cap17', '13827f55_cap19'
                           ],
                sensors=['L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci', 'L7ri'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10],
                clusters=[12]*11,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3', 'ctrl4', 'ctrl5', 'cap1', 'cap2', 'cap4', 'cap5', 'cap6', 'cap11',
                          'cap12', 'cap17', 'cap19']
            ),
        'e130903':
            Experiment(
                dpath=datapath,
                name='e130903',
                sampling=10256.4,
                datafiles=['e130903f20-cntrl1', 'e130903f22-cntrl2', 'e130903f25-cntrl3'],
                sensors=['L4ri', 'L4cd', 'L4ci', 'L5rd', 'L5ri', 'L5cd', 'L5ci', 'L6rd', 'L6ri', 'L6cd', 'L6ci'],
                abfsensors=[0,1,2,3,4,5,6,7,8,9,10,11],
                clusters=[12]*12,
                colors='rrr',
                peaks_id_params={'wtime': 120e-3, 'low':0, 'high':70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass':200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3']
            ),

        'e120503':
            Experiment(
                dpath=datapath,
                name='e120503',
                sampling=10416.66,
                datafiles=[
                    '12503f02', '12503f03', '12503f24', '12503f25', '12503f36', '12503f37', '12503f48', '12503f49',
                    '12503f61', '12503f62', '12503f73', '12503f74', '12503f85', '12503f86'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri'],
                abfsensors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                clusters=[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                colors='rryyyybbbggggg',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 20},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['ctrl1', 'ctrl2', 'lido11', 'lido12', 'lido13', 'lido14', 'capsa1', 'capsa2', 'capsa3',
                          'lido21', 'lido22', 'lido23', 'lido24', 'lido25']
            ),
        'e120511':
            Experiment(
                dpath=datapath,
                name='e120511',
                sampling=10204.0,
                datafiles=['12511000', '12511001', '12511005', '12511006', '12511010', '12511015', '12511019',
                           '12511020', '12511024', '12511025'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri'],
                abfsensors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                clusters=[12] * 11,
                colors='rryyyykbbbb',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['ctrl1', 'ctrl2', 'capsa1', 'capsa2', 'capsa3', 'capsa4',
                          'sec1', 'lido1', 'lido2', 'lido3', 'lido4']
            ),
        'e120511e':
            Experiment(
                dpath=datapath,
                name='e120511e',
                sampling=10204.0,
                datafiles=['12511000', '12511001', '12511005', '12511006', '12511010', '12511015', '12511019',
                           '12511020', '12511024', '12511025'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri'],
                abfsensors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                clusters=[12] * 11,
                colors='rryyyykbbbb',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['ctrl1', 'ctrl2', 'capsa1', 'capsa2', 'capsa3', 'capsa4',
                          'sec1', 'lido1', 'lido2', 'lido3', 'lido4']
            ),

        'e110616':
            Experiment(
                dpath=datapath,
                name='e110616',
                sampling=10416.08,
                datafiles=['11616f08', '11616f16', '11616f18', '11616f29', '11616f30', '11616f31', '11616f39',
                           '11616f40',
                           '11616f49', '11616f50', '11616f51', '11616f59', '11617f00', '11617f08', '11617f09'],
                sensors=['L4ci', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri'],
                abfsensors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                clusters=[12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                colors='rrryyyyybbbbbbb',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 20},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['ctrl1', 'ctrl2', 'ctrl3', 'capsa1', 'capsa2', 'capsa3', 'capsa4', 'capsa5',
                          'lido1', 'lido2', 'lido3', 'lido4', 'lido5', 'lido6', 'lido7']
            ),
        'e110906':
            Experiment(
                dpath=datapath,
                name='e110906',
                sampling=1670.08,
                datafiles=['11906001', '11906002', '11906028', '11906061', '11906087', '11906118',
                           '11906119', '11906145', '11906172', '11906173', '11906210'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri'],
                abfsensors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                clusters=[15] * 11,
                colors='rryyybbbbbb',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 1.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['ctrl1', 'ctrl2', 'capsa1', 'capsa2', 'capsa3',
                          'lido1', 'lido2', 'lido3', 'lido4', 'lido5', 'lido6']
            ),
        'e110906o':
            Experiment(
                dpath=datapath,
                name='e110906o',
                sampling=10204.08,
                datafiles=['11906001', '11906002', '11906028', '11906061', '11906087', '11906118',
                           '11906119', '11906145', '11906172', '11906173', '11906210'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri'],
                abfsensors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                extrasensors = [(12, 'IFPs'), (13, 'IFPp')],
                clusters=[15] * 11,
                colors='ggrrrbbbbbb',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['Control1', 'Control2', 'Capsaicin1', 'Capsaicin2', 'Capsaicin3',
                          'Lidocaine1', 'Lidocaine2', 'Lidocaine3', 'Lidocaine4', 'Lidocaine5', 'Lidocaine6']
            ),
        'e110906e':
            Experiment(
                dpath=datapath,
                name='e110906e',
                sampling=10204.08,
                datafiles=['11906001', '11906002', '11906028', '11906061', '11906087', '11906118',
                           '11906119', '11906145', '11906172', '11906173', '11906210'],
                sensors=['L4ci', 'L4cd', 'L5ri', 'L5rd', 'L5ci', 'L5cd', 'L6ri', 'L6rd', 'L6ci', 'L6cd', 'L7ri'],
                abfsensors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                clusters=[12] * 11,
                colors='rryyybbbbbb',
                peaks_id_params={'wtime': 120e-3, 'low': 0, 'high': 70, 'threshold': 0.05},
                peaks_resampling={'wtsel': 100, 'rsfactor': 6.0, 'filtered': False},
                peaks_smooth={'pcasmooth': True, 'components': 10, 'wbaseline': 10},
                peaks_filter={'lowpass': 1.0, 'highpass': 200.0},
                expnames=['Control1', 'Control2', 'Capsaicin1', 'Capsaicin2', 'Capsaicin3',
                          'Lidocaine1', 'Lidocaine2', 'Lidocaine3', 'Lidocaine4', 'Lidocaine5', 'Lidocaine6']
            ),

    }

if __name__ == '__main__':
    experiment = experiments['e160204']
    print(len(experiment.expnames))

