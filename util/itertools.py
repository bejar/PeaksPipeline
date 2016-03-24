"""
.. module:: itertools

itertools
******

:Description: itertools

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  24/03/2016
"""

__author__ = 'bejar'

def batchify(work, batchsize):
    """
    Returns the list in work as a list that ha sublists of size batchzise
    :param work:
    :param batchsize:
    :return:
    """
    batches = []
    for i in range(0, len(work), batchsize):
        batches.append(work[i: i + batchsize])
    return batches