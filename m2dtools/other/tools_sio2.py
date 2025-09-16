import sys
import numpy as np
import numpy.linalg as LA
import re
import scipy.constants as scc
import pandas as pd

def bond_angle_SiOSi(type_atoms, CN_idx, vectors):
    '''
    CN_idx stores ...
    vectors stores n*n*3 matrix, diff
    '''
    bond_angle = []
    for i in np.argwhere(type_atoms == 'O')[:, 0]:
        for j1 in range(CN_idx[i].shape[0]):
            for j2 in np.arange(j1+1, CN_idx[i].shape[0]):
                a1 = vectors[int(CN_idx[i][j1]), i, :]
                a2 = vectors[int(CN_idx[i][j2]), i, :]
                cos_tmp = np.dot(a1, a2)/LA.norm(a1)/LA.norm(a2)
#                 if cos_tmp>1:
#                     cos_tmp=1
#                 elif cos_tmp<-1
#                     cos_tmp=-1
                bond_angle.append(np.arccos(cos_tmp)/np.pi*180)
    bond_angle = np.array(bond_angle)
    bond_angle = bond_angle[np.logical_not(np.isnan(bond_angle))]

    return bond_angle


def bond_angle_OSiO(type_atoms, CN_idx, vectors):
    """
    CN_idx stores ...
    vectors stores n*n*3 matrix, diff
    """
    bond_angle = []
    for i in np.argwhere(type_atoms == 'Si')[:, 0]:
        for j1 in range(CN_idx[i].shape[0]):
            for j2 in range(CN_idx[i].shape[0]):
                a1 = vectors[int(CN_idx[i][j1]), i, :]
                a2 = vectors[int(CN_idx[i][j2]), i, :]
                cos_tmp = np.dot(a1, a2)/LA.norm(a1)/LA.norm(a2)

                bond_angle.append(np.arccos(cos_tmp)/np.pi*180)
    bond_angle = np.array(bond_angle)
    bond_angle = bond_angle[np.logical_not(np.isnan(bond_angle))]

    return bond_angle


def bond_length_SiO(type_atoms, CN_idx, vectors):
    """
    """
    bond_length = []
    for i in np.argwhere(type_atoms == 'Si')[:, 0]:
        for j1 in range(CN_idx[i].shape[0]):
            a1 = vectors[int(CN_idx[i][j1]), i, :]
            bond_length.append(np.sum(a1**2)**0.5)

    return bond_length


def density_SiO2(box, type_atom):
    density = (np.sum(type_atom == 'Si')*28.084 + np.sum(type_atom == 'O')*15.999)/scc.Avogadro/np.dot(np.cross(box[:, 0], box[:, 1]), box[:, 2])*1e21
    return density

