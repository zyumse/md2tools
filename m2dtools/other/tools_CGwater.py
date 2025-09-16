import numpy as np
import pandas as pd
import datetime
import zy_md.my_common as mc
import zy_md.tools_lammps as tl_lmp
import pdb


def dipole_dipole_correlation(positions, dipoles, L_list):
    """
    Compute the dipole-dipole correlation function.

    Parameters:
    - positions: (N, 3) array of particle positions.
    - dipoles: (N, 3) array of particle dipole moments.
    - box_size: [Lx, Ly, Lz] box size.

    Returns:
    - distances: (N*(N-1)/2,) array of pairwise distances.
    - product_cos_theta: (N*(N-1)/2,) array of pairwise product of cos(theta_ij).
    """
    N = len(positions)
    box_size = [L_list[0][1]-L_list[0][0],
                L_list[1][1]-L_list[1][0],
                L_list[2][1]-L_list[2][0]]
    box = np.array([[box_size[0], 0, 0],
                    [0, box_size[1], 0],
                    [0, 0, box_size[2]]])

    # Compute the distances with periodic boundary conditions
    distances = tl_lmp.distance_pbc(positions, positions, box=box)

    cos_theta = np.einsum('ij,kj->ik', dipoles, dipoles) / (np.linalg.norm(dipoles, axis=1)[:, None] * np.linalg.norm(dipoles, axis=1))
    
    # Calculate the pairwise product of cos(theta_ij)
    product_cos_theta = cos_theta.flatten()
    distances = distances.flatten()
    # remove prodcut_cos_theta where distance is zero
    product_cos_theta = product_cos_theta[distances != 0]
    distances = distances[distances != 0]
    
    return distances, product_cos_theta


def pdf_sq_1type(box, natom, type_atom, coors, r_cutoff=10, delta_r=0.01):
    """
    only one type of particles
    inputs: box,natom,type_atom,coors,r_cutoff=10,delta_r = 0.01
    outputs: R,g1,Q,S1
    """
    type_atom = np.array(type_atom)
    n1 = natom
    rcoors = np.dot(coors, np.linalg.inv(box))
    rdis = np.zeros([natom, natom, 3])
    for i in range(natom):
        tmp = rcoors[i]
        rdis[i, :, :] = tmp - rcoors
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))
    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1 / V
    c = np.array([rho1 * rho1]) * V
    g1 = np.histogram(dis[:n1, :n1], bins=r)[0] / (4 * np.pi *
                                                   (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def pdf_sq_cross(box, coors1, coors2, r_cutoff=10, delta_r=0.01):
    """
    """
    # type_atom = np.array(type_atom)
    n1 = len(coors1)
    n2 = len(coors2)
    natom = n1 + n2
    rcoors1 = np.dot(coors1, np.linalg.inv(box))
    rcoors2 = np.dot(coors2, np.linalg.inv(box))
    rdis = np.zeros([n1, n2, 3])
    for i in range(n1):
        tmp = rcoors1[i]
        rdis[i, :, :] = tmp - rcoors2
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))
    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1 / V
    c = np.array([rho1 * rho1]) * V
    g1 = np.histogram(dis[:n1, :n1], bins=r)[0] / (4 * np.pi *
                                                   (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) 
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def read_log_lammps(logfile):
    f = open(logfile, 'r')
    L = f.readlines()
    f.close()
    l1_list, l2_list = [], []
    for i in range(len(L)):
        if ('Step' in L[i]) and ('Temp' in L[i]):
            l1_list.append(i)
        if 'Loop time' in L[i]:
            l2_list.append(i)
    data_list = []
    for i, l1 in enumerate(l1_list):
        l2 = l2_list[i]
        data = np.array(L[l1+1].split())
        for i in range(l1+1, l2):
            data = np.vstack((data, L[i].split()))
        data = pd.DataFrame(data, dtype='float64', columns=L[l1].split())
        data_list.append(data)
    return data_list


def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # convert to angle degree
    angle = np.arccos(cos_angle)/np.pi*180
    return angle


def angle_distribution(coors, box, cutoff):
    """
    compute O-O-O angle distribution in water within a cutoff distance
    """
    n_atom = coors.shape[0]
    angles = []
    for i in range(n_atom):
        # find atoms in a small cubic neighborhood of i
        coors_i = coors[i]
        rcoors = coors - coors_i
        rcoors -= np.rint(rcoors @ np.linalg.inv(box)) @ box
        idx = np.argwhere((rcoors[:, 0] > -cutoff)*(rcoors[:, 0] < cutoff)*(rcoors[:, 1] > -cutoff)*(
            rcoors[:, 1] < cutoff)*(rcoors[:, 2] > -cutoff)*(rcoors[:, 2] < cutoff)).flatten()
        # loop over idx but avoid i itself
        for j in idx:
            for k in idx:
                if j < k and j != i and k != i:
                    rij = coors[j] - coors[i]
                    rik = coors[k] - coors[i]
                    rij -= np.rint(rij @ np.linalg.inv(box)) @ box
                    rik -= np.rint(rik @ np.linalg.inv(box)) @ box
                    if np.linalg.norm(rij) < cutoff and np.linalg.norm(rik) < cutoff:
                        angle = calculate_angle(rij, rik)
                        angles.append(angle)
    return angles


def tetrahedral_order_parameter(positions, box):
    """Calculate the tetrahedral order parameter for a central molecule and its four neighbors."""
    q = 0
    for j in range(1, 4):
        for k in range(j + 1, 5):
            # Compute vectors from central atom to each neighbor
            vj = positions[j] - positions[0]
            vj = vj - np.rint(vj @ np.linalg.inv(box)) @ box
            vk = positions[k] - positions[0]
            vk = vk - np.rint(vk @ np.linalg.inv(box)) @ box
            cos_theta = np.dot(vj, vk) / (np.linalg.norm(vj) * np.linalg.norm(vk))
            # pdb.set_trace()
            q += (cos_theta + 1/3) ** 2
    q = 1 - (3/8) * q
    return q


def compute_tetrahedral_order(coors, box, cutoff=3.5):
    """
    return q_list, the tetrahedral order parameter for each atom
    """
    n_atom = coors.shape[0]
    q_list = []
    CN, CN_idx, CN_dist, diff = mc.CN(box=box, coors=coors, cutoff=cutoff)

    for i in range(n_atom):
        if CN[i] < 4:
            print('Atom {} has less than 4 neighbors, its CN is {}'.format(i, CN[i]))
            continue
        #     raise ValueError('Atom {} has less than 4 neighbors'.format(i))
        positions = []
        positions.append(coors[i].flatten())
        # sort CN_idx based on CN_dist
        CN_idx[i] = CN_idx[i][np.argsort(CN_dist[i])]
        if CN[i] >= 4:
            for j in range(4):
                positions.append(coors[CN_idx[i][j]].flatten())
        q_tmp = tetrahedral_order_parameter(positions, box)
        # pdb.set_trace()
        q_list.append(q_tmp)
    return q_list


def write_pot_CGwater_bond(file_name, r, E_OO, F_OO, E_MM, F_MM, E_MO, F_MO,
                           r_bond, E_MO_bond, F_MO_bond):
    """
    write the two-body potential to a table file for lammps simulations
    it is for now only for the CG water model (2p with dipole information)
    with bonded interaction
    """
    t = datetime.date.today().strftime('%m/%d/%Y')
    f = open(file_name, 'w')
    f.write('# DATE: {}  UNITS: real \n'.format(t))
    f.write('# potential for CG2p \n\n')
    f.write('OO\n')
    f.write('N {0:} R {1:.5f} {2:.5f}\n'.format(len(r), r[0], r[-1]))
    f.write('\n')
    for i in range(len(r)):
        f.write('   {0:5d}      {1:12.8f}{2:21.12e}{3:21.12e}\n'.format(
            i+1, r[i], E_OO[i], F_OO[i]))

    f.write('\n')
    f.write('MM\n')
    f.write('N {0:} R {1:.5f} {2:.5f}\n'.format(len(r), r[0], r[-1]))
    f.write('\n')
    for i in range(len(r)):
        f.write('   {0:5d}      {1:12.8f}{2:21.12e}{3:21.12e}\n'.format(
            i+1, r[i], E_MM[i], F_MM[i]))

    f.write('\n')
    f.write('MO\n')
    f.write('N {0:} R {1:.5f} {2:.5f}\n'.format(len(r), r[0], r[-1]))
    f.write('\n')
    for i in range(len(r)):
        f.write('   {0:5d}      {1:12.8f}{2:21.12e}{3:21.12e}\n'.format(
            i+1, r[i], E_MO[i], F_MO[i]))

    f.write('\n')
    f.write('MO_bond\n')
    f.write('N {0:}\n'.format(len(r_bond)))
    f.write('\n')
    for i in range(len(r_bond)):
        f.write('   {0:5d}      {1:12.8f}{2:21.12e}{3:21.12e}\n'.format(
            i+1, r_bond[i], E_MO_bond[i], F_MO_bond[i]))

    f.close()


def write_OOOangle_table_CGwater(file_name, angles, e_pot, f_pot):
    """
    Write the angle distribution to a file.
    """
    with open(file_name, 'w') as f:
        f.write(f"{len(angles)}\n")
        f.write(f"O O O\n")
        for angle, e, force in zip(angles, e_pot, f_pot):
            f.write(f"{angle:.6f} {e:.6f} {force:.6f}\n")
