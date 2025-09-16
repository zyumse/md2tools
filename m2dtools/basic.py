"""
Basic analysis functions for molecular dynamics simulations.
Functions
- compute_bond_length: Calculate bond lengths between bonded atoms.
- compute_angle: Calculate angles formed by three atoms.
- pdf_sq_1type: Compute pair distribution function (PDF) and structure factor (SQ) for one type of particles.
- pdf_sq_cross: Compute PDF and SQ for two types of particles, excluding bonded atoms.
- pdf_sq_cross_mask: Compute PDF and SQ for two types of particles with a mask matrix.
- CN: Calculate coordination numbers and neighbor indices within a cutoff distance.
- CN_large: Calculate coordination numbers for large systems using a cubic search method.
- angle_distribution: Compute angle distribution for triplets of atoms within a cutoff distance.
- compute_autocorrelation: Compute force autocorrelation function (FAF).
- compute_msd_pbc: Compute mean-squared displacement (MSD) considering periodic boundary conditions.
- calc_compressibility: Calculate compressibility from volume fluctuations.
"""
import numpy as np

def compute_bond_length(coors, bonded_atoms, box_size):
    vector = coors[bonded_atoms[:, 0]]-coors[bonded_atoms[:, 1]]
    vector = vector - np.rint(vector/box_size)*box_size
    bond_length = np.linalg.norm(vector, axis=1)
    return bond_length


def compute_angle(coors, angle_atoms, box_size):
    vector1 = coors[angle_atoms[:, 0]]-coors[angle_atoms[:, 1]]
    vector2 = coors[angle_atoms[:, 2]]-coors[angle_atoms[:, 1]]
    vector1 = vector1 - np.rint(vector1/box_size)*box_size
    vector2 = vector2 - np.rint(vector2/box_size)*box_size
    angle = np.arccos(np.sum(vector1*vector2, axis=1) /
                      (np.linalg.norm(vector1, axis=1)*np.linalg.norm(vector2, axis=1)))
    return angle/np.pi*180

def pdf_sq_1type(box_size, natom, coors, r_cutoff=10, delta_r=0.01):
    """
    only one type of particles
    inputs: box,natom,type_atom,coors,r_cutoff=10,delta_r = 0.01
    outputs: R,g1,Q,S1
    """
    box = np.array([[box_size, 0, 0],
                    [0, box_size, 0],
                    [0, 0, box_size]])
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
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def pdf_sq_cross_mask(box, coors1, coors2,  mask_matrix, r_cutoff:float=10, delta_r:float=0.01):
    """
    compute pdf and sq for two types of particles (can be same type)
    inputs: box,coors1,coors2, mask_matrix,r_cutoff=10,delta_r = 0.01
    outputs: R,g1,Q,S1
    """
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

    dis = dis * mask_matrix

    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1/V
    rho2 = n2/V
    c = np.array([rho1 * rho2]) * V
    g1 = np.histogram(dis, bins=r)[0] / (4 * np.pi * (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def pdf_sq_cross(box, coors1, coors2,  bond_atom_idx, r_cutoff:float=10, delta_r:float=0.01):
    """
    compute pdf and sq for two types of particles (can be same type)
    inputs: box,coors1,coors2,bond_atom_idx,r_cutoff=10,delta_r = 0.01
    outputs: R,g1,Q,S1
    """
    # check if coors1 and coors2 are exactly the same
    if np.array_equal(coors1, coors2):
        is_same = True
    else:
        is_same = False

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
    # Exclude bonded atoms by setting the distances to NaN
    if bond_atom_idx is not None:
        for bond_pair in bond_atom_idx:
            i1 = int(bond_pair[0])
            i2 = int(bond_pair[1])
            dis[i1, i2] = np.nan
            if is_same:
                dis[i2, i1] = np.nan

    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1/V
    rho2 = n2/V
    c = np.array([rho1 * rho2]) * V
    g1 = np.histogram(dis, bins=r)[0] / (4 * np.pi * (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def CN(box, coors, cutoff):
    """ 
    return CN, CN_idx, CN_dist, diff 
    """

    rcoors = np.dot(coors, np.linalg.inv(box))

    r1 = rcoors[:, np.newaxis, :]
    r2 = rcoors[np.newaxis, :, :]

    rdis = r1-r2

    while np.sum((rdis < -0.5) | (rdis > 0.5)) > 0:
        rdis[rdis < -0.5] = rdis[rdis < -0.5]+1
        rdis[rdis > 0.5] = rdis[rdis > 0.5]-1

    diff = np.dot(rdis, box)

    dis = np.sqrt(np.sum(np.square(diff), axis=2))

    CN_idx = []
    CN_dist = []
    CN = np.zeros(coors.shape[0])
    for i in range(coors.shape[0]):
        tmp = np.argwhere((dis[i, :] < cutoff) & (dis[i, :] > 0))
        CN[i] = tmp.shape[0]
        CN_idx.append(tmp)
        CN_dist.append(dis[i, (dis[i, :] < cutoff) & (dis[i, :] > 0)])
    return CN, CN_idx, CN_dist, diff


def CN_large(box, coors, cutoff):
    """
    return CN, CN_idx, CN_dist
    """
    CN = []
    CN_idx = []
    CN_dist = []
    for i in range(coors.shape[0]):
        # find atom in the cubic at the center of coord[1], within the cutoff
        coord0 = coors[i]
        diff_coord = coors - coord0
        # periodic boundary condition
        diff_coord = diff_coord - np.round(diff_coord @ np.linalg.inv(box) ) @ box
        idx_interest = np.argwhere((diff_coord[:, 0] >= -cutoff)*(diff_coord[:, 0] <= cutoff)*(diff_coord[:, 1] >= -cutoff)*(diff_coord[:, 1] <= cutoff)*(diff_coord[:, 2] >= -cutoff)*(diff_coord[:, 2] <= cutoff)).flatten()
        dist_tmp = np.linalg.norm(diff_coord[idx_interest,:], axis=1)
        idx_CN_tmp = np.argwhere(dist_tmp<=cutoff).flatten()
        CN.append(idx_CN_tmp.shape[0])
        CN_idx.append(idx_interest[idx_CN_tmp])
        CN_dist.append(dist_tmp[idx_CN_tmp])
    return CN, CN_idx, CN_dist


def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # convert to angle degree
    angle = np.arccos(cos_angle)/np.pi*180
    return angle


def angle_distribution(coors, box, cutoff):
    """
    compute O-O-O angle distribution in water within a cutoff distance
    inputs: coors, box, cutoff
    outputs: angles
    """
    n_atom = coors.shape[0]
    angles = []
    rcoors = np.dot(coors, np.linalg.inv(box))
    rdis = np.zeros([n_atom, n_atom, 3])
    for i in range(n_atom):
        tmp = rcoors[i]
        rdis[i, :, :] = tmp - rcoors
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))

    for i in range(n_atom):
        for j in np.arange(i+1, n_atom):
            for k in np.arange(j+1, n_atom):
                if dis[i, j] < cutoff and dis[i, k] < cutoff and dis[j, k] < cutoff:
                    angle = calculate_angle(a[j, i, :], a[k, i, :])
                    angles.append(angle)
                    angle = calculate_angle(a[i, j, :], a[k, j, :])
                    angles.append(angle)
                    angle = calculate_angle(a[i, k, :], a[j, k, :])
                    angles.append(angle)
    return angles


def compute_autocorrelation(forces, max_lag):
    """
    Compute the force autocorrelation function (FAF).

    Parameters:
    -----------
    forces : np.ndarray
        Array of shape (n_frames, n_atoms, 3), forces per atom.
    max_lag : int
        Maximum lag time (in frames) for computing the autocorrelation.

    Returns:
    --------
    faf : np.ndarray
        Force autocorrelation function as a 1D array of length max_lag.
    """
    n_frames, n_atoms, _ = forces.shape

    # Initialize autocorrelation array
    faf = np.zeros(max_lag)

    # Loop over lag times
    for lag in range(max_lag):
        # Compute dot product of forces separated by lag
        dot_products = np.sum(forces[:n_frames - lag] * forces[lag:], axis=(1, 2))
        faf[lag] = np.mean(dot_products)

    # Normalize by the zero-lag correlation
    faf /= faf[0]

    return faf


def compute_msd_pbc(positions, box_lengths, lag_array):
    """
    Compute mean-squared displacement (MSD) considering periodic boundary conditions.

    Parameters:
    -----------
    positions : np.ndarray
        Atom positions array of shape (n_frames, n_atoms, 3).
    box_lengths : np.ndarray or list
        Simulation box lengths of shape (n_frames, 3).
    lag_array : np.ndarray or list
        Array of lag times for which to compute the MSD.

    Returns:
    --------
    msd : np.ndarray
        Mean-squared displacement as 1D array of length max_lag.
    """
    n_frames, n_atoms, _ = positions.shape

    # Unwrap positions considering periodic boundary conditions
    unwrapped_pos = np.zeros_like(positions)
    unwrapped_pos[0] = positions[0]

    for t in range(1, n_frames):
        box_length_tmp = (box_lengths[t] + box_lengths[t - 1]) / 2
        delta = positions[t] - unwrapped_pos[t - 1]
        delta -= box_length_tmp * np.round(delta / box_length_tmp)
        unwrapped_pos[t] = unwrapped_pos[t - 1] + delta

    msd = np.zeros(len(lag_array))

    for i, lag in enumerate(lag_array):
        displacements = unwrapped_pos[lag:] - unwrapped_pos[:-lag]
        squared_displacements = np.sum(displacements**2, axis=2)
        msd[i] = np.mean(squared_displacements)

    return msd


def calc_compressibility(V, T=300):
    """
    Calculate compressibility from volume and temperature.
    :param V: Volume in nm^3
    :param T: Temperature in K
    :return: Compressibility in 1/Pa
    """
    kB = 1.380649e-23  # J/K
    V = V * 1e-27  # Convert from nm^3 to m^3
    kappa_T = np.var(V) / (V.mean() * kB * T)
    return kappa_T * 1e9 # to 1/GPa
