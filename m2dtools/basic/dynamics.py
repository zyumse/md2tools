"""
Module for computing mean-squared displacement (MSD) with periodic boundary conditions
"""

import numpy as np


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


def compute_D(time, msd, fit_from=0, dim=3):
    """
    Compute diffusion coefficient from MSD using linear fit.

    Parameters:
    -----------
    time : np.ndarray
        Time array corresponding to the MSD values.
    msd : np.ndarray
        Mean-squared displacement array.
    fit_from : int
        Index to start the linear fit from.
    dim : int
        Dimensionality of the system (default is 3 for 3D).

    Returns:
    --------
    D : float
        Diffusion coefficient.
    """
    coeffs = np.polyfit(time[fit_from:], msd[fit_from:], 1)
    D = coeffs[0] / 2 / dim # in unit of msd / time
    return D


def random_vectors(length, num_vectors):
    phi = np.random.uniform(0, 2*np.pi, size=num_vectors)
    costheta = np.random.uniform(-1, 1, size=num_vectors)
    theta = np.arccos(costheta)

    x = length * np.sin(theta) * np.cos(phi)
    y = length * np.sin(theta) * np.sin(phi)
    z = length * np.cos(theta)

    return np.column_stack((x, y, z))   # shape = (num_vectors, 3)

#compute self-intermediate scattering function
def compute_self_intermediate_scattering_function(positions, box_lengths, lag_array, k, num_vectors=100, n_repeat=100):
    """
    Compute self-intermediate scattering function (SISF) considering periodic boundary conditions.

    Parameters:
    -----------
    positions : np.ndarray
        Atom positions array of shape (n_frames, n_atoms, 3).
    box_lengths : np.ndarray or list
        Simulation box lengths of shape (n_frames, 3).
    lag_array : np.ndarray or list
        Array of lag times for which to compute the SISF.
    k : absolute value of wave vector
    num_vectors : int
        Number of random k-vectors to average over.
    n_repeat : int
        Number of time origins to average over.
    Returns:
    --------
    sisf : np.ndarray
        Self-intermediate scattering function as 1D array of length max_lag.
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

    sisf = np.zeros(len(lag_array))
    vectors=random_vectors(k, num_vectors)  
    for i, lag in enumerate(lag_array):
        if len(positions)-lag < n_repeat:
            displacements = unwrapped_pos[lag:] - unwrapped_pos[:-lag]
            #compute cos(k*r) for each random vector and average
            cos_kr = np.cos(np.einsum('ij,tkj->tki', vectors, displacements))
        else:
            random_indices = np.random.choice(len(positions)-lag, n_repeat)
            displacements = unwrapped_pos[lag + random_indices] - unwrapped_pos[random_indices]
            cos_kr = np.cos(np.einsum('ij,tkj->tki', vectors, displacements))
        sisf[i] = np.mean(cos_kr)
    return sisf