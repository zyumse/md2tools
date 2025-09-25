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

def compute_self_intermediate_scattering_function_mask(
    positions, box_lengths, lag_array, k, num_vectors=100, n_repeat=100, atom_mask=None
):
    """
    Compute self-intermediate scattering function (SISF) with optional atom masking, 
    considering periodic boundary conditions (orthorhombic box).

    Parameters
    ----------
    positions : np.ndarray
        Shape (n_frames, n_atoms, 3). Wrapped coordinates per frame.
    box_lengths : np.ndarray
        Shape (n_frames, 3). Box lengths per frame.
    lag_array : array-like of int
        Lags in *frames*. (Convert to ps for plotting via t_ps = lag * dt.)
    k : float
        Absolute value of the wave vector |k|.
    num_vectors : int
        Number of random k-directions to average over.
    n_repeat : int
        Number of time origins t0 to sample when there are many possible origins.
    atom_mask : None or np.ndarray
        If None: use all atoms.
        If 1D (n_atoms,): time-independent mask; True atoms are included for all t0.
        If 2D (n_frames, n_atoms): time-dependent mask; atom j is included only when atom_mask[t0, j] is True.
        In both cases, only masked atoms contribute to the SISF average.

    Returns
    -------
    sisf : np.ndarray
        SISF values for each lag in lag_array (same length).
    """
    n_frames, n_atoms, _ = positions.shape

    # ---- unwrap positions under PBC (orthorhombic) ----
    unwrapped_pos = np.zeros_like(positions)
    unwrapped_pos[0] = positions[0]
    for t in range(1, n_frames):
        box_length_tmp = (box_lengths[t] + box_lengths[t - 1]) / 2.0
        delta = positions[t] - unwrapped_pos[t - 1]
        # minimum image in fractional coordinates, then back to Cartesian
        delta -= box_length_tmp * np.round(delta / box_length_tmp)
        unwrapped_pos[t] = unwrapped_pos[t - 1] + delta

    # ---- pre-generate random k-directions (shape: (num_vectors, 3)) ----
    vectors = random_vectors(k, num_vectors)

    # ---- validate mask shape (if provided) ----
    if atom_mask is not None:
        if atom_mask.ndim == 1:
            if atom_mask.shape[0] != n_atoms:
                raise ValueError("atom_mask 1D must have shape (n_atoms,).")
        elif atom_mask.ndim == 2:
            if atom_mask.shape != (n_frames, n_atoms):
                raise ValueError("atom_mask 2D must have shape (n_frames, n_atoms).")
        else:
            raise ValueError("atom_mask must be None, 1D (n_atoms,), or 2D (n_frames, n_atoms).")

    sisf = np.zeros(len(lag_array), dtype=float)

    for i, lag in enumerate(lag_array):
        if lag <= 0 or lag >= n_frames:
            sisf[i] = np.nan
            continue

        # choose time origins t0
        n_origins_total = n_frames - lag
        if n_origins_total <= 0:
            sisf[i] = np.nan
            continue

        if n_origins_total <= n_repeat:
            # use all possible time origins
            origins = np.arange(n_origins_total)  # shape (O,)
        else:
            # random subset of origins
            origins = np.random.choice(n_origins_total, size=n_repeat, replace=False)

        # displacements for these origins: shape (O, n_atoms, 3)
        disp = unwrapped_pos[origins + lag] - unwrapped_pos[origins]

        # compute k·dr for all vectors and atoms/origins
        # vectors: (V,3), disp: (O,A,3) -> dots: (V,O,A)
        dots = np.einsum('vj,oaj->voa', vectors, disp)  # (num_vectors, n_origins, n_atoms)
        cos_kr = np.cos(dots)  # (V,O,A)

        if atom_mask is None:
            # simple average over vectors, origins, atoms
            sisf[i] = np.mean(cos_kr)

        elif atom_mask.ndim == 1:
            # time-independent selection of atoms
            sel = atom_mask.astype(bool)
            if not np.any(sel):
                sisf[i] = np.nan
                continue
            cos_sel = cos_kr[:, :, sel]  # (V,O,A_sel)
            sisf[i] = np.mean(cos_sel)

        else:
            # time-dependent selection per origin: mask[t0, atom]
            # build a (O, A) mask for these origins
            sel_oa = atom_mask[origins, :]  # (O, A), boolean
            counts = sel_oa.sum()
            if counts == 0:
                sisf[i] = np.nan
                continue
            # weight average only over True entries
            # expand sel_oa to (V,O,A) for broadcasting
            sel_broadcast = sel_oa[None, :, :]  # (1,O,A)
            # multiply then divide by number of selected (origins, atoms) per vector set
            # globally average: sum over V,O,A then divide by (V * selected OA count)
            num = (cos_kr * sel_broadcast).sum()
            den = float(num_vectors) * float(counts)
            sisf[i] = num / den

    return sisf

