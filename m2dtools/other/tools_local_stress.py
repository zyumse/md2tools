import numpy as np

def midpoint_pbc(coords, idx_a, box):
    """
    Calculate the midpoint between two atoms in a bond considering PBC.

    Parameters:
    - coords: Array containing atomic coordinates (N x 3 where N is the number of atoms).
    - idx_a: Indices of atoms forming bonds (M x 2 where M is the number of bonds).
    - box: 3x3 matrix representing the simulation box.

    Returns:
    - Array containing midpoints (M x 3).
    """
    midpoints = []
    box_diag = np.diag(box)
    
    for idx in idx_a:
        r1 = coords[idx[0]]
        r2 = coords[idx[1]]
        # Calculate bond vector
        dr = r2 - r1
        # Apply minimum image convention
        dr = dr - np.round(dr / box_diag) * box_diag
        # Calculate midpoint
        midpoint = r1 + dr / 2
        # Make sure the midpoint is within the box
        midpoint = (midpoint + box_diag) % box_diag
        midpoints.append(midpoint)
    return np.array(midpoints)

def concatenate_xyz(filenames, output_filename="combined.xyz"):
    """
    Concatenate multiple XYZ files into a single file.

    Parameters:
    - filenames: List of names of XYZ files to concatenate.
    - output_filename: Name of the output concatenated XYZ file.
    """
    with open(output_filename, 'w') as outfile:
        for fname in filenames:
            with open(fname, 'r') as infile:
                # Copy content of infile to outfile
                for line in infile:
                    outfile.write(line)

def write_xyz_from_density(density, threshold, bin_size, filename="output.xyz"):
    """
    Write an XYZ file from a density array.

    Parameters:
    - density: 3D numpy array of density values.
    - threshold: minimum density value to include in the output.
    - filename: name of the output XYZ file.
    """
    # Identify points exceeding the threshold
    indices = np.where(density > threshold)
    densities_above_threshold = density[indices]
    
    # Open file for writing
    with open(filename, 'w') as f:
        # Write the number of atoms (pseudo-atoms)
        f.write(f"{len(densities_above_threshold)}\n")
        f.write("Atoms. Generated from density data.\n")
        
        # Loop through and write pseudo-atoms to the file
        for (ix, iy, iz), dens_value in zip(zip(*indices), densities_above_threshold):
            atom_type = "C"  # Using Carbon as a placeholder. Can be modified based on need.
            f.write(f"{atom_type} {ix*bin_size[0]:.5f} {iy*bin_size[1]:.5f} {iz*bin_size[2]:.5f} {dens_value:.5f}\n")
