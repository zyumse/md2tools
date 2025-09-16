"""
A collection of common functions for basic MD process: supercell, read and write POSCAR, PDF, XYZ, progressbar, replace_in_file, read_ORCA
"""

import sys
import numpy as np


def supercell(natoms, box0, nx, ny, nz, index0, atom_type0, coors0):
    """
    at this moment, only for orthogonal cell
    """

    box_new = box0@np.array([[nx, 0, 0], [0, ny, 0], [0, 0, nz]])
    natoms_new = natoms*nx*ny*nz

    # coors_new = np.empty([1,3])
    # index_new = np.empty

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if ix+iy+iz == 0:
                    coors_new = coors0
                    atom_type_new = atom_type0
                    index_new = index0
                    index_tmp = index0
                else:
                    coors_tmp = coors0 + ix*box0[0, :] + iy*box0[1, :] + iz*box0[2, :]
                    coors_new = np.vstack((coors_new, coors_tmp))

                    atom_type_new = np.concatenate((atom_type_new, atom_type0))
                    index_tmp = index_tmp + natoms
                    index_new = np.concatenate((index_new, index_tmp))

    return natoms_new, box_new, index_new, atom_type_new, coors_new


def is_data_line(line):
    # Strip whitespace and check if the line is not empty
    if not line.strip():
        return False

    # Split line into parts and check if all parts are numeric
    parts = line.strip().split()
    return all(part.replace('.', '', 1).replace('-', '', 1).isdigit() for part in parts)


def read_pos(file_name):
    """
    read POSCAR format structure file for VASP calculations
    at this moment, only 'C' is applied
    """
    f = open(file_name, 'r')
    lf = list(f)
    f.close()
    box = np.zeros((3, 3))
    ratio = float(lf[1].split()[0])
    box[0, :] = np.array(lf[2].split()).astype(float)*ratio
    box[1, :] = np.array(lf[3].split()).astype(float)*ratio
    box[2, :] = np.array(lf[4].split()).astype(float)*ratio
    a_type = np.array(lf[5].split())
    num_type = np.array(lf[6].split()).astype(int)

    natom = np.sum(num_type)
    coors = np.zeros((natom, 3))

    if lf[7].split()[0] == 'C' or lf[7].split()[0] == 'c':
        l = 0
        for ia in lf[8:8+natom]:
            coors[l, :] = np.array(ia.split()[0:3:1]).astype('float')
            l += 1

    if lf[7].split()[0][0] == 'D' or lf[7].split()[0][0] == 'd':
        l = 0
        rcoors = np.zeros((natom, 3))
        for ia in lf[8:8+natom]:
            rcoors[l, :] = np.array(ia.split()[0:3:1]).astype('float')
            l += 1
        coors = rcoors @ box

    return box, a_type, num_type, coors


def write_pos(file_name, box, a_type, num_type, coors):
    """
    write POSCAR format structure file for VASP calculations
    input: file_name,box,a_type,num_type,coors
    """
    f = open(file_name, 'w')
    f.write('written by python script\n')
    f.write('1.0\n')

    for i in range(3):
        f.write('{0:20.12f}{1:20.12f}{2:20.12f}\n'.format(box[i, 0], box[i, 1], box[i, 2]))

    for i in range(len(a_type)):
        f.write(' {}'.format(a_type[i]))
    f.write('\n')
    for i in range(len(a_type)):
        f.write(' {}'.format(num_type[i]))
    f.write('\n')

    natom = np.sum(num_type)
    f.write('C\n')
    for i in range(natom):
        f.write('{0:20.12f}{1:20.12f}{2:20.12f}\n'.format(coors[i, 0], coors[i, 1], coors[i, 2]))

    f.close()


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
    g1 = np.histogram(dis[:n1, :n1], bins=r)[0] / (4 * np.pi * (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) / 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def write_xyz(file_name, natoms, type_atoms, coors):
    f = open('{}'.format(file_name), 'w')
    f.write('{0:5d}\n'.format(natoms))
    f.write('Generated by python script\n')
    for ia in range(natoms):
        f.write('{0:}     {1:.16f} {2:.16f} {3:.16f}\n'.format(type_atoms[ia], coors[ia, 0], coors[ia, 1], coors[ia, 2]))
    f.close()


def progressbar(it, prefix="", size=60, out=sys.stdout):  # Python3.3+
    count = len(it)

    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count),
              end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


def replace_in_file(file_path_old, file_path_new, old_string, new_string):
    """
    Replace all occurrences of old_string with new_string in the file at file_path.

    :param file_path: Path to the file where the replacement will be made.
    :param old_string: The string to be replaced.
    :param new_string: The new string that will replace old_string.
    """
    try:
        # Read the content of the file
        with open(file_path_old, 'r') as file:
            content = file.read()

        # Replace old_string with new_string
        modified_content = content.replace(old_string, new_string)

        # Write the modified content back to the file
        with open(file_path_new, 'w') as file:
            file.write(modified_content)

        print("File updated successfully.")
    except IOError as e:
        print(f"An error occurred: {e}")


def read_ORCA(filename, search_words, skip_lines, stop_words, dtype='object'):
    result = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith(search_words):
                for i in range(skip_lines):
                    file.readline()
                while True:
                    lc = file.readline()
                    if stop_words in lc:
                        break
                    result.append(lc.split())

    result = np.array(result, dtype=dtype)
    return result


def write_xyz_file(atom_types, coordinates, filename, comment=""):
    """
    Write an XYZ file from atom types and coordinates.

    Parameters:
    - atom_types (list of str): Atom types/symbols.
    - coordinates (list of list of floats): Coordinates of each atom.
    - filename (str): Path to the output XYZ file.
    - comment (str): A comment for the second line of the XYZ file.
    """
    num_atoms = len(atom_types)
    with open(filename, 'w') as file:
        file.write(f"{num_atoms}\n")
        file.write(f"{comment}\n")
        for atom_type, (x, y, z) in zip(atom_types, coordinates):
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")


def evaluate_linear_fit_np(x, y):
    """
    return r2 and MSE 
    """
    # Fit a linear model
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)

    # Predict y values
    y_pred = polynomial(x)

    # Calculate R2 score
    ss_res = np.sum((y - y_pred) ** 2)  # MSE
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return r2, np.sqrt(ss_res)
