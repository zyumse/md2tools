import numpy as np
from . import tools_lammps as tl
import datetime


def lmp_AA2CG_PEO(AA_file, CG_file):
    """
    """
    lmp = tl.read_lammps_full(AA_file)
    lmp.atom_info = lmp.atom_info[np.argsort(lmp.atom_info[:, 0])]

    # each molecule, find O and the two associated C onnected to it, group them as a CG bead
    # find the O and the two C connected to it
    # type 3 is O, type 2 and 4 are C, type 1 is H

    index_O = lmp.atom_info[lmp.atom_info[:, 2] == 3, 0]

    # find all end carbon atoms
    index_C_end = lmp.atom_info[lmp.atom_info[:, 2] == 2, 0]

    # based on lmp.bond_info, find the two C connected to O
    index_C_O = []
    CG_mapping = []
    type_CG = []
    mol_idx_CG = []
    for i in index_O:
        C1 = lmp.bond_info[lmp.bond_info[:, 2] == i, 3]
        C2 = lmp.bond_info[lmp.bond_info[:, 3] == i, 2]
        index_C_O.append(np.concatenate([C1, C2]))
        CG_mapping.append(np.concatenate([[i], C1, C2]))
        if C1 in index_C_end or C2 in index_C_end:
            type_CG.append(2)
        else:
            type_CG.append(1)
        mol_idx_CG.append(lmp.atom_info[lmp.atom_info[:, 0] == i, 1])

    type_CG = np.array(type_CG, dtype=int)
    CG_mapping = np.array(CG_mapping, dtype=int)
    mol_idx_CG = np.array(mol_idx_CG, dtype=int)

    atom_info_sort = lmp.atom_info[np.argsort(lmp.atom_info[:, 0])]

    # box size, assume cubic box
    box_size = lmp.x[1] - lmp.x[0]
    mass_CG = np.array([lmp.mass[3, 1], lmp.mass[2, 1], lmp.mass[2, 1]], dtype=float)
    # find the coordinates of the CG beads, based on center of mass
    CG_coord = np.zeros((len(CG_mapping), 3))
    for i in range(len(CG_mapping)):
        coors_AA = atom_info_sort[CG_mapping[i]-1, 4:7]
        rcoors_AA = coors_AA - coors_AA[0]
        # periodic boundary condition
        rcoors_AA = rcoors_AA - np.rint(rcoors_AA/box_size)*box_size
        # center of mass
        CG_coord[i] = coors_AA[0] + np.sum(rcoors_AA*mass_CG, axis=0)/np.sum(mass_CG)

    # CG atom info,

    CG_atom_info = []
    for i in range(len(CG_mapping)):
        CG_atom_info.append([i+1, mol_idx_CG[i][0], type_CG[i], 0,
                            CG_coord[i, 0], CG_coord[i, 1], CG_coord[i, 2]])
    CG_atom_info = np.array(CG_atom_info)

    # CG bond info, 1-1 bond (type 1) and 1-2 bond (type 2)
    CG_bond_info = []
    n = 0
    # for every all-atom bond, check if they are in the different CG bead, if so, add a CG bond
    for i in range(len(lmp.bond_info)):
        bond = np.array(lmp.bond_info[i, 2:4], dtype=int)
        # mol = lmp.atom_info[lmp.atom_info[:,0]==bond[0],1]
        CG1 = np.where(np.any(CG_mapping == bond[0], axis=1))[0]
        CG2 = np.where(np.any(CG_mapping == bond[1], axis=1))[0]
        if len(CG1) == 0 or len(CG2) == 0:
            continue
        if CG1 != CG2:
            n += 1
            if type_CG[CG1] == 1 and type_CG[CG2] == 1:
                type_bond_CG = 1
            else:
                type_bond_CG = 2
            CG_bond_info.append([n, type_bond_CG, CG1.squeeze()+1, CG2.squeeze()+1])

    CG_bond_info = np.array(CG_bond_info, dtype=int)

    mass_C = float(lmp.mass[3, 1])
    mass_O = float(lmp.mass[2, 1])
    mass_H = float(lmp.mass[0, 1])

    mass_CG1 = mass_O + 2*mass_C + 4*mass_H
    mass_CG2 = mass_O + 2*mass_C + 5*mass_H

    # find chains of CG beads
    chains = []
    # start from type 2 CG beads
    CG2 = np.where(type_CG == 2)[0]+1
    for i in CG2:
        if len(chains) == 0 or i not in np.concatenate(chains):
            chain = []
            chain.append(i)
            # find the bead connected to CG2 bead based on CG bond info
            while True:
                neighbors = []
                if np.sum(CG_bond_info[:, 2] == i) > 0:
                    neighbors.append(CG_bond_info[CG_bond_info[:, 2] == i, 3][0])
                if np.sum(CG_bond_info[:, 3] == i) > 0:
                    neighbors.append(CG_bond_info[CG_bond_info[:, 3] == i, 2][0])
                if len(neighbors) == 0:
                    break
                else:
                    next_bead = [neighbor for neighbor in neighbors if neighbor not in chain]
                    if len(next_bead) == 0:
                        break
                    else:
                        chain.append(next_bead[0])
                        i = next_bead[0]
            chains.append(chain)

    # generate angles for chains
    CG_angle_info = []
    angle_id = 1
    for chain in chains:
        for i in range(len(chain) - 2):
            # Define angle between monomers i, i+1, and i+2
            if i == 0 or i == len(chain)-3:
                CG_angle_info.append([angle_id, 2, chain[i], chain[i + 1], chain[i + 2]])
            else:
                CG_angle_info.append([angle_id, 1, chain[i], chain[i + 1], chain[i + 2]])
            angle_id += 1
    CG_angle_info = np.array(CG_angle_info)

    # write into lammps file
    lmp_CG = tl.lammps(natoms=len(CG_atom_info),
                       x=lmp.x,
                       y=lmp.y,
                       z=lmp.z,
                       mass=np.array([[1.0, mass_CG1],
                                      [2.0, mass_CG2]]),
                       natom_types=2,
                       nbond_types=2,
                       nangle_types=2,
                       nbonds=len(CG_bond_info),
                       nangles=len(CG_angle_info),
                       atom_info=CG_atom_info,
                       bond_info=CG_bond_info,
                       angle_info=CG_angle_info,
                       )
    tl.write_lammps(CG_file, lmp_CG, mode='full')
    return lmp_CG, CG_mapping, chains


def map_CG_one(coors_AA, box_size, mass_AA):
    """
    map the all-atom coordinates to CG coordinate
    inputs:
        coors_AA: all-atom coordinates
        box_size: box size of the system (cubic box)
        mass_AA: mass of all-atom particles
    return coord_CG: CG coordinates
    """
    rcoors_AA = coors_AA - coors_AA[0]
    rcoors_AA = rcoors_AA - np.rint(rcoors_AA/box_size)*box_size
    coord_CG = coors_AA[0] + np.sum(rcoors_AA*mass_AA.reshape(-1, 1), axis=0)/np.sum(mass_AA)
    return coord_CG


def map_CG_coord(AA_coord, CG_mapping, box_size, mass_AA):
    """
    map the all-atom coordinates to CG coordinates
    inputs:
        AA_coord: all-atom coordinates
        CG_mapping: a list of all-atom indices for each CG bead (list of list)
        box_size: box size of the system (cubic box)
        mass_AA: mass of all-atom particles
    return CG_coord: CG coordinates
    """
    CG_coord = np.zeros((len(CG_mapping), 3))
    for i in range(len(CG_mapping)):
        coors_AA = AA_coord[CG_mapping[i]-1, :]
        rcoors_AA = coors_AA - coors_AA[0]
        # periodic boundary condition
        rcoors_AA = rcoors_AA - np.rint(rcoors_AA/box_size)*box_size
        # center of mass
        mass_CG = mass_AA[CG_mapping[i]-1]
        CG_coord[i] = coors_AA[0] + np.sum(rcoors_AA*mass_CG.reshape(-1, 1), axis=0)/np.sum(mass_CG)
    CG_coord = CG_coord - np.floor(CG_coord/box_size)*box_size
    return CG_coord


def lmpdump_AA2CG_PEO(AA_file, dump_file):
    lmp_AA = tl.read_lammps_full(AA_file)
    lmp_CG, CG_mapping = lmp_AA2CG_PEO(AA_file, 'tmp.data')
    mass_CG = np.array([lmp_AA.mass[3, 1], lmp_AA.mass[2, 1], lmp_AA.mass[2, 1]], dtype=float)
    frame_list, t_list, L_list = tl.read_lammps_dump_custom(dump_file)
    CG_coord_list, box_size_list = [], []
    for i in range(len(frame_list)):
        box_size = L_list[i][0][1] - L_list[i][0][0]
        coord_AA = frame_list[i][:, 3:6]
        CG_coord = map_CG_coord(coord_AA, CG_mapping, box_size, mass_CG)
        CG_coord_list.append(CG_coord)
        box_size_list.append(box_size)
    return lmp_CG, CG_coord_list, box_size_list

# structures


def find_neighbor(atom_index, bond_info):
    C1 = bond_info[bond_info[:, 2] == atom_index, 3]
    C2 = bond_info[bond_info[:, 3] == atom_index, 2]
    neighbors = np.concatenate([C1, C2]).astype(int)
    return neighbors


def lmp_AA2CG_PMMA1(AA_file, CG_file):

    lmp = tl.read_lammps_full(AA_file)
    lmp.atom_info = lmp.atom_info[np.argsort(lmp.atom_info[:, 0])]
    index_C4 = lmp.atom_info[lmp.atom_info[:, 2] == 4, 0]

    CG_mapping = []
    type_CG = []
    mol_idx_CG = []
    for i in index_C4:
        mol = lmp.atom_info[lmp.atom_info[:, 0] == i, 1]
        neighbors = find_neighbor(i, lmp.bond_info)
        O5 = neighbors[lmp.atom_info[neighbors-1, 2] == 5]
        O6 = neighbors[lmp.atom_info[neighbors-1, 2] == 6]
        C10_3 = neighbors[(lmp.atom_info[neighbors-1, 2] == 10) |
                          (lmp.atom_info[neighbors-1, 2] == 3)]
        neighbor_O6 = find_neighbor(O6, lmp.bond_info)
        C7 = neighbor_O6[(lmp.atom_info[neighbor_O6-1, 2] == 7)]
        bead1 = np.concatenate([[i], O5, O6, C7])
        assert len(bead1) == 4, "Error: bead1 length is not 4"
        # CG_mapping.append(bead1)
        # type_CG.append(1)
        # mol_idx_CG.append(mol)

        neighbor_C10_3 = find_neighbor(C10_3, lmp.bond_info)
        C2 = neighbor_C10_3[(lmp.atom_info[neighbor_C10_3-1, 2] == 2)]
        C9 = neighbor_C10_3[(lmp.atom_info[neighbor_C10_3-1, 2] == 9)]
        C9_first = C9[C9 < i]
        bead2 = np.concatenate([C10_3, C2, C9_first])
        assert len(bead2) == 3, "Error: bead2 length is not 3"
        # CG_mapping.append(bead2)
        # type_CG.append(2)
        # mol_idx_CG.append(mol)

        CG_mapping.append(np.concatenate([bead1, bead2]))
        type_CG.append(1)
        mol_idx_CG.append(mol)

    type_CG = np.array(type_CG, dtype=int)
    CG_mapping = np.array(CG_mapping, dtype=int)
    mol_idx_CG = np.array(mol_idx_CG, dtype=int)

    # box size, assume cubic box
    box_size = lmp.x[1] - lmp.x[0]
    mass_CG1 = np.array([lmp.mass[4-1, 1], lmp.mass[5-1, 1],
                        lmp.mass[6-1, 1], lmp.mass[7-1, 1]], dtype=float)
    mass_CG2 = np.array([lmp.mass[10-1, 1], lmp.mass[2-1, 1], lmp.mass[9-1, 1]], dtype=float)
    mass_CG = np.concatenate([mass_CG1, mass_CG2])
    # find the coordinates of the CG beads, based on center of mass
    CG_coord = np.zeros((len(CG_mapping), 3))
    for i in range(len(CG_mapping)):
        coors_AA = lmp.atom_info[CG_mapping[i]-1, 4:7]
        rcoors_AA = coors_AA - coors_AA[0]
        # periodic boundary condition
        rcoors_AA = rcoors_AA - np.rint(rcoors_AA/box_size)*box_size
        CG_coord[i] = coors_AA[0] + np.sum(rcoors_AA*mass_CG.reshape(-1, 1), axis=0)/np.sum(mass_CG)

    # CG atom info,

    CG_atom_info = []
    for i in range(len(CG_mapping)):
        CG_atom_info.append([i+1, mol_idx_CG[i][0], type_CG[i], 0,
                            CG_coord[i, 0], CG_coord[i, 1], CG_coord[i, 2]])
    CG_atom_info = np.array(CG_atom_info)

    # CG bond info, 1-1 bond (type 1) and 1-2 bond (type 2)
    CG_bond_info = []
    n = 0
    # for every all-atom bond, check if they are in the different CG bead, if so, add a CG bond
    for i in range(len(lmp.bond_info)):
        bond = np.array(lmp.bond_info[i, 2:4], dtype=int)
        # mol = lmp.atom_info[lmp.atom_info[:,0]==bond[0],1]
        CG1 = np.where(np.any(CG_mapping == bond[0], axis=1))[0]
        # CG2 = [i for i, arr in enumerate(CG_mapping) if bond[1] in arr]
        CG2 = np.where(np.any(CG_mapping == bond[1], axis=1))[0]
        if len(CG1) == 0 or len(CG2) == 0:
            continue
        if CG1 != CG2:
            n += 1
            if type_CG[CG1] == 1 and type_CG[CG2] == 1:
                type_bond_CG = 1
            # else:
            #     type_bond_CG = 2
            CG_bond_info.append([n, type_bond_CG, CG1.squeeze()+1, CG2.squeeze()+1])

    CG_bond_info = np.array(CG_bond_info, dtype=int)

    mass_C = float(lmp.mass[1, 1])
    mass_O = float(lmp.mass[4, 1])
    mass_H = float(lmp.mass[0, 1])

    mass_CG1 = 2*mass_O + 2*mass_C + 3*mass_H
    mass_CG2 = 3*mass_C + 5*mass_H
    mass_CG = mass_CG1 + mass_CG2

    # find chains of CG beads
    chains = []
    # start from end CG2 with only two bonds
    CG2 = np.where(type_CG == 1)[0]+1
    CG_end = []
    for i in CG2:
        if len(find_neighbor(i, CG_bond_info)) == 1:
            CG_end.append(i)

    for i in CG_end:
        if len(chains) == 0 or i not in np.concatenate(chains):
            chain = []
            chain.append(i)
            # find the bead connected to CG2 bead based on CG bond info
            while True:
                neighbors = find_neighbor(i, CG_bond_info)
                if len(neighbors) == 0:
                    break
                else:
                    next_bead = [neighbor for neighbor in neighbors if neighbor not in chain]
                    if len(next_bead) == 0:
                        break
                    else:
                        chain.append(next_bead[0])
                        i = next_bead[0]
            chains.append(chain)

    # generate angles for chains
    CG_angle_info = []
    angle_id = 1
    for chain in chains:
        for i in range(len(chain) - 2):
            # Define angle between monomers i, i+1, and i+2
            # if i==0 or i==len(chain)-3:
            #     CG_angle_info.append([angle_id, 2, chain[i], chain[i + 1], chain[i + 2]])
            # else:
            CG_angle_info.append([angle_id, 1, chain[i], chain[i + 1], chain[i + 2]])
            angle_id += 1
    CG_angle_info = np.array(CG_angle_info)

    # write into lammps file
    lmp_CG = tl.lammps(natoms=len(CG_atom_info),
                       x=lmp.x,
                       y=lmp.y,
                       z=lmp.z,
                       mass=np.array([[1.0, mass_CG]]),
                       natom_types=1,
                       nbond_types=1,
                       nangle_types=1,
                       nbonds=len(CG_bond_info),
                       nangles=len(CG_angle_info),
                       atom_info=CG_atom_info,
                       bond_info=CG_bond_info,
                       angle_info=CG_angle_info,
                       )
    tl.write_lammps(CG_file, lmp_CG)
    return lmp_CG, CG_mapping, chains






def write_pot_table(file_name, zip_FF):
    """
    write the two-body potential to a table file for lammps simulations
    inputs: file_name, zip_FF (zip of r, E, F, key)
    """
    t = datetime.date.today().strftime('%m/%d/%Y')
    f = open(file_name, 'w')
    f.write('# DATE: {}  UNITS: real \n'.format(t))
    f.write('# potential\n\n')
    # zip_FF contains r, E, F, 'keys' of each interaction
    # unzip the zip_FF, check how many types of interactions
    num_types = len(zip_FF)
    for i in range(num_types):
        r, E, F, key = zip_FF[i]

        f.write(f'{key}\n')
        f.write('N {0:}\n'.format(len(r)))
        f.write('\n')
        for i in range(len(r)):
            f.write('   {0:5d}      {1:12.8f} {2:21f} {3:21f}\n'.format(
                i+1, r[i], E[i], F[i]))
        f.write('\n')
    f.close()


def lmp_AA2CG_toluene(AA_file, CG_file):
    """
    one bead for each toluene molecule, CG bead is the center of mass of the toluene molecule
    """
    lmp = tl.read_lammps_full(AA_file)
    atom_info_sort = lmp.atom_info[np.argsort(lmp.atom_info[:, 0])]
    type_AA = atom_info_sort[:, 2]
    mol_id = np.unique(atom_info_sort[:, 1])

    CG_mapping = []
    type_CG = []
    type_CG_AA = []
    mol_idx_CG = []
    for i in mol_id:
        atoms = atom_info_sort[atom_info_sort[:, 1] == i, 0]
        CG_mapping.append(atoms)
        type_CG_AA.append([type_AA[int(iatom)-1] for iatom in atoms])
        type_CG.append(1)
        mol_idx_CG.append(i)

    type_CG = np.array(type_CG, dtype=int)
    CG_mapping = np.array(CG_mapping, dtype=int)
    # print(len(CG_mapping))
    mol_idx_CG = np.array(mol_idx_CG, dtype=int)

    # box size, assume cubic box
    box_size = lmp.x[1] - lmp.x[0]
    # find the coordinates of the CG beads, based on center of mass
    CG_coord = np.zeros((len(CG_mapping), 3))
    for i in range(len(CG_mapping)):
        mass_CG = np.array([lmp.mass[int(type_AA[iatom-1])-1, 1]
                           for iatom in CG_mapping[i]], dtype=float).reshape(-1, 1)
        coors_AA = atom_info_sort[CG_mapping[i]-1, 4:7]
        rcoors_AA = coors_AA - coors_AA[0]
        # periodic boundary condition
        rcoors_AA = rcoors_AA - np.rint(rcoors_AA/box_size)*box_size
        # center of mass
        CG_coord[i] = coors_AA[0] + np.sum(rcoors_AA*mass_CG, axis=0)/np.sum(mass_CG)

    # print(mass_CG)
    # CG atom info,

    CG_atom_info = []
    for i in range(len(CG_mapping)):
        CG_atom_info.append([i+1, mol_idx_CG[i], type_CG[i], 0,
                            CG_coord[i, 0], CG_coord[i, 1], CG_coord[i, 2]])
    CG_atom_info = np.array(CG_atom_info)

    mass_C = float(lmp.mass[1, 1])
    # mass_O = float(lmp.mass[2,1])
    mass_H = float(lmp.mass[-1, 1])

    mass_CG1 = 7*mass_C + 8*mass_H

    # write into lammps file
    lmp_CG = tl.lammps(natoms=len(CG_atom_info),
                       x=lmp.x,
                       y=lmp.y,
                       z=lmp.z,
                       mass=np.array([[1.0, mass_CG1]]),
                       natom_types=1,
                       atom_info=CG_atom_info,
                       )
    tl.write_lammps(CG_file, lmp_CG)
    return lmp_CG, CG_mapping, type_CG_AA

