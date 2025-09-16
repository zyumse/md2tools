import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import my_common as mc
import extract_local_str as els
import tools_lammps as tool_lmp
import copy
import scipy.constants as scc

from scipy.signal import savgol_filter
import re
import time

import scienceplots
plt.style.use(['science','ieee'])

def read_MLABT_log(file):
    log_read = pd.read_csv(file)
    idx_collect = []
    step_unique = np.unique((log_read.Step.to_numpy()))
    last_step = int(step_unique[-1])
    step_interval = int(step_unique[-1]) - int(step_unique[-2])
    for istep in range(0,last_step,step_interval):
        if np.sum(log_read.to_numpy()[:,0]==istep)>0:
            idx_collect.append(np.squeeze(np.argwhere(log_read.to_numpy()[:,0]==istep)[0]))
    log_refine = log_read.loc[idx_collect,:].reset_index()
    return log_refine, log_read

def read_MLABT_BBIr(file):
    
    with open(file, 'r') as file:
        content = file.read()
    
    # pattern = r'\[([\d\.e+-]+) ([\d\.e+-]+) ([\d\.e+-]+) ([\d\.e+-]+)\]'
    pattern = r'\[\s*([\d]+\.?[\deE+-]*)\s+([\d]+\.?[\deE+-]*)\s+([\d]+\.?[\deE+-]*)\s+([\d]+\.?[\deE+-]*)\s*\]'
    matches = re.findall(pattern, content)
    return np.array(matches).astype(float)

def create_periodic_image(G, image_count, box_size, coors):
    G_new = nx.Graph()
    
    # Helper function: Get translated coordinates for nodes
    def get_translated_coordinate(coordinate, translation_vector):
        return tuple(np.add(coordinate, translation_vector))
    
    # Helper function: Check if an edge crosses PBC inside an image
    def edge_crosses_pbc(u_coor, v_coor, box_size):
        # print(u_coor,v_coor,box_size)
        if abs(u_coor[0] - v_coor[0]) > box_size / 2:
            return True
        return False
    
    # 1. Duplicate nodes
    for i in range(image_count):
        for node, data in G.nodes(data=True):
            translated_coor = get_translated_coordinate(coors[node], [i * box_size, 0, 0])
            # print(translated_coor)
            G_new.add_node((node, i), coordinate=translated_coor)  # Using a tuple (node, i) to uniquely identify nodes in different images
            
    # 2. Add edges within each image, avoiding those that cross PBC
    for i in range(image_count):
        for u, v in G.edges():
            u_coor = G_new.nodes[(u, i)]['coordinate']
            v_coor = G_new.nodes[(v, i)]['coordinate']
            
            if not edge_crosses_pbc(u_coor, v_coor, box_size):
                G_new.add_edge((u, i), (v, i), capacity=1)
    # print(len(G_new.edges))
    
   # 3. Add edges between images considering PBC
    for i in range(image_count - 1):  # No PBC between the last image and the first one for G2 and G3
        for u in G.nodes():
            u_coor = G_new.nodes[(u, i)]['coordinate']
            for v in G.nodes():
                v_coor = G_new.nodes[(v, i+1)]['coordinate']
                if G.has_edge(u, v):  # Checking if nodes are connected in original G1
                    if (v_coor[0]-u_coor[0])<box_size/2:
                        G_new.add_edge((u, i), (v, i+1), capacity=1)  # Connect the nodes between adjacent images
                    
    return G_new

def create_NN_lammps(lammps_file,new_file,select_idx=[],natom_DGEBA=49,natom_MDA=29):
    """
    inputs: lammps_file,new_file,select_idx (optional for making the selected N atoms another type in visualization)
    output: print new lammps file 
    
    """
    lmp_tmp = els.read_lammps_full(lammps_file)
    bond_info = lmp_tmp.bond_info

    natoms,box,index,atom_type,coors = mc.read_lammps(lammps_file,lmp_mode='full')
    coors = coors[np.argsort(index)]
    atom_type = atom_type[np.argsort(index)]
    index = index[np.argsort(index)]
    idx_N = np.squeeze(np.argwhere(atom_type==8))
    
    n_mol = int(natoms/(natom_DGEBA*2+natom_MDA)*3) # total number of molecules 
    idx_N_mol = 3*((idx_N+1)//(natom_DGEBA*2+natom_MDA))+2 # mol index (MDA) of which the N is in

    e = [] # index of neighbor molecule of N 
    CN_N = []
    for iN in range(len(idx_N)):
        tmp = np.concatenate((bond_info[bond_info[:,2] == idx_N[iN]+1,3]-1,bond_info[bond_info[:,3] == idx_N[iN]+1,2]-1)) # neighbor atom
        CN_N.append(len(tmp))

        tmp_idx1 = (tmp)//(natom_DGEBA*2+natom_MDA)
        tmp_idx2 = (tmp)%(natom_DGEBA*2+natom_MDA) // natom_DGEBA
        idx_neigh_mol = tmp_idx1*3+tmp_idx2
        # print(idx_neigh_mol,idx_N_mol[iN])
        idx_neighbor = idx_neigh_mol
        # idx_neighbor = idx_neigh_mol[idx_neigh_mol!=idx_N_mol[iN]] ## ignore self
        # print(idx_neighbor,idx_N_mol[iN])
        e.append(idx_neighbor)
        # for ie in range(len(idx_neighbor)):
        #     e.append((int(idx_neighbor[ie])))
    # e = np.array(e)
    CN_N=np.array(CN_N) # all CN_N should be 3 
    
    ue = [np.unique(e[i]) for i in range(len(e))]
    bond_info_N = []
    n_bond_N = 0
    idx_mol_betweenNN= []

    for i in range(len(ue)):
        for j in range(i+1,len(ue)):
            if np.size(np.intersect1d(ue[i],ue[j]))>=1:
                n_bond_N += 1
                idx_mol_betweenNN.append(np.intersect1d(ue[i],ue[j]))
                if np.size(np.intersect1d(ue[i],ue[j]))>=2:
                    bond_info_N.append([n_bond_N,3,i+1,j+1])
                    # print(np.intersect1d(ue[i],ue[j])%3)  # number of self-loops
                else:
                    if np.intersect1d(ue[i],ue[j])%3==2:
                        bond_info_N.append([n_bond_N,2,i+1,j+1])
                    else:
                        bond_info_N.append([n_bond_N,1,i+1,j+1])
    bond_info_N = np.array(bond_info_N)
    num_mol_ue = np.array([np.sum(np.concatenate(ue) == i) for i in range(n_mol)])
    
    atom_info_N = lmp_tmp.atom_info[np.argsort(lmp_tmp.atom_info[:,0]),:][idx_N]
    atom_info_N[:,2] = 1
    atom_info_N[:,0] = np.arange(1,len(atom_info_N)+1)
    lmp_N = tool_lmp.lammps(len(idx_N),1,lmp_tmp.x,lmp_tmp.y,lmp_tmp.z,
                            mass=np.array([[1,14.007,'N']]).reshape(-1,3),
                            atom_info = atom_info_N,
                            )
    lmp_N.bond_info = bond_info_N
    lmp_N.nbond_types = 3
    lmp_N.nbonds = len(bond_info_N)
    
    if len(select_idx)>0:
        lmp_N.atom_info[select_idx,2]=2
        lmp_N.natom_types = 2 
        lmp_N.mass = np.array([[1,14.007,'N'],[2,15.999,'O']]).reshape(-1,3)
        
    tool_lmp.write_lammps_full(new_file,lmp_N)
    return ue

def find_shortest_path_across_PB(lammps_file,image_number,slice_x_max=None,direction='x'):
    """
    find the shortest paths across the PB (might be multiple),

    lammps_file needs to be a file based on NN !!!
    """
    lmp_new = tool_lmp.read_lammps_full(lammps_file)
    if slice_x_max is None:
        slice_x_max = lmp_new.x[1]-lmp_new.x[0]
    lmp_new.atom_info = lmp_new.atom_info[np.argsort(lmp_new.atom_info[:,0])]
    coors = lmp_new.atom_info[:,4:7]-np.array([lmp_new.x[0],lmp_new.y[0],lmp_new.z[0]])
    G1 = nx.Graph()
    G1.add_nodes_from(np.arange(0,lmp_new.natoms))
    for ib in range(len(lmp_new.bond_info)):
        G1.add_edges_from([lmp_new.bond_info[ib,2:].astype(int)-1])
    if direction=='x':
        box_size = float(np.diff(lmp_new.x))
        Gn = create_periodic_image(G1, image_number, box_size,coors)
        
    # idx_slice = np.argwhere((coors[:,0]<lmp_new.x[0]+slice_x_max)).squeeze()
    idx_slice = np.argwhere((coors[:,0]<slice_x_max)).squeeze()
    SP_10 = []
    path_10 = []
    for i in idx_slice:
        if nx.has_path(Gn,(i,0),(i,image_number-1)):
            # SP_10.append(nx.shortest_path_length(Gn,(i,0),(i,image_number-1)))
            path_tmp = np.array(nx.shortest_path(Gn,(i,0),(i,image_number-1)))
            SP_10.append(len(path_tmp)-1)
            path_10.append(path_tmp)
    return SP_10, path_10 

def find_shortest_path_across_PB_type2(lammps_file,image_number,slice_x_max=None,direction='x'):
    """
    find the shortest paths across the PB (might be multiple),
    
    lammps_file needs to be a file based on NN !!!
    """
    lmp_new = tool_lmp.read_lammps_full(lammps_file)
    if slice_x_max is None:
        slice_x_max = lmp_new.x[1]-lmp_new.x[0]
    lmp_new.atom_info = lmp_new.atom_info[np.argsort(lmp_new.atom_info[:,0])]
    coors = lmp_new.atom_info[:,4:7]-np.array([lmp_new.x[0],lmp_new.y[0],lmp_new.z[0]])
    G1 = nx.Graph()
    G1.add_nodes_from(np.arange(0,lmp_new.natoms))
    for ib in range(len(lmp_new.bond_info)):
        G1.add_edges_from([lmp_new.bond_info[ib,2:].astype(int)-1])
    if direction=='x':
        box_size = float(np.diff(lmp_new.x))
        Gn = create_periodic_image(G1, image_number, box_size,coors)
        
    idx_slice = np.argwhere((coors[:,0]<lmp_new.x[0]+slice_x_max) & (lmp_new.atom_info[:,2]==2)).squeeze()
    SP_10 = []
    path_10 = []
    for i in idx_slice:
        if nx.has_path(Gn,(i,0),(i,image_number-1)):
            # SP_10.append(nx.shortest_path_length(Gn,(i,0),(i,image_number-1)))
            path_tmp = np.array(nx.shortest_path(Gn,(i,0),(i,image_number-1)))
            SP_10.append(len(path_tmp)-1)
            path_10.append(path_tmp)
        # else:
        #     SP_10.append(1000)
        #     path_10.append([])
    return SP_10, path_10 

def find_SPL_type2(lammps_file,image_number,slice_x_max=None,direction='x'):
    """
    
    """
    lmp_new = tool_lmp.read_lammps_full(lammps_file)
    if slice_x_max is None:
        slice_x_max = lmp_new.x[1]-lmp_new.x[0]
    lmp_new.atom_info = lmp_new.atom_info[np.argsort(lmp_new.atom_info[:,0])]
    coors = lmp_new.atom_info[:,4:7]-np.array([lmp_new.x[0],lmp_new.y[0],lmp_new.z[0]])
    G1 = nx.Graph()
    G1.add_nodes_from(np.arange(0,lmp_new.natoms))
    for ib in range(len(lmp_new.bond_info)):
        G1.add_edges_from([lmp_new.bond_info[ib,2:].astype(int)-1])
    if direction=='x':
        box_size = float(np.diff(lmp_new.x))
        Gn = create_periodic_image(G1, image_number, box_size,coors)
        
    idx_slice = np.argwhere((coors[:,0]<lmp_new.x[0]+slice_x_max) & (lmp_new.atom_info[:,2]==2)).squeeze()
    SP_10 = []
    for i in idx_slice:
        try:
            SP_10.append(nx.shortest_path_length(Gn,(i,0),(i,image_number-1)))
            # path_tmp = np.array(nx.shortest_path(Gn,(i,0),(i,image_number-1)))
            # SP_10.append(len(path_tmp)-1)
            # path_10.append(path_tmp)
        except:
            SP_10.append(1000)
        #     path_10.append([])
    return SP_10

def create_NN_traj(dump_file,dump_new,lammps_file):
    """
    natoms, idx_N, 
    """
    natoms,box,index,atom_type,coors = mc.read_lammps(lammps_file,lmp_mode='full')
    coors = coors[np.argsort(index)]
    atom_type = atom_type[np.argsort(index)]
    index = index[np.argsort(index)]
    idx_N = np.squeeze(np.argwhere(atom_type==8))
    with open(dump_file, 'r') as f_in, open(dump_new, 'w') as f_out:
        for line in f_in:
            # modify the line here, e.g.:
            if line.startswith('ITEM: ATOMS'):
                f_out.write(line)
                for i in range(natoms):
                    next_line = next(f_in)
                    columns = next_line.split()
                    if int(columns[2]) == 8:
                        columns[0] = str(np.squeeze(np.argwhere((idx_N+1)==int(columns[0])))+1)
                        f_out.write(' '.join(columns) + '\n')
            elif line.startswith('ITEM: NUMBER OF ATOMS'):
                f_out.write(line)
                next_line = next(f_in)
                f_out.write('{}\n'.format(len(idx_N)))
            else:
                f_out.write(line) 


def get_SP(NN_file,image_min=2,image_max=15,remove_case3=True,slice_x_max=15):
    SPL_list = []
    SPL_all = []
    path_all = []
    # NN_file = '../applications/deform9_soapDFT0d25_300_redo/out_0d1Kps_NN.dat'
    for image_number in range(image_min,image_max):
        SPL, path = find_shortest_path_across_PB(NN_file,slice_x_max=slice_x_max,image_number=image_number)
        # np.sort(np.array(SPL)/12)
        SPL_true = []
        path_true = []
        for ipath in range(len(SPL)):
            if len(path[ipath][:,0]) - len(np.unique(path[ipath][:,0])) -1 == 0:
                SPL_true.append(SPL[ipath])
                path_true.append(path[ipath])
                SPL_all.append(SPL[ipath]/(image_number-1))
                path_all.append(path[ipath])
        if len(SPL_true)>0:
            SPL_list.append([image_number-1,np.sort(np.array(SPL_true)/(image_number-1))[0]])
            # print(SPL_list[-1])
        else:
            break
    SPL_all_unique = []
    path_all_unique = []
    for ipath in range(len(SPL_all)):
        if SPL_all[ipath] not in SPL_all_unique:
            SPL_all_unique.append(SPL_all[ipath])
            path_all_unique.append(path_all[ipath])
        else:
            if remove_case3:
                idx_given_paths = (np.argwhere(np.array(SPL_all_unique)==SPL_all[ipath]))
                n=0
                for jpath in idx_given_paths:
                        if (len(np.intersect1d(path_all_unique[jpath.squeeze()][:-1,0],
                            path_all[ipath][:-1,0])) == len(path_all[ipath][:-1,0])):
                            n+=1
                if n==0:
                    SPL_all_unique.append(SPL_all[ipath])
                    path_all_unique.append(path_all[ipath])
            else:
                SPL_all_unique.append(SPL_all[ipath])
                path_all_unique.append(path_all[ipath])
                
    return SPL_all_unique,path_all_unique

# def get_SP_new2(NN_file,image_min=2,image_max=10,slice_x_max=None):
#     SPL_list = []
#     SPL_all = []
#     # NN_file = '../applications/deform9_soapDFT0d25_300_redo/out_0d1Kps_NN.dat'
#     lmp_new = tool_lmp.read_lammps_full(NN_file)
#     atom_info = lmp_new.atom_info[np.argsort(lmp_new.atom_info[:,0])]
#     natoms = lmp_new.natoms
#     idx_type2 = atom_info[atom_info[:,2]==2,0]-1
#     for image_number in range(image_min,image_max):
#         SPL = find_SPL_type2(NN_file,slice_x_max=slice_x_max,image_number=image_number)
#         SPL_all.append(np.array(SPL)/(image_number-1))
#     SPL_all = np.vstack(SPL_all)
#     SPL_nodes = np.min(SPL_all,axis=0)             
#     return SPL_nodes

def get_SP_new(NN_file,image_min=2,image_max=15,remove_case3=True,slice_x_max=15):
    """
    the latest version, considering CASE 1-3
    """
    SPL_list = []
    SPL_all = []
    path_all = []
    # NN_file = '../applications/deform9_soapDFT0d25_300_redo/out_0d1Kps_NN.dat'
    for image_number in range(image_min,image_max):
        SPL, path = find_shortest_path_across_PB(NN_file,slice_x_max=slice_x_max,image_number=image_number)
        # np.sort(np.array(SPL)/12)
        SPL_true = []
        path_true = []
        for ipath in range(len(SPL)):
            if len(path[ipath][:,0]) - len(np.unique(path[ipath][:,0])) -1 == 0:
                SPL_true.append(SPL[ipath])
                path_true.append(path[ipath])
                SPL_all.append(SPL[ipath]/(image_number-1))
                path_all.append(path[ipath])
        if len(SPL_true)>0:
            SPL_list.append([image_number-1,np.sort(np.array(SPL_true)/(image_number-1))[0]])
            # print(SPL_list[-1])
        else:
            break
    SPL_all_unique = []
    path_all_unique = []
    for ipath in range(len(SPL_all)):
        if SPL_all[ipath] not in SPL_all_unique:
            SPL_all_unique.append(SPL_all[ipath])
            path_all_unique.append(path_all[ipath])
        else:
            if remove_case3:
                idx_given_paths = (np.argwhere(np.array(SPL_all_unique)==SPL_all[ipath]))
                n=0
                for jpath in idx_given_paths:
                    if (len(np.intersect1d(path_all_unique[jpath.squeeze()][:-1,0],
                        path_all[ipath][:-1,0])) == len(path_all[ipath][:-1,0])):
                        n+=1
                if n==0:
                    SPL_all_unique.append(SPL_all[ipath])
                    path_all_unique.append(path_all[ipath])
            else:
                SPL_all_unique.append(SPL_all[ipath])
                path_all_unique.append(path_all[ipath])

    # path_flat = [path_all_unique[i][:-1,0] for i in range(len(path_all_unique))]
    starting_point = np.unique([path[0,0] for path in path_all_unique])
    spl_final = []
    path_final = []
    for i in range(len(starting_point)):
        idx_starting = [path[0,0]==starting_point[i] for path in path_all_unique]
        idx_tmp = np.argwhere(idx_starting).flatten()
        if len(idx_tmp)==1:
            spl_final.append(SPL_all_unique[idx_tmp[0]])
            path_final.append(path_all_unique[idx_tmp[0]])
        else:
            spl_values = np.array(SPL_all_unique)[idx_tmp]
            min_spl_idx = idx_tmp[np.argmin(spl_values)]

            spl_final.append(spl_values.min())
            path_final.append(path_all_unique[min_spl_idx])
    return spl_final,path_final

def get_SP_network(NN_file,image_min=2,image_max=10,slice_x_max=None):
    SPL_list = []
    SPL_all = []
    path_all = []
    # NN_file = '../applications/deform9_soapDFT0d25_300_redo/out_0d1Kps_NN.dat'
    lmp_new = tool_lmp.read_lammps_full(NN_file)
    atom_info = lmp_new.atom_info[np.argsort(lmp_new.atom_info[:,0])]
    natoms = lmp_new.natoms
    # idx_type2 = atom_info[atom_info[:,2]==2,0]-1
    for image_number in range(image_min,image_max):
        SPL, path = find_shortest_path_across_PB(NN_file,slice_x_max=slice_x_max,image_number=image_number)
        # np.sort(np.array(SPL)/12)
        SPL_true = []
        path_true = []
        for ipath in range(len(SPL)):
            # if len(path[ipath][:,0]) - len(np.unique(path[ipath][:,0])) -1 == 0:
            SPL_true.append(SPL[ipath])
            path_true.append(path[ipath])
            SPL_all.append(SPL[ipath]/(image_number-1))
            path_all.append(path[ipath])
        if len(SPL_true)>0:
            SPL_list.append([image_number-1,np.sort(np.array(SPL_true)/(image_number-1))[0]])
            # print(SPL_list[-1])
        else:
            break
    SPL_nodes = np.ones(natoms)*1000
    path_nodes = []
    for ii,iatom in enumerate(range(natoms)):
        try:
            tmp_SPL = 1000
            tmp_path = []
            for i,ipath in enumerate(path_all):
                if (ipath[0,0]==iatom) and (SPL_all[i]<tmp_SPL):
                    tmp_SPL = SPL_all[i]
                    tmp_path = ipath
            SPL_nodes[ii] = tmp_SPL
            path_nodes.append(tmp_path)
        except:
            continue
                
    return SPL_nodes,path_nodes

# bond breaking
def count_BB_CG(file_bonds,file_trj,threshold):
    lmp = els.read_lammps_full(file_bonds)
    idx_bonded_atoms = (lmp.bond_info[:,2:]-1).astype(int)

    # distances of bonds in the trajectory 
    result, t = els.read_multiple_xyz(file_trj)

    bond_length = []
    strain = []
    for i in range(len(t)):
        box = result[i][0]
        coors = result[i][4]
        strain.append(np.log(box[0,0]/result[0][0][0,0]))
        # for ibond in range(len(idx_bonded_atoms)):
        dist = mc.pbc_distance(coors[idx_bonded_atoms[:,0]],
                        coors[idx_bonded_atoms[:,1]],
                        box)
        bond_length.append(dist)

    # number of broken bonds, if length>1.5
    num_BB = []
    for i in range(len(t)):
        num_BB.append(np.sum(bond_length[i]>threshold))
    return num_BB, np.array(strain)


# visualize local stress 
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

def write_xyz_from_density(density, threshold, filename="output.xyz"):
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

def output_data_afterBB(location, threshold_BB = 1.5):
    # write dump files after each bond breaking to lammps data file with modified topology
    # location = './length/M40_v1'
    file_bonds = '{}/cool.dat'.format(location)
    file_trj = '{}/dump_relax.data'.format(location)

    # SPL,path = get_SP(file_bonds,2,8)
    lmp = els.read_lammps_full(file_bonds)
    lmp_tmp = copy.copy(lmp)
    # result, t = els.read_multiple_xyz(file_trj)
    result, t, L = tool_lmp.read_lammps_dump_custom(file_trj)

    bond_length = []
    idx_bead_NN_list = []
    strain_list = []
    for i in range(len(t)):
        idx_bonded_atoms = (lmp_tmp.bond_info[:,2:]-1).astype(int)
        box = np.array([[L[i][0][1]-L[i][0][0],0,0],
                        [0,L[i][1][1]-L[i][1][0],0],
                        [0,0,L[i][2][1]-L[i][2][0]]])
        coors = result[i].loc[:,'x':'z'].to_numpy()
        
        # for ibond in range(len(idx_bonded_atoms)):
        dist = mc.pbc_distance(coors[idx_bonded_atoms[:,0]],
                        coors[idx_bonded_atoms[:,1]],
                        box)
        bond_length.append(dist)
        if np.sum(dist>threshold_BB)>0:
            strain_list.append(np.log(box[0,0]/(L[0][0][1]-L[0][0][0])))
            idx_bead_NN = (idx_bonded_atoms[np.argwhere(dist>1.5).squeeze()]).reshape(-1,2)
            idx_bead_NN_list.append(idx_bead_NN)
            for ibb in idx_bead_NN:
                A = np.argwhere((lmp_tmp.bond_info[:,2]==ibb[0]+1) & (lmp_tmp.bond_info[:,3]==ibb[1]+1))
                if len(A)>0:
                    lmp_tmp.bond_info = np.delete(lmp_tmp.bond_info, A.squeeze(), axis=0)
                    print(len(lmp_tmp.bond_info)+1,len(lmp_tmp.bond_info))
                    lmp_tmp.nbonds = lmp_tmp.nbonds -1
                else:
                    print('error on the bond')
            atom_info = lmp_tmp.atom_info[np.argsort(lmp_tmp.atom_info[:,0])]
            atom_info[:,4:7] = coors
            lmp_tmp.atom_info = atom_info
            lmp_tmp.x = [np.min(coors[:,0]),np.min(coors[:,0])+box[0,0]]
            lmp_tmp.y = [np.min(coors[:,1]),np.min(coors[:,1])+box[1,1]]
            lmp_tmp.z = [np.min(coors[:,2]),np.min(coors[:,2])+box[2,2]]
            tool_lmp.write_lammps_full('{}/t{}.dat'.format(location,i),lmp_tmp)
    
    return idx_bead_NN_list,strain_list

def first_BB_segment(file_trj,SPL,path,file_bonds,threshold):

    # 1st bond breakage 
    lmp = els.read_lammps_full(file_bonds)
    idx_bonded_atoms = (lmp.bond_info[:,2:]-1).astype(int)

    # distances of bonds in the trajectory 
    result, t = els.read_multiple_xyz(file_trj)

    bond_length = []
    strain = []
    for i in range(len(t)):
        box = result[i][0]
        coors = result[i][4]
        strain.append(np.log(box[0,0]/result[0][0][0,0]))
        # for ibond in range(len(idx_bonded_atoms)):
        dist = mc.pbc_distance(coors[idx_bonded_atoms[:,0]],
                        coors[idx_bonded_atoms[:,1]],
                        box)
        bond_length.append(dist)
        if np.sum(dist>threshold)>0:
            break

    # print(t[i])
    idx_bead_NN = (idx_bonded_atoms[np.argwhere(dist>1.5).squeeze()]).reshape(-1,2)
    # print(idx_bead_NN)

    idx_onpath_list = []
    for i in range(idx_bead_NN.shape[0]):
        for j in range(len(path)):
            if idx_bead_NN[i] in path[np.argsort(SPL)[j]]:
                path_length = np.sort(SPL)[j]
                idx_path = np.argwhere(np.sort(SPL)==path_length)[0]
                print(idx_path)
                print('0th broken bond in {}th SP (length), {}th SP'.format(idx_path, j),path_length-1)
                if j==idx_path:
                    # try:
                    which_edge = np.min([np.argwhere(path[np.argsort(SPL)[j]][:-1,0]==idx_bead_NN[i][0]).squeeze(),
                                        np.argwhere(path[np.argsort(SPL)[j]][:-1,0]==idx_bead_NN[i][1]).squeeze()])
                    # except:
                    print(len(path[np.argsort(SPL)[j]][:-1,0])-1,which_edge,idx_bead_NN[i])
                    idx_onpath_list.append([j,which_edge])
                break
    idx_onpath_list = np.concatenate([idx_onpath_list])
    return idx_onpath_list

def compute_RD(lammps_file,direction):
    # lammps_file = './length/M40_v0/cool.dat'
    lmp_new = tool_lmp.read_lammps_full(lammps_file)
    slice_x_max = lmp_new.x[1]-lmp_new.x[0]
    lmp_new.atom_info = lmp_new.atom_info[np.argsort(lmp_new.atom_info[:,0])]
    coors = lmp_new.atom_info[:,4:7]-np.array([lmp_new.x[0],lmp_new.y[0],lmp_new.z[0]])
    G1 = nx.Graph()
    G1.add_nodes_from(np.arange(0,lmp_new.natoms))
    for ib in range(len(lmp_new.bond_info)):
        G1.add_edges_from([lmp_new.bond_info[ib,2:].astype(int)-1])

    image_number = 2
    if direction=='x':
        box_size = float(np.diff(lmp_new.x))
        Gn = create_periodic_image(G1, image_number, box_size,coors)

    largest_cc = max(nx.connected_components(Gn),key=len)
    Gn_connected = Gn.subgraph(largest_cc).copy()

    N = len(Gn_connected.nodes())
    rd_vector = np.zeros(lmp_new.natoms)  # Initialize matrix
    for i, nodeA in enumerate(Gn_connected.nodes()):
        atom_A,image_A = nodeA
        for j, nodeB in enumerate(Gn_connected.nodes()):
            atom_B,image_B = nodeB
            if atom_A == atom_B and image_A == 0 and image_B == image_number-1:  # Compute RD once for each pair due to symmetry
                rd = nx.resistance_distance(Gn_connected, nodeA, nodeB)
                rd_vector[atom_A] = rd
    return rd_vector