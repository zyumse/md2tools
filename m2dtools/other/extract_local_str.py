import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import my_common as mc
import scipy.constants as scc

def read_lammps_dump(filename):
    """
    Reads a LAMMPS dump file with multiple configurations (timesteps) and extracts timestep, number of atoms, 
    box bounds, and atom data into pandas DataFrame.

    Parameters:
    filename (str): The path to the LAMMPS dump file.

    Returns:
    list of dicts: Each dict contains timestep, number of atoms, box bounds, and atom data as a DataFrame.
    """
    configurations = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if 'TIMESTEP' in line:
                i += 1
                timestep = int(lines[i].strip())
                i += 1
                line = lines[i].strip()
                if 'NUMBER OF ATOMS' in line:
                    i += 1
                    num_atoms = int(lines[i].strip())
                    i += 1
                    line = lines[i].strip()
                    box_bounds = []
                    if 'BOX BOUNDS' in line:
                        for _ in range(3):
                            i += 1
                            bounds = lines[i].strip().split()
                            box_bounds.append((float(bounds[0]), float(bounds[1])))
                        i += 1
                        line = lines[i].strip()
                        if 'ATOMS' in line:
                            headers = line.split()[2:]  # Get atom property names
                            i += 1
                            atoms = []
                            for _ in range(num_atoms):
                                atom_data = lines[i].strip().split()
                                atom_info = {headers[j]: float(atom_data[j]) for j in range(len(headers))}
                                atoms.append(atom_info)
                                i += 1
                            # Convert atom data to DataFrame
                            if atoms:
                                atom_df = pd.DataFrame(atoms)
                                configurations.append({
                                    "timestep": timestep,
                                    "num_atoms": num_atoms,
                                    "box_bounds": box_bounds,
                                    "atoms": atom_df
                                })
            else:
                i += 1

    return configurations

# Usage example (commented out as it requires an actual file)
# configurations = read_lammps_dump("path_to_your_dump_file.dump")
# for config in configurations:
#     print(f"Timestep: {config['timestep']}, Number of Atoms: {config['num_atoms']}")
#     print(config["atoms"].head())  # Display first few rows of the atoms DataFrame



def read_bond_info(file):

    f=open(file,'r')
    L=f.readlines()
    f.close()

    isxyxzyz = 0
    for iline in range(len(L)):
        if 'atoms' in L[iline]:
            natoms = int(L[iline].split()[0])
        if 'bonds' in L[iline]:
            nbonds = int(L[iline].split()[0])
        if 'angles' in L[iline]:
            nangles = int(L[iline].split()[0])
        if 'dihedrals' in L[iline]:
            ndihedrals = int(L[iline].split()[0])
        if 'impropers' in L[iline]:
            nimpropers = int(L[iline].split()[0])

        if 'atom types' in L[iline]:
            natom_types = int(L[iline].split()[0])
        if 'bond types' in L[iline]:
            nbond_types = int(L[iline].split()[0])
        if 'angle types' in L[iline]:
            nangle_types = int(L[iline].split()[0])
        if 'dihedral types' in L[iline]:
            ndihedral_types = int(L[iline].split()[0])
        if 'improper types' in L[iline]:
            nimproper_types = int(L[iline].split()[0])

        if 'xlo' in L[iline]:
            xlo=float(L[iline].split()[0])
            xhi=float(L[iline].split()[1])
        if 'ylo' in L[iline]:
            ylo=float(L[iline].split()[0])
            yhi=float(L[iline].split()[1])
        if 'zlo' in L[iline]:
            zlo=float(L[iline].split()[0])
            zhi=float(L[iline].split()[1])

        if 'xy' in L[iline]:
            isxyxzyz=1
            xy=float(L[iline].split()[0])
            xz=float(L[iline].split()[1])
            yz=float(L[iline].split()[2])

        if 'Masses' in L[iline]:
            lmass = iline+2
            mass = []
            for ia in range(natom_types):
                mass.append(L[lmass+ia].split()[:2])
            mass = np.vstack(mass).astype(float)

        ############ potential coeff ############## 
        if 'Pair Coeffs' in L[iline]:
            lpc = iline+2
            pc = []
            for ia in range(natom_types):
                pc.append(L[lpc+ia].split())
            pc = np.vstack(pc).astype(float)

        if 'Bond Coeffs' in L[iline]:
            lbc = iline+2
            bc = []
            for ia in range(nbond_types):
                bc.append(L[lbc+ia].split())
            bc = np.vstack(bc).astype(float)
            
        if 'Angle Coeffs' in L[iline]:
            lac = iline+2
            ac = []
            for ia in range(nangle_types):
                ac.append(L[lac+ia].split())
            ac = np.vstack(ac).astype(float)
            
        if 'Dihedral Coeffs' in L[iline]:
            ldc = iline+2
            dc = []
            for ia in range(ndihedral_types):
                dc.append(L[ldc+ia].split())
            dc = np.vstack(dc).astype(float)

        ########### atoms ################
        if 'Atoms' in L[iline]:
            lia = iline+2
            atom_info = []
            for ia in range(natoms):
                atom_info.append(L[lia+ia].split())
            atom_info = np.vstack(atom_info).astype(float)

    #     ########## topology ##############
        if 'Bonds' in L[iline]:
            lib = iline+2
            bond_info = []
            for ia in range(nbonds):
                bond_info.append(L[lib+ia].split())
            bond_info = np.vstack(bond_info).astype(float)
            
        if 'Angles' in L[iline]:
            liangle = iline+2
            angle_info = []
            for ia in range(nangles):
                angle_info.append(L[liangle+ia].split())
            angle_info = np.vstack(angle_info).astype(float)
        
        if 'Dihedrals' in L[iline]:
            lid = iline+2
            dihedral_info = []
            for ia in range(ndihedrals):
                dihedral_info.append(L[lid+ia].split())
            dihedral_info = np.vstack(dihedral_info).astype(float)
            
    if isxyxzyz==0:
            xy=0; xz=0; yz=0
            box = np.array([[xhi-xlo,0,0],[xy,yhi-ylo,0],[xz,yz,zhi-zlo]])
            shift = xlo
    else:
        print('Error: non-orthogonal')
# atom_info = pd.read_table(file,header=None,sep='\s+',skiprows=lia-1,nrows=natoms)
    return bc, bond_info, ac, angle_info, dc, dihedral_info

def box_coors_from_lmp(lmp):
    box = np.array([[lmp.x[1]-lmp.x[0],0,0],
                    [0,lmp.y[1]-lmp.y[0],0],
                    [0,0,lmp.z[1]-lmp.z[0]]])
    coors = lmp.atom_info[:, 4:7] - np.array([lmp.x[0],lmp.y[0],lmp.z[0]])
    return box, coors

def read_multiple_xyz(file):
    """
    read multiple frames in output xyz of lammps, 
    input: dump_file, !!!! now change to 'custum'
    output: result (index,type_atoms, coors) and t 
    """
    f = open(file)
    lft = list(f)
    f.close()
    lt=[]
    t=[]
    natom = int(lft[3].split()[0])
    for il in range(len(lft)):
        if 'ITEM: TIMESTEP' in lft[il]:
            lt.append(il)
            t.append(lft[il+1].split()[0])

    def read_lf(lf):
        box = np.zeros([3,3])      ###### now only orthogonal box 
        coors = np.zeros([natom,3])
        mol = []
        type_atom = []
        index = []
        l=0
        
        xlo = float(lf[5].split()[0]); xhi = float(lf[5].split()[1]);
        ylo = float(lf[6].split()[0]); yhi = float(lf[6].split()[1]);
        zlo = float(lf[7].split()[0]); zhi = float(lf[7].split()[1]);
        box[0,0] = xhi-xlo
        box[1,1] = yhi-ylo
        box[2,2] = zhi-zlo
        
        for ia in lf[9:9+natom]:
            coors[l,:] = np.array(ia.split()[3:6:1]).astype('float')
            coors[l,:] = coors[l,:] - np.array([xlo,ylo,zlo])
            type_atom.append(int(ia.split()[2]))
            mol.append(int(ia.split()[1]))
            index.append(int(ia.split()[0]))
            l+=1

        type_atom = np.array(type_atom)
        mol = np.array(mol)
        index = np.array(index)
        
        mol = mol[np.argsort(index)]
        type_atom = type_atom[np.argsort(index)]
        coors = coors[np.argsort(index)]
        index = index[np.argsort(index)]
        
        return box,index,mol,type_atom,coors

    # lf=[]
    result = []
    for it in range(len(lt)):
        if it==len(lt)-1:
    #         lf.append(lft[lt[it]:])
            result.append(read_lf(lft[lt[it]:]))
        else:
    #         lf.append(lft[lt[it]:lt[it+1]-1])
            result.append(read_lf(lft[lt[it]:lt[it+1]]))
    
    t=np.array(t)
    return result,t

def BL(box,coors,bond_idx):
    dcoor_bond = coors[bond_idx[:,0],:] - coors[bond_idx[:,1],:] 
    dcoor_bond[dcoor_bond[:,0]<-0.5*box[0,0],0] = dcoor_bond[dcoor_bond[:,0]<-0.5*box[0,0],0] + box[0,0]
    dcoor_bond[dcoor_bond[:,0]> 0.5*box[0,0],0] = dcoor_bond[dcoor_bond[:,0]> 0.5*box[0,0],0] - box[0,0]
    dcoor_bond[dcoor_bond[:,1]<-0.5*box[1,1],1] = dcoor_bond[dcoor_bond[:,1]<-0.5*box[1,1],1] + box[1,1]
    dcoor_bond[dcoor_bond[:,1]> 0.5*box[1,1],1] = dcoor_bond[dcoor_bond[:,1]> 0.5*box[1,1],1] - box[1,1]
    dcoor_bond[dcoor_bond[:,2]<-0.5*box[2,2],2] = dcoor_bond[dcoor_bond[:,2]<-0.5*box[2,2],2] + box[2,2]
    dcoor_bond[dcoor_bond[:,2]> 0.5*box[2,2],2] = dcoor_bond[dcoor_bond[:,2]> 0.5*box[2,2],2] - box[2,2]
    bond_length = np.sum(dcoor_bond**2,axis=1)**0.5
    return bond_length

def write_xyz(file_name,natoms,type_atoms,coors):
    f=open('{}'.format(file_name),'w')
    f.write('{0:5d}\n'.format(natoms))
    f.write('Generated by python script\n')
    for ia in range(natoms):
        f.write('{0:}     {1:.16f} {2:.16f} {3:.16f}\n'.format(type_atoms[ia],coors[ia,0],coors[ia,1],coors[ia,2]))
    f.close()

def search_ring_id(id1,id2,bd_neigh,atom_type):
    """
    id1,id2 are two idx in one ring, make sure they are connected and belong to 'CA'
    """
    if id2 not in bd_neigh[id1]:
        print('Error inputs')
    else:
        idx_ring = []
        idx_ring.append(id1)
        idx_ring.append(id2)
        id3 = np.setdiff1d(bd_neigh[id2][atom_type[bd_neigh[id2]]==1],id1)[0]
        id4 = np.setdiff1d(bd_neigh[id3][atom_type[bd_neigh[id3]]==1],id2)[0]
        id5 = np.setdiff1d(bd_neigh[id4][atom_type[bd_neigh[id4]]==1],id3)[0]
        id6 = np.setdiff1d(bd_neigh[id5][atom_type[bd_neigh[id5]]==1],id4)[0]
        idx_ring.append(id3)
        idx_ring.append(id4)
        idx_ring.append(id5)
        idx_ring.append(id6)
    return idx_ring

def extr_local_str(idx_atom,box,atom_type,coors,bond_info,bd_neigh,rcut1=8,rcut2=6.5):
    """
    Extract the local structure around a centered bond
    input:idx_atom,box,atom_type,coors,bond_info,diff,r1,r2
    where idx_atom is indices of two atoms in the bond, diff is the vector matrix NN3,
    rcut1 is the overall cutoff
    rcut2 is the boundary between inner and outer parts
    """
    # create a small box around the centered atom to save memory 
#     lsmall = 20
    coors_new = coors - coors[idx_atom[0],:]
    coors_new[coors_new[:,0]<-0.5*box[0,0],0] = coors_new[coors_new[:,0]<-0.5*box[0,0],0] + box[0,0]
    coors_new[coors_new[:,0]> 0.5*box[0,0],0] = coors_new[coors_new[:,0]> 0.5*box[0,0],0] - box[0,0]
    coors_new[coors_new[:,1]<-0.5*box[1,1],1] = coors_new[coors_new[:,1]<-0.5*box[1,1],1] + box[1,1]
    coors_new[coors_new[:,1]> 0.5*box[1,1],1] = coors_new[coors_new[:,1]> 0.5*box[1,1],1] - box[1,1]
    coors_new[coors_new[:,2]<-0.5*box[2,2],2] = coors_new[coors_new[:,2]<-0.5*box[2,2],2] + box[2,2]
    coors_new[coors_new[:,2]> 0.5*box[2,2],2] = coors_new[coors_new[:,2]> 0.5*box[2,2],2] - box[2,2]
    diff0 = np.sum(coors_new**2,axis=1)**0.5
    
    coors_new = coors - coors[idx_atom[1],:]
    coors_new[coors_new[:,0]<-0.5*box[0,0],0] = coors_new[coors_new[:,0]<-0.5*box[0,0],0] + box[0,0]
    coors_new[coors_new[:,0]> 0.5*box[0,0],0] = coors_new[coors_new[:,0]> 0.5*box[0,0],0] - box[0,0]
    coors_new[coors_new[:,1]<-0.5*box[1,1],1] = coors_new[coors_new[:,1]<-0.5*box[1,1],1] + box[1,1]
    coors_new[coors_new[:,1]> 0.5*box[1,1],1] = coors_new[coors_new[:,1]> 0.5*box[1,1],1] - box[1,1]
    coors_new[coors_new[:,2]<-0.5*box[2,2],2] = coors_new[coors_new[:,2]<-0.5*box[2,2],2] + box[2,2]
    coors_new[coors_new[:,2]> 0.5*box[2,2],2] = coors_new[coors_new[:,2]> 0.5*box[2,2],2] - box[2,2]
    diff1 = np.sum(coors_new**2,axis=1)**0.5
    
    # intital index screening 
    idx_include1 = np.argwhere(diff0<rcut1) 
    idx_include2 = np.argwhere(diff1<rcut1)  
    idx_include = np.unique(np.squeeze(np.concatenate((idx_include1,idx_include2)))) # small
#     idx_include = idx[idx_include_s] # index in large 
    
    idx_inner1 = np.argwhere(diff0<rcut2)  # small 
    idx_inner2 = np.argwhere(diff1<rcut2) # small
    idx_inner = np.unique(np.squeeze(np.concatenate((idx_inner1,idx_inner2)))) 
    idx_outer = np.setdiff1d(idx_include,idx_inner)
#     idx_inner = idx[idx_inner_s]
#     idx_outer = idx[idx_outer_s]
    idx_include = np.concatenate((idx_inner,idx_outer)) # re-order 
    
    ##### bond in the inner region ##### 
    
    ##### end #####
        
    atom_type_tmp = atom_type[idx_include]
    atom_type_select = np.empty(len(atom_type_tmp),dtype='U8')
    atom_type_select[np.squeeze(atom_type_tmp==1)] = 'CA'
    atom_type_select[np.squeeze(atom_type_tmp==2)] = 'CT'
    atom_type_select[np.squeeze(atom_type_tmp==3)] = 'CY'
    atom_type_select[np.squeeze(atom_type_tmp==4)] = 'H'
    atom_type_select[np.squeeze(atom_type_tmp==5)] = 'HA'
    atom_type_select[np.squeeze(atom_type_tmp==6)] = 'HC'
    atom_type_select[np.squeeze(atom_type_tmp==7)] = 'HO'
    atom_type_select[np.squeeze(atom_type_tmp==8)] = 'N'
    atom_type_select[np.squeeze(atom_type_tmp==9)] = 'O$'
    atom_type_select[np.squeeze(atom_type_tmp==10)] = 'OH'
    atom_type_select[np.squeeze(atom_type_tmp==11)] = 'OS'
    
    coors_tmp = coors - coors[idx_atom[0]]
    while np.sum(coors_tmp[:,0]<-0.5*box[0,0])>0:
        coors_tmp[coors_tmp[:,0]<-0.5*box[0,0],0] = coors_tmp[coors_tmp[:,0]<-0.5*box[0,0],0] + box[0,0]
    while np.sum(coors_tmp[:,0]> 0.5*box[0,0])>0:
        coors_tmp[coors_tmp[:,0]>0.5*box[0,0],0] = coors_tmp[coors_tmp[:,0]>0.5*box[0,0],0] - box[0,0]
    while np.sum(coors_tmp[:,1]<-0.5*box[1,1])>0:
        coors_tmp[coors_tmp[:,1]<-0.5*box[1,1],1] = coors_tmp[coors_tmp[:,1]<-0.5*box[1,1],1] + box[1,1]
    while np.sum(coors_tmp[:,1]>0.5*box[1,1])>0:
        coors_tmp[coors_tmp[:,1]>0.5*box[1,1],1] = coors_tmp[coors_tmp[:,1]>0.5*box[1,1],1] - box[1,1]
    while np.sum(coors_tmp[:,2]<-0.5*box[2,2])>0:
        coors_tmp[coors_tmp[:,2]<-0.5*box[2,2],2] = coors_tmp[coors_tmp[:,2]<-0.5*box[2,2],2] + box[2,2]
    while np.sum(coors_tmp[:,2]>0.5*box[2,2])>0:
        coors_tmp[coors_tmp[:,2]>0.5*box[2,2],2] = coors_tmp[coors_tmp[:,2]>0.5*box[2,2],2] - box[2,2]    
    
    coors_select = coors_tmp[idx_include]
    
    write_xyz('test0.xyz',len(atom_type_select),atom_type_select,coors_select)
    
    n_add = 0
    atom_type_add = []
    coors_add = []
    idx_add_CA = []
    idx_add_tmp = []

    ###### get rid of the single H ############
    for i in idx_outer:
        bond_out_i = bd_neigh[i]
        tmp_neigh_in = 0
        for j in bond_out_i:
            if j in idx_include:
                tmp_neigh_in+=1
                continue
        if (tmp_neigh_in==0) and (atom_type[i] in [4,5,6,7]):
#            i_small = np.squeeze(np.argwhere(idx == i))
            atom_type_select = np.delete(atom_type_select,np.squeeze(np.argwhere(idx_include==i)))
            coors_select = np.delete(coors_select,np.squeeze(np.argwhere(idx_include==i)),0)  
            idx_outer = np.delete(idx_outer,np.squeeze(np.argwhere(idx_outer==i)))
            idx_include = np.delete(idx_include,np.squeeze(np.argwhere(idx_include==i)))
    
    write_xyz('test01.xyz',len(atom_type_select),atom_type_select,coors_select)
    
    for i in idx_include:
        bond_out_i = bd_neigh[i]  # bonded neighbor of i 
        for j in bond_out_i:
            if j in idx_include:
                continue
            else:  # j is outside 
                bond_out_j = np.setdiff1d(bd_neigh[j],i)  
                if (atom_type[j]!=1) or (atom_type[i]!=1): ### if it is not CA (benzene ring)
                    n_tmp_bridge = 0
                    
                else:
                    idx_ring = search_ring_id(i,j,bd_neigh,atom_type)
                    for k in idx_ring:
                        if k in idx_include:
                            continue
                        else:
                            n_add += 1
                            idx_add_CA.append(k)
                            idx_include = np.append(idx_include,k)
                            atom_type_add.append('CA')
                            coors_add_tmp = coors_tmp[k]
                            coors_add.append(coors_add_tmp)
    
    if len(coors_add)==0:
        write_xyz('test1.xyz',len(idx_include),atom_type_select,coors_select)
    else:
        write_xyz('test1.xyz',len(idx_include),np.concatenate((atom_type_select,np.array(atom_type_add))),np.concatenate((coors_select,np.array(coors_add))))
    
    for i in idx_include:
        bond_out_i = bd_neigh[i]  # bonded neighbor of i 
        for j in bond_out_i:
            if j in idx_include:
                continue
            else:  # j is outside 
                bond_out_j = np.setdiff1d(bd_neigh[j],i)  
                if (atom_type[j]!=1) or (atom_type[i]!=1): ### if it is not CA (benzene ring)
                    n_tmp_bridge = 0
                    for k in bond_out_j:
                        if k in idx_include:
                            n_tmp_bridge+=1
                    if n_tmp_bridge == 0: ###### if j is a radical totally outside, then replace it by H
                        n_add += 1
                        
                        if atom_type[j] in [4,5,6,7]:
                            idx_include = np.append(idx_include,j)
                            atom_type_add.append('H')
                            coors_add_tmp = coors_tmp[j]
                        else:
                            idx_include = np.append(idx_include,-1)
                            idx_add_tmp.append(j)
                            atom_type_add.append('H')
                            coors_add_tmp = coors_tmp[i]+(coors_tmp[j]-coors_tmp[i])/np.sum((coors_tmp[j]-coors_tmp[i])**2)**0.5*1.09 ### 1.09 might have some problem
                        coors_add.append(coors_add_tmp)
                        
                    if n_tmp_bridge>0: ####### j is a bridge atom, add it 
                        n_add += 1
                        #print('bridge')  
                        idx_include = np.append(idx_include,j)
                        if atom_type[j]==2:
                            atom_type_add.append('CT')
                        elif atom_type[j]==11:
                            atom_type_add.append('OS')
                        elif atom_type[j]==1:
                            atom_type_add.append('CA')
                        elif atom_type[j]==3:
                            atom_type_add.append('CY')
                        elif atom_type[j]==8:
                            atom_type_add.append('NT')
                        elif atom_type[j]==9:
                            atom_type_add.append('O$')
                        else:
                            print('warning',atom_type[j]) ##### may need to add types here 
                        coors_add_tmp = coors_tmp[j] 
                        coors_add.append(coors_add_tmp)

    #                     bond_out_j = np.setdiff1d(bd_neigh[j],i)
                        for k in bond_out_j:
                            if k not in idx_include:
                                if atom_type[k] in [4,5,6,7]:
    #                                 print('H')
                                    n_add += 1
                                    idx_include = np.append(idx_include,k)
                                    atom_type_add.append('H')
                                    coors_add_tmp = coors_tmp[k]
                                    coors_add.append(coors_add_tmp)
                                else:
    #                                 print('not H')
                                    n_add += 1
                                    idx_include = np.append(idx_include,-1)
#                                     idx_include = np.append(idx_include,k)
                                    idx_add_tmp.append(k)
                                    #### include the atom and replace it by H
                                    atom_type_add.append('H')
                                    coors_add_tmp = coors_tmp[j]+(coors_tmp[k]-coors_tmp[j])/np.sum((coors_tmp[k]-coors_tmp[j])**2)**0.5*1.09 ### 1.09 might have some problem
                                    coors_add.append(coors_add_tmp)
    
    if len(coors_add)==0:
        write_xyz('test2.xyz',len(idx_include),atom_type_select,coors_select)
    else:
        write_xyz('test2.xyz',len(idx_include),np.concatenate((atom_type_select,np.array(atom_type_add))),np.concatenate((coors_select,np.array(coors_add))))
                   
    
    atom_type_add = np.array(atom_type_add)
    coors_add = np.vstack(coors_add)
    n_inner = len(idx_inner)
    n_include = len(idx_include)
    idx_atom_interest = np.squeeze(np.array([np.argwhere(idx_include==idx_atom[0]),np.argwhere(idx_include==idx_atom[1])])) # start from 0
    
#     print(atom_type_select[2],bd_neigh[idx_include[2]])
    
    return n_include,n_inner,atom_type_select,atom_type_add,coors_select,coors_add,idx_atom_interest,idx_include

def write_orca_input(file_name,first_line,num_CPU,natoms,type_atoms,coors,n_inner,charge):
    """
    prepare input for the ORCA DFT calculation, geometry optimization with constraints 
    (fixing the positions of the outer-shell atoms to maintain the local stress)
    """
    f=open('{}'.format(file_name),'w')
    f.write('{}\n'.format(first_line))
    f.write('%pal\n')
    f.write('nprocs {}\n'.format(num_CPU))
    f.write('end\n\n')
    
    n_outer=natoms-n_inner
    f.write('%geom Constraints\n')
    for i in range(natoms-n_inner):
        f.write('{{ C {} C }}\n'.format(int(n_inner+i)))
    f.write('end\n')
    f.write('end\n\n')
    
    type_atoms = type_atoms.astype('<U1')
    f.write('* xyz {} 1\n'.format(charge)) ####### charge and spin, in this case, we dont need to worry
    for ia in range(natoms):
        f.write('{0:}     {1:20.12f} {2:20.12f} {3:20.12f}\n'.format(type_atoms[ia],coors[ia,0],coors[ia,1],coors[ia,2]))
    f.write('*')
    f.close()

def minimum_image(dcoor,box):
    while np.sum(dcoor[:,0]<-0.5*box[0,0])>0:
        dcoor[dcoor[:,0]<-0.5*box[0,0],0] = dcoor[dcoor[:,0]<-0.5*box[0,0],0] + box[0,0]
    while np.sum(dcoor[:,0]> 0.5*box[0,0])>0:
        dcoor[dcoor[:,0]>0.5*box[0,0],0] = dcoor[dcoor[:,0]>0.5*box[0,0],0] - box[0,0]
    while np.sum(dcoor[:,1]<-0.5*box[1,1])>0:
        dcoor[dcoor[:,1]<-0.5*box[1,1],1] = dcoor[dcoor[:,1]<-0.5*box[1,1],1] + box[1,1]
    while np.sum(dcoor[:,1]>0.5*box[1,1])>0:
        dcoor[dcoor[:,1]>0.5*box[1,1],1] = dcoor[dcoor[:,1]>0.5*box[1,1],1] - box[1,1]
    while np.sum(dcoor[:,2]<-0.5*box[2,2])>0:
        dcoor[dcoor[:,2]<-0.5*box[2,2],2] = dcoor[dcoor[:,2]<-0.5*box[2,2],2] + box[2,2]
    while np.sum(dcoor[:,2]>0.5*box[2,2])>0:
        dcoor[dcoor[:,2]>0.5*box[2,2],2] = dcoor[dcoor[:,2]>0.5*box[2,2],2] - box[2,2]
    return dcoor

class lammps:
    def __init__(self, natoms, nbonds=None, nangles=None, ndihedrals=None, nimpropers=None, natom_types=None, nbond_types=None, nangle_types=None, ndihedral_types=None, nimproper_types=None,x=None, y=None, z=None, mass=None, pair_coeff=None, bond_coeff=None, angle_coeff=None, dihedral_coeff=None, improper_coeff=None,atom_info=None, velocity_info=None, bond_info=None, angle_info=None, dihedral_info=None, improper_info=None):
        self.natoms = natoms
        self.nbonds = nbonds
        self.nangles = nangles
        self.ndihedrals = ndihedrals
        self.nimpropers = nimpropers
        
        self.natom_types = natom_types
        self.nbond_types = nbond_types
        self.nangle_types = nangle_types
        self.ndihedral_types = ndihedral_types
        self.nimproper_types = nimproper_types
        
        self.x = x
        self.y = y
        self.z = z
        
        self.mass = mass
        self.pair_coeff = pair_coeff
        self.bond_coeff = bond_coeff
        self.angle_coeff = angle_coeff
        self.dihedral_coeff = dihedral_coeff
        self.improper_coeff = improper_coeff
        
        self.atom_info = atom_info
        self.velocity_info = velocity_info
        self.bond_info = bond_info
        self.angle_info = angle_info
        self.dihedral_info = dihedral_info
        self.improper_info = improper_info

    def add_bond(self,index_1,index_2,bond_type):
        self.bond_info = np.vstack([self.bond_info, [len(self.bond_info)+1,bond_type,index_1,index_2]])
        self.nbonds = len(self.bond_info)

def read_lammps_full(file):
    # Initialize variables
    natoms = nbonds = nangles = ndihedrals = nimpropers = 0
    natom_types = nbond_types = nangle_types = ndihedral_types = nimproper_types = 0
    xlo = xhi = ylo = yhi = zlo = zhi = 0
    xy = xz = yz = 0
    mass = pc = bc = ac = dc = ic = None
    atom_info = velocity_info = bond_info = angle_info = dihedral_info = impropers_info = None
    isxyxzyz = 0

    with open(file, 'r') as f:
        L = f.readlines()

    # Rest of your code for parsing the file
    for iline in range(len(L)):
        if 'atoms' in L[iline]:
            natoms = int(L[iline].split()[0])
        if 'bonds' in L[iline]:
            nbonds = int(L[iline].split()[0])
        if 'angles' in L[iline]:
            nangles = int(L[iline].split()[0])
        if 'dihedrals' in L[iline]:
            ndihedrals = int(L[iline].split()[0])
        if 'impropers' in L[iline]:
            nimpropers = int(L[iline].split()[0])

        if 'atom types' in L[iline]:
            natom_types = int(L[iline].split()[0])
        if 'bond types' in L[iline]:
            nbond_types = int(L[iline].split()[0])
        if 'angle types' in L[iline]:
            nangle_types = int(L[iline].split()[0])
        if 'dihedral types' in L[iline]:
            ndihedral_types = int(L[iline].split()[0])
        if 'improper types' in L[iline]:
            nimproper_types = int(L[iline].split()[0])

        if 'xlo' in L[iline]:
            xlo=float(L[iline].split()[0])
            xhi=float(L[iline].split()[1])
        if 'ylo' in L[iline]:
            ylo=float(L[iline].split()[0])
            yhi=float(L[iline].split()[1])
        if 'zlo' in L[iline]:
            zlo=float(L[iline].split()[0])
            zhi=float(L[iline].split()[1])

        if 'xy' in L[iline]:
            isxyxzyz=1
            xy=float(L[iline].split()[0])
            xz=float(L[iline].split()[1])
            yz=float(L[iline].split()[2])

        if 'Masses' in L[iline]:
            lmass = iline+2
            mass = []
            for ia in range(natom_types):
                mass.append(L[lmass+ia].split())
            mass = np.vstack(mass)

        ############ potential coeff ############## 
        if 'Pair Coeffs' in L[iline]:
            lpc = iline+2
            pc = []
            for ia in range(natom_types):
                pc.append(L[lpc+ia].split())
            pc = np.vstack(pc).astype(float)

        if 'Bond Coeffs' in L[iline]:
            lbc = iline+2
            bc = []
            for ia in range(nbond_types):
                bc.append(L[lbc+ia].split())
            bc = np.vstack(bc).astype(float)

        if 'Angle Coeffs' in L[iline]:
            lac = iline+2
            ac = []
            for ia in range(nangle_types):
                ac.append(L[lac+ia].split())
            ac = np.vstack(ac).astype(float)

        if 'Dihedral Coeffs' in L[iline]:
            ldc = iline+2
            dc = []
            for ia in range(ndihedral_types):
                dc.append(L[ldc+ia].split())
            dc = np.vstack(dc).astype(float)

        if 'Improper Coeffs' in L[iline]:
            lic = iline+2
            ic = []
            for ia in range(nimproper_types):
                ic.append(L[lic+ia].split())
            ic = np.vstack(ic).astype(float)


        ########### atoms ################
        if 'Atoms' in L[iline]:
            lia = iline+2
            atom_info = []
            for ia in range(natoms):
                atom_info.append(L[lia+ia].split())
            atom_info = np.vstack(atom_info).astype(float)
            
        if 'Velocities' in L[iline]:
            liv = iline+2
            velocity_info = []
            for ia in range(natoms):
                velocity_info.append(L[liv+ia].split())
            velocity_info = np.vstack(velocity_info).astype(float)

    #     ########## topology ##############
        if 'Bonds' in L[iline]:
            lib = iline+2
            bond_info = []
            for ia in range(nbonds):
                bond_info.append(L[lib+ia].split())
            bond_info = np.vstack(bond_info).astype(float)

        if 'Angles' in L[iline]:
            lian = iline+2
            angle_info = []
            for ia in range(nangles):
                angle_info.append(L[lian+ia].split())
            angle_info = np.vstack(angle_info).astype(float)

        if 'Dihedrals' in L[iline]:
            lidi = iline+2
            dihedral_info = []
            for ia in range(ndihedrals):
                dihedral_info.append(L[lidi+ia].split())
            dihedral_info = np.vstack(dihedral_info).astype(float)

        if 'Impropers' in L[iline]:
            liim = iline+2
            impropers_info = []
            for ia in range(nimpropers):
                impropers_info.append(L[liim+ia].split())
            impropers_info = np.vstack(impropers_info).astype(float)

    # Checks before creating the lammps object
    if mass is None:
        mass = np.array([])
    if pc is None:
        pc = np.array([])
    # Repeat for other variables (bc, ac, dc, ic, etc.)

    if atom_info is None:
        atom_info = np.array([])
    # Repeat for other variables (velocity_info, bond_info, etc.)

    if isxyxzyz == 0:
        xy = xz = yz = 0
        box = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])

    # Create the lammps object
    result = lammps(natoms, nbonds, nangles, ndihedrals, nimpropers, 
                    natom_types, nbond_types, nangle_types, ndihedral_types, nimproper_types,
                    [xlo, xhi], [ylo, yhi], [zlo, zhi], mass, pc, bc, ac, dc, ic,
                    atom_info, velocity_info, bond_info, angle_info, dihedral_info, impropers_info)
    return result

def write_lammps_full(file_name,lmp_tmp):
    """
    lammps is a class of lammps with full attributes 
    """
    f=open('{}'.format(file_name),'w')
    f.write('Generated by ZY\n\n')
    f.write('{} atoms\n'.format(lmp_tmp.natoms))
    f.write('{} atom types\n'.format(lmp_tmp.natom_types))
    if lmp_tmp.nbonds is not None:
        f.write('{} bonds\n'.format(lmp_tmp.nbonds))
    if lmp_tmp.nbond_types is not None:
        f.write('{} bond types\n'.format(lmp_tmp.nbond_types))
    if lmp_tmp.nangles is not None:
        f.write('{} angles\n'.format(lmp_tmp.nangles))
    if lmp_tmp.nangle_types is not None:
        f.write('{} angle types\n'.format(lmp_tmp.nangle_types)) 
    if lmp_tmp.ndihedrals is not None:
        f.write('{} dihedrals\n'.format(lmp_tmp.ndihedrals))
    if lmp_tmp.ndihedral_types is not None:
        f.write('{} dihedral types\n'.format(lmp_tmp.ndihedral_types))
    if lmp_tmp.nimpropers is not None:
        f.write('{} impropers\n'.format(lmp_tmp.nimpropers))
    if lmp_tmp.nimproper_types is not None:
        f.write('{} improper types\n'.format(lmp_tmp.nimproper_types))
    f.write('\n')
    
    f.write('{0:.16f} {1:.16f} xlo xhi\n'.format(lmp_tmp.x[0],lmp_tmp.x[1]))
    f.write('{0:.16f} {1:.16f} ylo yhi\n'.format(lmp_tmp.y[0],lmp_tmp.y[1]))
    f.write('{0:.16f} {1:.16f} zlo zhi\n'.format(lmp_tmp.z[0],lmp_tmp.z[1]))
    f.write('\n')
    
    if lmp_tmp.mass is not None:
        f.write('Masses\n\n')
        if lmp_tmp.mass.shape[1]==2:
            for i in range(len(lmp_tmp.mass)):
                f.write('{0:d} {1:.3f}\n'.format(int(lmp_tmp.mass[i,0]),float(lmp_tmp.mass[i,1])))
        if lmp_tmp.mass.shape[1]==3:
            for i in range(len(lmp_tmp.mass)):
                f.write('{0:d} {1:.3f} # {2:s}\n'.format(int(lmp_tmp.mass[i,0]),float(lmp_tmp.mass[i,1]),lmp_tmp.mass[i,2]))
        f.write('\n')

    if lmp_tmp.pair_coeff is not None:
        f.write('Pair Coeffs\n\n')
        for i in range(len(lmp_tmp.pair_coeff)):
            f.write('{0:d} {1:f} {2:f}\n'.format(int(lmp_tmp.pair_coeff[i,0]),lmp_tmp.pair_coeff[i,1],lmp_tmp.pair_coeff[i,2]))
        f.write('\n')
    
    if lmp_tmp.bond_coeff is not None:
        f.write('Bond Coeffs\n\n')
        for i in range(len(lmp_tmp.bond_coeff)):
            f.write('{0:d} {1:f} {2:f}\n'.format(int(lmp_tmp.bond_coeff[i,0]),lmp_tmp.bond_coeff[i,1],lmp_tmp.bond_coeff[i,2]))
        f.write('\n')
    
    if lmp_tmp.angle_coeff is not None:
        f.write('Angle Coeffs\n\n')
        for i in range(len(lmp_tmp.angle_coeff)):
            f.write('{0:d} {1:f} {2:f}\n'.format(int(lmp_tmp.angle_coeff[i,0]),lmp_tmp.angle_coeff[i,1],lmp_tmp.angle_coeff[i,2]))
        f.write('\n')
    
    if lmp_tmp.dihedral_coeff is not None:
        f.write('Dihedral Coeffs\n\n')
        for i in range(len(lmp_tmp.dihedral_coeff)):
            f.write('{0:d} {1:f} {2:f} {3:f} {4:f}\n'.format(int(lmp_tmp.dihedral_coeff[i,0]),lmp_tmp.dihedral_coeff[i,1],lmp_tmp.dihedral_coeff[i,2],
                                                lmp_tmp.dihedral_coeff[i,3],lmp_tmp.dihedral_coeff[i,4]))
        f.write('\n')
    
    if lmp_tmp.improper_coeff is not None:
        f.write('Improper Coeffs\n\n')
        for i in range(len(lmp_tmp.improper_coeff)):
            f.write('{0:d} {1:f} {2:d} {3:d}\n'.format(int(lmp_tmp.improper_coeff[i,0]),lmp_tmp.improper_coeff[i,1],int(lmp_tmp.improper_coeff[i,2]),
                                                int(lmp_tmp.improper_coeff[i,3])))
        f.write('\n')
    
    if lmp_tmp.atom_info is not None:
        f.write('Atoms # full\n\n')
        for i in range(len(lmp_tmp.atom_info)):
            f.write('{0:d} {1:d} {2:d} {3:f} {4:f} {5:f} {6:f} {8:d} {7:d} {9:d}\n'.format(int(lmp_tmp.atom_info[i,0]),int(lmp_tmp.atom_info[i,1]),int(lmp_tmp.atom_info[i,2]),
                                                    lmp_tmp.atom_info[i,3],lmp_tmp.atom_info[i,4],lmp_tmp.atom_info[i,5],lmp_tmp.atom_info[i,6],
                            int(lmp_tmp.atom_info[i,7]),int(lmp_tmp.atom_info[i,8]),int(lmp_tmp.atom_info[i,9]) ))
        f.write('\n')
    
    if lmp_tmp.velocity_info is not None:
        f.write('Velocities \n\n')
        for i in range(len(lmp_tmp.velocity_info)):
            f.write('{0:d} {1:f} {2:f} {3:f}\n'.format(int(lmp_tmp.velocity_info[i,0]),lmp_tmp.velocity_info[i,1],lmp_tmp.velocity_info[i,2],lmp_tmp.velocity_info[i,3]))
        f.write('\n')
    
    if lmp_tmp.bond_info is not None:
        f.write('Bonds\n\n')
        for i in range(len(lmp_tmp.bond_info)):
            f.write('{0:d} {1:d} {2:d} {3:d}\n'.format(int(lmp_tmp.bond_info[i,0]),int(lmp_tmp.bond_info[i,1]),int(lmp_tmp.bond_info[i,2]),int(lmp_tmp.bond_info[i,3])))
        f.write('\n')
    
    if lmp_tmp.angle_info is not None:
        f.write('Angles\n\n')
        for i in range(len(lmp_tmp.angle_info)):
            f.write('{0:d} {1:d} {2:d} {3:d} {4:d}\n'.format(int(lmp_tmp.angle_info[i,0]),int(lmp_tmp.angle_info[i,1]),int(lmp_tmp.angle_info[i,2]),
                                                            int(lmp_tmp.angle_info[i,3]),int(lmp_tmp.angle_info[i,4])))
        f.write('\n')
    
    if lmp_tmp.dihedral_info is not None:
        f.write('Dihedrals\n\n')
        for i in range(len(lmp_tmp.dihedral_info)):
            f.write('{0:d} {1:d} {2:d} {3:d} {4:d} {5:d}\n'.format(int(lmp_tmp.dihedral_info[i,0]),int(lmp_tmp.dihedral_info[i,1]),int(lmp_tmp.dihedral_info[i,2]),
                                                            int(lmp_tmp.dihedral_info[i,3]),int(lmp_tmp.dihedral_info[i,4]),int(lmp_tmp.dihedral_info[i,5])))
        f.write('\n')
    
    if lmp_tmp.improper_info is not None:
        f.write('Impropers\n\n')
        for i in range(len(lmp_tmp.improper_info)):
            f.write('{0:d} {1:d} {2:d} {3:d} {4:d} {5:d}\n'.format(int(lmp_tmp.improper_info[i,0]),int(lmp_tmp.improper_info[i,1]),int(lmp_tmp.improper_info[i,2]),
                                                            int(lmp_tmp.improper_info[i,3]),int(lmp_tmp.improper_info[i,4]),int(lmp_tmp.improper_info[i,5])))
        f.write('\n')
    
    f.close()

def read_mutiple_xyz_orca(file):
    """
    read multiple frames in output xyz of lammps, 
    input: dump_file, !!!! now change to 'custum'
    output: result (index,type_atoms, coors) and t 
    """
    f = open(file)
    lft = list(f)
    f.close()
    lt=[]
    t=[]
    natom = int(lft[0].split()[0])
    for il in range(len(lft)):
        if 'Coordinates' in lft[il]:
            lt.append(il)
#             t.append(lft[il+1].split()[0])

    def read_lf(lf):
        coors = np.zeros([natom,3])
        type_atom = []
        index = []
        l=0
        
        for ia in lf[1:1+natom]:
            coors[l,:] = np.array(ia.split()[1:4:1]).astype('float')
#             coors[l,:] = coors[l,:] - np.array([xlo,ylo,zlo])
            type_atom.append((ia.split()[0]))
            l+=1

        type_atom = np.array(type_atom)
        
        return type_atom,coors

    # lf=[]
    result = []
    for it in range(len(lt)):
        if it==len(lt)-1:
    #         lf.append(lft[lt[it]:])
            result.append(read_lf(lft[lt[it]:]))
        else:
    #         lf.append(lft[lt[it]:lt[it+1]-1])
            result.append(read_lf(lft[lt[it]:lt[it+1]]))
    
#     t=np.array(t)
    return result

def read_info_tmp(file):
    '''
    this is for reading the index_atom_interest.txt
    '''
    f=open(file,'r')
    L = f.readlines()
    f.close()
    idx_atom = [int(x) for x in L[0].split()]
    ebond_r0,ebond_k = [float(x) for x in L[1].split()[:2]]
    energy_loc, strain_loc = [float(x) for x in L[2].split()[:2]]
    
    bond_info = np.loadtxt(file,skiprows=4)
    return idx_atom, ebond_r0, ebond_k, energy_loc, strain_loc, bond_info

def gen_dump2lmp(dump_file,sample_lmp_file,time_frames):
    """
    from dump custum of lammps trajectory to generate full lammps str data with bond info
    """
    result, t = read_mutiple_xyz(dump_file)
    lmp0 = read_lammps_full(sample_lmp_file)
    for i in range(len(time_frames)):
        lmp_tmp = lmp0
        lmp_tmp.atom_info[:,4:7] = result[time_frames[i]][4][lmp0.atom_info[:,0].astype(int)-1]
        lmp_tmp.x = [0,result[time_frames[i]][0][0,0]]
        lmp_tmp.y = [0,result[time_frames[i]][0][1,1]]
        lmp_tmp.z = [0,result[time_frames[i]][0][2,2]]
        write_lammps_full('s{}.dat'.format(time_frames[i]),lmp_tmp)
