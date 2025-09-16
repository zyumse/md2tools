import networkx as nx
import numpy as np
import my_common as mc
import extract_local_str as els
import itertools

def create_ini_lammps(file_loc,box_size,density=0.05):
    
    # Parameters
    num_nodes = int(box_size**3*density)
    natoms = num_nodes
    max_degree = 3

    # Initialize the graph
    G = nx.Graph()

    # Uniformly generate random points in a cubic box and add them as nodes
    nodes = {i: (np.random.uniform(0, box_size), np.random.uniform(0, box_size), np.random.uniform(0, box_size))
             for i in range(num_nodes)}
    G.add_nodes_from(nodes)
    
    coors = [] 
    for i in range(len(nodes)): 
        coors.append(np.array(nodes[i]))
    coors = np.vstack(coors)
    # Compute all possible pairs of nodes and the distances between them

    atom_info = np.hstack([
          np.arange(1,natoms+1).reshape(-1,1), # atom id
          np.repeat(1,natoms).reshape(-1,1),   # mol id 
          np.repeat(1,natoms).reshape(-1,1),   # atom type 
          np.repeat(0,natoms).reshape(-1,1),   # atom charge 
          coors,
          np.zeros([natoms,3]),
          ])
    
    lmp = els.lammps(
        natoms=natoms,
        # nbonds=num_edges,
        natom_types = 1,
        # nbond_types = 1,
        x = [0, box_size],
        y = [0, box_size],
        z = [0, box_size], 
        mass = np.array([1,1.000]).reshape(1,2),
        # pair_coeff = pair_coeff,
        # bond_coeff = bond_coeff,
        atom_info= atom_info,
        # bond_info=bond_info
        )
    els.write_lammps_full(file_loc,lmp)
    
def gaussian_probability(distance, sigma=2):
    return np.exp(-0.5 * (distance / sigma) ** 2)

def lmp_add_edge(lmp_file,lmp_file_new,ck_degree=0.9,max_degree=3,sigma=10):
    """
    from the relaxed structure, creating the bonds based on the gaussian probability of distances
    """
    # sigma = 10
    # ck_degree = 0.95
    # lmp_file = 'relax.dat'
    # max_degree = 3

    lmp = els.read_lammps_full(lmp_file)
    atom_info = lmp.atom_info[np.argsort(lmp.atom_info[:,0]),:]
    coors = atom_info[:,4:7]
    box = np.array([[lmp.x[1]-lmp.x[0],0,0],
                    [0,lmp.y[1]-lmp.y[0],0],
                    [0,0,lmp.z[1]-lmp.z[0]]])
    num_nodes = lmp.natoms
    num_edges = int(num_nodes*max_degree/2*ck_degree)

    node_pairs = list(itertools.combinations(range(len(coors)), 2))
    distances = []
    for i,j in node_pairs:
        distances.append(mc.pbc_distance(coors[i],coors[j],box))
    distances = np.array(distances)
    probabilities = gaussian_probability(distances,sigma=sigma).squeeze()
    probabilities /= probabilities.sum()  # Normalize probabilities

    # Add edges based on Gaussian probability until we reach the desired number of edges
    edges_added = 0

    bond_info = []
    A = np.zeros([num_nodes,num_nodes])
    while edges_added < num_edges:
        # Choose a pair based on probability
        chosen_pair = np.random.choice(len(node_pairs), p=probabilities)

        # Try to add this edge if both nodes have less than max_degree
        i, j = node_pairs[chosen_pair]
        if np.sum(A[i]) < max_degree and np.sum(A[j]) < max_degree and A[i,j]==0:
            bond_info.append([edges_added+1,1,i+1,j+1])
            A[i,j] +=1
            A[j,i] +=1
            edges_added+=1
    bond_info = np.array(bond_info)
    lmp_new = els.lammps(
            natoms=lmp.natoms,
            nbonds=num_edges,
            natom_types = 1,
            nbond_types = 1,
            x = lmp.x,
            y = lmp.y,
            z = lmp.z, 
            mass = np.array([1,1.000]).reshape(1,2),
            # pair_coeff = pair_coeff,
            # bond_coeff = bond_coeff,
            atom_info= atom_info,
            bond_info=bond_info
            )
    els.write_lammps_full(lmp_file_new,lmp_new)
    return lmp_new

# def lammps_to_network(lmp_file,):

#     lmp = els.read_lammps_full(lmp_file)


# lammps_to_network 
    
# def generate_lj_table(epsilon, sigma, n, r_min, r_max, num_points, filename):
#     """
#     Generates a LAMMPS-compatible table for a modified Lennard-Jones potential.

#     Parameters:
#     - epsilon: Depth of the potential well (float).
#     - sigma: Finite distance at which the inter-particle potential is zero (float).
#     - n: Exponent for the repulsive term (int).
#     - r_min: Minimum distance for the table (float).
#     - r_max: Maximum distance for the table (float).
#     - num_points: Number of points in the table (int).
#     - filename: Output filename for the table (str).
#     """
    
#     # Define the modified Lennard-Jones potential and force functions
#     def modified_lj_potential(r):
#         attractive = -(sigma / r)**6
#         repulsive = (sigma / r)**n
#         return 4 * epsilon * (repulsive + attractive)

#     def modified_lj_force(r):
#         attractive_force = -6 * (sigma**6) / (r**7)
#         repulsive_force = -n * (sigma**n) / (r**(n+1))
#         return 4 * epsilon * (repulsive_force + attractive_force)

#     # Generate distances
#     r_values = np.linspace(r_min, r_max, num_points)

#     # Open file and write header
#     with open(filename, "w") as file:
#         file.write("# Custom Lennard-Jones potential table\n")
#         file.write("# r V(r) F(r)\n")
        
#         # Calculate and write potential and force values
#         for r in r_values:
#             V = modified_lj_potential(r)
#             F = modified_lj_force(r)
#             file.write(f"{r} {V} {F}\n")

#     print(f"Table written to {filename}")
#     return r_values,modified_lj_potential(r_values), modified_lj_force(r_values)



def network_to_lammps(G,coors,box_size,file_loc='./network.data'):
    natoms = len(G)
    num_edges = len(G.edges)
    atom_info = np.hstack([
          np.arange(1,natoms+1).reshape(-1,1), # atom id
          np.repeat(1,natoms).reshape(-1,1),   # mol id 
          np.repeat(1,natoms).reshape(-1,1),   # atom type 
          np.repeat(0,natoms).reshape(-1,1),   # atom charge 
          coors,
          np.zeros([natoms,3]),
          ])
    bond_idx = np.array(G.edges)
    bond_info = np.hstack([
        np.arange(1,num_edges+1).reshape(-1,1),
        np.ones([num_edges,1]),
        np.array([[i+1,j+1] for i,j in bond_idx])
        ])
    lmp = els.lammps(
        natoms=natoms,
        nbonds=num_edges,
        natom_types = 1,
        nbond_types = 1,
        x = [0, box_size],
        y = [0, box_size],
        z = [0, box_size], 
        mass = np.array([1,1.000]).reshape(1,2),
        # pair_coeff = pair_coeff,
        # bond_coeff = bond_coeff,
        atom_info= atom_info,
        bond_info=bond_info
        )
    els.write_lammps_full(file_loc,lmp)