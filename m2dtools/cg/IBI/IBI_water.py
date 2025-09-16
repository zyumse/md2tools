import argparse
import os
import pdb
import subprocess
# from scipy.interpolate import CubicSpline
import time
from copy import copy
import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import Parallel, delayed
from scipy.constants import Avogadro as NA
from scipy.signal import savgol_filter

from zy_md import my_common as mc
from zy_md import tools_CG_polymer as tcp
from zy_md import tools_CGwater as tcw
from zy_md import tools_lammps as tl_lmp
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d

def lj(r, epsilon1, epsilon2, sigma, inner_cut=2):
    e_pot = 4 * (epsilon1 * (sigma / r) ** 12 - epsilon2 * (sigma / r) ** 6)
    # linear extrapolation using the slope at r=4
    idx = np.argmin(np.abs(r - inner_cut))
    slope = (e_pot[idx + 1] - e_pot[idx]) / (r[idx + 1] - r[idx])
    e_pot[:idx] = e_pot[idx] + slope * (r[:idx] - r[idx])
    return e_pot

def harmonic(r, k, r0):
    return 0.5 * k * (r - r0) ** 2

def morse(r, D, alpha, r0):
    """
    Morse potential
    :param r: distance
    :param D: depth of the potential well
    :param alpha: width of the potential well
    :param r0: equilibrium distance
    :return: potential energy
    """
    return D * (1 - np.exp(-alpha * (r - r0))) ** 2


def rscale_table_pot(r, e, scale):
    # r_scaled = r*scale # scale from the head
    scale = 1/scale
    # interpolate r and e to a very dense grid
    r_dense = np.linspace(r[0], r[-1], 100000)
    e_dense = np.interp(r_dense, r, e)
    r_scaled_dense = (r_dense - r_dense[-1]) * scale + r_dense[-1]  # scale from the tail
    # corresponding e to the original r in the scaled r
    e_scaled_dense = np.interp(r_dense, r_scaled_dense, e_dense)
    # interpolate back to the original r
    e_scaled = np.interp(r, r_dense, e_scaled_dense)
    e_scaled -= e_scaled[-1]
    # # apply a window function to make the gradient of e_scale to zero at the tail
    # G = np.exp(-1**2/(r-r[-1]+1e-8)**2)
    # e_scaled = e_scaled*G
    return r, e_scaled


def rweight(r, r0, delta=0.5):
    weight = 1 / 2 * (1 + np.tanh((r0 - r) / delta))
    return weight

def rweight_back(r, r0, delta=0.5):
    weight = 1 / 2 * (1 - np.tanh((r0 - r) / delta))
    return weight


def linear_attra(r, e_pot, A, rcut):
    e_pot[r>rcut] = e_pot[r>rcut] + A * (r[-1] - r[r>rcut])
    e_pot[r<=rcut] = e_pot[r<=rcut] + A * (r[-1] - r[r>rcut][0])
    return e_pot

def lowpass(U, r, cutoff):
    U_k = fft(U)
    freqs = fftfreq(len(U), r[1] - r[0])
    U_k[np.abs(freqs) > cutoff] = 0
    return np.real(ifft(U_k))

def shift_table_pot(r, e, delta):
    r_cutoff_right = r[-1]
    r_cutoff_left = r[0]
    # we first think the shift delta is positive
    if delta > 0:
        # shifted r values
        r_shifted = r + delta
        # cut the right end
        e_shifted = e[r_shifted <= r_cutoff_right]
        # linear extrapolation on the left end
        p1 = np.polyfit(r_shifted[:5], e_shifted[:5], 1)
        r_extra = r[r < r_shifted[0]]
        offset = e_shifted[0] - r_shifted[0] * p1[0]
        e_extra = p1[0] * r_extra + offset
        # combine the extrapolated part and the shifted part
        e_new = np.concatenate([e_extra, e_shifted])
    else:
        # shifted r values
        r_shifted = r + delta
        # cut the left end
        e_shifted = e[r_shifted >= r_cutoff_left]
        # linear extrapolation on the right end
        p1 = np.polyfit(r_shifted[-5:], e_shifted[-5:], 1)
        r_extra = r[r > r_shifted[-1]]
        offset = e_shifted[-1] - r_shifted[-1] * p1[0]
        e_extra = p1[0] * r_extra + offset
        # combine the extrapolated part and the shifted part
        e_new = e_new = np.concatenate([e_shifted, e_extra])
    e_new -= e_new[-1]
    # apply a window function to make the gradient of e_new to zero at the tail
    G = np.exp(-(0.5**2) / (r - r[-1] + 1e-8) ** 2)
    e_new = e_new * G
    return r, e_new


def apply_tail_window(r, e, window_width=5):
    """
    Apply a window function to the tail of the potential to make the gradient of the potential to zero at the tail
        :param r: r values
        :param e: e values
        :param window_width: the width of the window function
        :return: r, e
    """
    G = np.exp(-(window_width**2) / (r - r[-1] + 1e-8) ** 2)
    e = e * G
    return r, e


class IBISimulation:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.initialize_parameters()
        self.all_potentials = {}

    def initialize_parameters(self):
        self.turn_on_LC = False
        self.rho = []
        self.run_lammps = self.config["run_lammps_command"]
        # self.alpha1 = self.config["alpha1"]
        # self.alpha2 = self.config["alpha2"]
        self.alpha = self.config["alpha"]
        self.n_iter = self.config["n_iter"]
        # define r_pot in a more flexible way
        self.RDF_cutoff, self.RDF_delta_r = self.config["RDF_cutoff"], self.config["RDF_delta_r"]
        r_tmp = np.arange(0, self.config["RDF_cutoff"], self.config["RDF_delta_r"])
        self.r_pot = (r_tmp[1:] + r_tmp[:-1]) / 2
        self.pair_types = self.config["pair_types"]
        rdf_data = np.loadtxt(self.config["RDF_ref"])
        # cut the RDF reference to the same length as r_pot
        rdf_data = rdf_data[:len(self.r_pot), :]

        self.r_RDF_ref = rdf_data[:,0]
        self.pdf_ref = {}
        self.e_pot = {}
        self.f_pot = {}
        self.effective_rmin = 10
        for i, keyname in enumerate(self.pair_types):
            self.pdf_ref[f"pair_{keyname}"] = rdf_data[:,i+1]
            if self.r_RDF_ref[np.argwhere(rdf_data[:, i+1] > 1e-6)][0] < self.effective_rmin:
                self.effective_rmin = self.r_RDF_ref[np.argwhere(rdf_data[:, i+1] > 1e-6)][0][0]
            if self.config["use_existing_FF"]:
                self.e_pot[f"pair_{keyname}"] = tl_lmp.read_table_pot(self.config["use_existing_FF"], f"pair_{keyname}")[1]
                self.f_pot[f"pair_{keyname}"] = tl_lmp.read_table_pot(self.config["use_existing_FF"], f"pair_{keyname}")[2]
                print(f"Read existing FF for pair_{keyname}", flush=True)
                if "add_LC" in self.config:
                    self.e_pot[f"pair_{keyname}"] = linear_attra(self.r_pot, self.e_pot[f"pair_{keyname}"], self.config["add_LC"], self.r_pot[-1]/5*4)
                    print("Add linear correction", flush=True)
            else:
                if 'lj_params' in self.config:
                    self.e_pot[f"pair_{keyname}"] = lj(self.r_pot, *self.config["lj_params"][i])
                if 'morse_params' in self.config:
                    self.e_pot[f"pair_{keyname}"] = morse(self.r_pot, *self.config["morse_params"][i])
                self.f_pot[f"pair_{keyname}"] = -np.gradient(self.e_pot[f"pair_{keyname}"], self.r_pot)
        print(f"Effective rmin: {self.effective_rmin}", flush=True) 
        # self.density_ref = np.loadtxt(self.config["density_ref"])[0]
        # self.density_ref_std = np.loadtxt(self.config["density_ref"])[1]
        self.density_ref = self.config["density_ref"]

        if isinstance(self.config["n_cpus"], int):
            self.n_cpus = self.config["n_cpus"]
        else:
            self.n_cpus = int(os.getenv("SLURM_CPUS_ON_NODE", default=1))
        self.temp = self.config["temp"]

        self.bond_types = self.config["bond_types"] if 'bond_types' in self.config else []
        if len(self.bond_types) > 0:
            self.alpha_bond = self.config["alpha_bond"]
            bond_data = np.loadtxt(self.config["bond_ref"])
            self.r_bond = bond_data[:, 0]
            self.bond_length_ref = {}
            self.e_bond = {}
            self.f_bond = {}
            for i,keyname in enumerate(self.bond_types):
                self.bond_length_ref[f"bond_{keyname}"] = bond_data[:, i + 1]

            r_bond_dist = np.zeros(len(self.r_bond) + 1)
            r_bond_dist[0] = self.r_bond[0] - 0.5 * (self.r_bond[1] - self.r_bond[0])
            r_bond_dist[-1] = self.r_bond[-1] + 0.5 * (self.r_bond[-1] - self.r_bond[-2])
            r_bond_dist[1:-1] = 0.5 * (self.r_bond[1:] + self.r_bond[:-1])
            self.r_bond_dist = r_bond_dist
            
            for i,keyname in enumerate(self.bond_types):
                if self.config["use_existing_bondFF"]:
                    table = tl_lmp.read_table_pot(self.config["use_existing_bondFF"], f"bond_{keyname}")
                    self.e_bond[f"bond_{keyname}"] = table[1]
                    self.f_bond[f"bond_{keyname}"] = table[2]
                else:
                    self.e_bond[f"bond_{keyname}"] = harmonic(self.r_bond, *self.config["bond_params"][i])
                    self.f_bond[f"bond_{keyname}"] = -np.gradient(self.e_bond[f"bond_{keyname}"], self.r_bond)
        
        self.angle_types = self.config["angle_types"] if 'angle_types' in self.config else []
        if len(self.angle_types) > 0:
            self.alpha_angle = self.config["alpha_angle"]
            angle_data = np.loadtxt(self.config["angle_ref"])
            self.r_angle = angle_data[:, 0]
            self.angle_dist_ref = {}
            self.e_angle = {}
            self.f_angle = {}
            for i, keyname in enumerate(self.angle_types):
                self.angle_dist_ref[f"angle_{keyname}"] = angle_data[:, i + 1]

            r_angle_dist = np.zeros(len(self.r_angle) + 1)
            r_angle_dist[0] = self.r_angle[0] - 0.5 * (self.r_angle[1] - self.r_angle[0])
            r_angle_dist[-1] = self.r_angle[-1] + 0.5 * (self.r_angle[-1] - self.r_angle[-2])
            r_angle_dist[1:-1] = 0.5 * (self.r_angle[1:] + self.r_angle[:-1])
            self.r_angle_dist = r_angle_dist
            
            for i, keyname in enumerate(self.angle_types):
                if self.config["use_existing_bondFF"]:
                    table = tl_lmp.read_table_pot(self.config["use_existing_bondFF"], f"angle_{keyname}")
                    self.e_angle[f"angle_{keyname}"] = table[1]
                    self.f_angle[f"angle_{keyname}"] = table[2]
                else:
                    self.e_angle[f"angle_{keyname}"] = harmonic(self.r_angle, *self.config["angle_params"][i])
                    # self.angle = np.arange(0, 180.5, 1)
                    self.f_angle[f"angle_{keyname}"] = -np.gradient(self.e_angle[f"angle_{keyname}"], self.r_angle)

        self.error_list = {}
        self.density_perror_list = {}

        # density corretion
        if self.config['density_correction']:
            if self.config['PM'] == 'rscale' or self.config['PM'] == 'hybrid':
                self.gamma = self.config["gamma"]
            if self.config['PM'] == 'linear':
                self.LC_A = self.config["A"]

        # smooth
        if 'smooth_sigma' in self.config:
            self.smooth_sigma = self.config["smooth_sigma"]
        else:
            self.smooth_sigma = 3
        print(f"Smooth sigma: {self.smooth_sigma}", flush=True)

        self.lambda_value = self.config["lambda"] if "lambda" in self.config else 23.15
        self.tetra_aa = self.config["tetra_aa"] if "tetra_aa" in self.config else 0.556
        self.alpha_lambda = 0.1

    def run_simulation(self, start=1):
        for i in range(start, self.n_iter + start):
            start_time = time.time()
            self.i_iter = i
            self.run_iteration()
            end_time = time.time()
            print(f"Iteration {i}, Time taken: {end_time - start_time:.2f} seconds", flush=True)
            if self.config["is_lr_decay"]:
                if i % self.config["decay_freq"] == 0 and np.abs(self.alpha) > np.abs(
                    self.config["min_alpha"]
                ):
                    self.alpha *= self.config["decay_rate"]

    def run_iteration(self):
        # self.save_potentials(iteration_number)
        self.directory = f"CG{self.i_iter}"
        os.makedirs(self.directory, exist_ok=True)
        # if os.path.exists(f"CG{self.i_iter-1}/log"):
        #     os.system(f"cp CG{self.i_iter-1}/out.dat {self.directory}/tmp.dat")
        # else:
        os.system(f"cp {self.config['init_str']} {self.directory}/tmp.dat")
        os.system(f"cp {self.config['lmp_input']} {self.directory}/")
        os.system(f"cp mW_real.sw {self.directory}/mW_real.sw")
        # change XXX in mW_real.sw to self.lambda_value using sed
        os.system(f"sed -i 's/XXX/{self.lambda_value}/g' {self.directory}/mW_real.sw")
        r_pot_write = copy(self.r_pot)
        r_pot_write[0] = 1e-8
        tmp_args = [(r_pot_write, self.e_pot[f"pair_{key}"], self.f_pot[f"pair_{key}"], f"pair_{key}") for key in self.pair_types]
        if len(self.bond_types) > 0:
            r_bond_write = copy(self.r_bond)
            r_bond_write[0] = 1e-8
            r_bond_write[-1] = 100
            for key in self.bond_types:
                tmp_args.append((r_bond_write, self.e_bond[f"bond_{key}"], self.f_bond[f"bond_{key}"], f"bond_{key}"))
        if len(self.angle_types) > 0:
            for key in self.angle_types:
                tmp_args.append((self.r_angle, self.e_angle[f"angle_{key}"], self.f_angle[f"angle_{key}"], f"angle_{key}"))
        tcp.write_pot_table(f"{self.directory}/pot.table", tmp_args)

        subprocess.run(f"{self.run_lammps} -in {self.config['lmp_input']} > log", shell=True, cwd=self.directory)

        # if self.config["is_bonded"]:
        self.process_results_bonded()
        # else:
        #     self.process_results_nonbonded(directory, iteration_number)

    def process_results_bonded(self):
        dump_file = os.path.join(self.directory, "dump.xyz")
        output_file = os.path.join(self.directory, "out.dat")
        log_file = os.path.join(self.directory, "log.lammps")

        # Read the LAMMPS dump and output files
        frame, _, L_list = tl_lmp.read_lammps_dump_custom(dump_file)
        nframes = len(frame)
        self.lmp = tl_lmp.read_lammps_full(output_file)
        self.lmp.atom_info = self.lmp.atom_info[np.argsort(self.lmp.atom_info[:, 0]), :]
        # natom_types = lmp.natom_types
        mass_AA = self.lmp.mass[self.lmp.atom_info[:, 2].astype(int)-1, 1].astype(float)
        self.total_mass = np.sum(mass_AA)
        self.type_atom = self.lmp.atom_info[:, 2]

        self.compute_RDF_matrix = {}
        # if self.lmp.nbonds > 0:
        for i in range(self.lmp.natom_types):
            for j in range(i, self.lmp.natom_types):
                idx_type1 = np.where(self.type_atom == i+1)[0]
                idx_type2 = np.where(self.type_atom == j+1)[0]
                mol_type1 = self.lmp.atom_info[idx_type1, 1]
                mol_type2 = self.lmp.atom_info[idx_type2, 1]
                self.compute_RDF_matrix[(i+1, j+1)] = (mol_type1[:, None] != mol_type2[None, :]).astype(int)

        # Map the process_frame function across all frames
        args = [
            (self.lmp, frame[i], L_list[i])
            for i in np.arange(int(nframes / 2), nframes)
        ]
        results = Parallel(n_jobs=self.n_cpus)(delayed(self.process_frame)(*arg) for arg in args)

        # Collect the results, directly for average
        property_avg = {}
        for result in results:
            for key1 in result.keys():
                if type(result[key1]) == dict:
                    if key1 not in property_avg.keys():
                        property_avg[key1] = {}
                    for key2 in result[key1].keys():
                        if key2 not in property_avg[key1]:
                            property_avg[key1][key2] = []
                        property_avg[key1][key2].append(result[key1][key2])
                else:
                    if key1 == 'tetra':
                        if 'tetra' not in property_avg.keys():
                            property_avg['tetra'] = []
                        property_avg['tetra'].append(result['tetra'][1])
                    else:
                        property_avg[key1] = []
                        property_avg[key1].append(result[key1])
        # Compute the average
        for key1 in property_avg.keys():
            if type(property_avg[key1]) == dict:
                for key2 in property_avg[key1].keys():
                    property_avg[key1][key2] = np.array(property_avg[key1][key2])
                    property_avg[key1][key2] = np.mean(property_avg[key1][key2], axis=0)
            else:
                if key1 == 'tetra':
                    property_avg[key1] = np.array(property_avg[key1])
                    # insert result['tetra'][0] as the first element
                    property_avg[key1] = (results[0]['tetra'][0], np.mean(property_avg[key1], axis=0))
                else:
                    property_avg[key1] = np.mean(property_avg[key1], axis=0)

        log = tl_lmp.read_log_lammps(log_file)
        if self.config["target"] == "density":
            self.target = np.array(log[-1]["Density"])
        elif self.config["target"] == "pressure":
            self.target = np.array(log[-1]["Press"])
        else:
            raise ValueError("Target not supported")

        self.box_size = np.mean(log[-1]["Lx"])

        property_avg["target"] = np.mean(self.target)

        # Save pdfs
        if 'RDF' in property_avg.keys():
            g_list = []
            for key in property_avg["RDF"].keys():
                g_list.append(property_avg["RDF"][key])
            g_list = np.array(g_list)
            np.savetxt(os.path.join(self.directory, "pdf.txt"), np.column_stack((self.r_RDF_ref, g_list.T)))

        if 'bl' in property_avg.keys():
            bl_list = []
            for key in property_avg["bl"].keys():
                bl_list.append(property_avg["bl"][key])
            bl_list = np.array(bl_list)
            np.savetxt(os.path.join(self.directory, "bond_length_dist.txt"), np.column_stack((self.r_bond, bl_list.T)),)

        if 'angle' in property_avg.keys():
            angle_list = []
            for key in property_avg["angle"].keys():
                angle_list.append(property_avg["angle"][key])
            angle_list = np.array(angle_list)
            np.savetxt(os.path.join(self.directory, "angle_dist.txt"), np.column_stack((self.r_angle, angle_list.T)))

        if 'tetra' in property_avg.keys():
            np.savetxt(os.path.join(self.directory, "tetra_dist.txt"), np.column_stack((property_avg['tetra'][0], property_avg['tetra'][1])))

        self.property = property_avg
        self.plot_results()

        self.update_potentials()

    def process_frame(self, lmp, frame, L):
        # property to compute: density, all RDFs, all bond length distributions, all angle distributions
        property = {}
        box_size = L[0][1] - L[0][0]
        box = np.array([[box_size, 0, 0], [0, box_size, 0], [0, 0, box_size]])
        atom_types = lmp.atom_info[:, 2].astype(int)
        natom_types = lmp.natom_types
        bond_atom_idx = []
        if lmp.nbonds > 0:
            bond_atom_idx = lmp.bond_info[:, 2:4] - 1
            bond_atom_idx = bond_atom_idx.astype(int)

        coors = np.hstack((
                frame["x"].to_numpy().reshape(-1, 1),
                frame["y"].to_numpy().reshape(-1, 1),
                frame["z"].to_numpy().reshape(-1, 1),
                ))

        # Compute the density
        V = box_size**3
        rho = self.total_mass / V * 1e24 / NA
        property["density"] = rho

        # Compute the RDF
        # n_RDFs = len(self.pair_types)
        # assert n_RDFs == natom_types * (natom_types + 1) / 2, "RDFs not equal to atom types"
        coors_type = {}
        RDFs = {}
        for i in range(natom_types):
            coors_type[i] = coors[atom_types == i+1, :]
        for i in range(natom_types):
            for j in range(i, natom_types):
                keyname = f"pair_{i+1}{j+1}"
                if self.config["RDF_type"] == 1: # non-bonded RDF
                    if lmp.nbonds > 0:
                        bond_atom_idx_type = []
                        idx_typei = np.where(atom_types == i + 1)[0]
                        idx_typej = np.where(atom_types == j + 1)[0]
                        for k, l in bond_atom_idx:
                            if atom_types[k] == i+1 and atom_types[l] == j+1:
                                bond_atom_idx_type.append([np.where(idx_typei==k)[0], np.where(idx_typej==l)[0]])
                            if atom_types[k] == j+1 and atom_types[l] == i+1:
                                bond_atom_idx_type.append([np.where(idx_typei==l)[0], np.where(idx_typej==k)[0]])
                    else:
                        bond_atom_idx_type = None
                    if keyname in self.pdf_ref.keys():
                        _, g, _, _ = tcp.pdf_sq_cross(box, coors_type[i], coors_type[j], bond_atom_idx_type, r_cutoff=self.RDF_cutoff, delta_r=self.RDF_delta_r)
                        RDFs[keyname] = g
                elif self.config["RDF_type"] == 2: # different molecules
                    if keyname in self.pdf_ref.keys(): 
                        _, g, _, _ = tcp.pdf_sq_cross_mask(box, coors_type[i], coors_type[j], self.compute_RDF_matrix[(i+1, j+1)], r_cutoff=self.RDF_cutoff, delta_r=self.RDF_delta_r)
                        RDFs[keyname] = g
        property["RDF"] = RDFs

        # Compute the bond length distribution
        if len(self.bond_types) > 0:
            bl_dict = {}
            # nbond_types = len(self.bond_types)
            for i in self.bond_types:
                keyname = f"bond_{i}"
                if keyname not in self.bond_length_ref.keys():
                    continue
                bond_atom_idx_btype = bond_atom_idx[lmp.bond_info[:, 1] == int(i)]
                bond_length = tcp.compute_bond_length(coors, bond_atom_idx_btype, box_size)
                a, _ = np.histogram(bond_length, bins=self.r_bond_dist, density=True)
                bl_dict[keyname] = a
            property["bl"] = bl_dict

        # Compute angle distribution
        if len(self.angle_types) > 0:
            angle_dict = {}
            # nangle_types = len(self.angle_types)
            angle_atoms = np.array(lmp.angle_info[:, 2:5] - 1, dtype=int)
            for i in self.angle_types:
                keyname = f"angle_{i}"
                if keyname not in self.angle_dist_ref.keys():
                    continue
                angle_atoms_atype = angle_atoms[lmp.angle_info[:, 1] == int(i)].astype(int)
                angle = tcp.compute_angle(coors, angle_atoms_atype, box_size)
                aa, _ = np.histogram(angle, bins=self.r_angle_dist, density=True)
                angle_dict[keyname] = aa
            property["angle"] = angle_dict

        # compute tetrahedral order parameter based on type 1 atoms
        if 'tetra' in self.config and self.config['tetra']:
            q_list = tcw.compute_tetrahedral_order(coors_type[0], box, cutoff=4)
            q_hist, q_bins = np.histogram(q_list, bins=np.arange(0,1,0.01), density=True)
            q_r = 0.5 * (q_bins[1:] + q_bins[:-1])
            property['tetra'] = (q_r, q_hist)

        return property

    def _smooth_potential(self, r, e, num_points=10000, sigma=3):
        # interpolate and then smooth the data
        r_new = np.linspace(r[0], r[-1], num_points)
        interp_func = interp1d(r, e, kind='linear')
        e_interp = interp_func(r_new)
        # Gaussian smoothing
        e_smooth = gaussian_filter1d(e_interp, sigma=sigma)

        # # cubic spline smoothing
        # # only apply to the range beyond self.effective_rmin
        # length_tmp = self.effective_rmin - 0.5
        # r_new1 = r_new[r_new > length_tmp]
        # e_interp1 = e_interp[r_new > length_tmp]
        # e_smooth1 = UnivariateSpline(r_new1, e_interp1, s=sigma, ext=3)(r_new1)
        # e_smooth = np.zeros_like(e_interp)
        # e_smooth[r_new > length_tmp] = e_smooth1
        # e_interp0 = e_interp[r_new <= length_tmp]
        # e_interp0 = e_interp0 + e_smooth1[0] - e_interp1[0]
        # e_smooth[r_new <= length_tmp] = e_interp0

        interp_back = interp1d(r_new, e_smooth, kind='linear')
        e_smoothed_on_old_r = interp_back(r)
        f_smooth = -np.gradient(e_smoothed_on_old_r, r)
        return e_smoothed_on_old_r, f_smooth

    def update_potentials(self):
        rho = self.property["target"]
        self.rho.append(rho)
        # density_perror = np.abs(rho / self.density_ref - 1)
        if len(self.rho)>2 and (rho-self.density_ref)*(self.rho[-2]-self.density_ref)<0: 
            self.turn_on_LC = True
            print("Turn on linear correction", flush=True)

        # based on PDF
        new_weight = np.ones_like(self.r_pot)
        if "use_weight" in self.config and self.config["use_weight"]:
            if self.config["use_weight"] == "r-1":
                new_weight = np.ones_like(self.r_pot) / self.r_pot * 5
            elif self.config["use_weight"] == "r-2":
                new_weight = np.ones_like(self.r_pot) / self.r_pot**2 * 5**2
            elif self.config["use_weight"] == "r-3":
                new_weight = np.ones_like(self.r_pot) / self.r_pot**3 * 5**3
            elif self.config["use_weight"] == "grad":
                r_grad = np.min([self.config["grad_r1"], (self.config["grad_r1"] - self.config["grad_r0"]) * self.i_iter / self.config["grad_steps"] + self.config["grad_r0"]])
                new_weight = rweight(self.r_pot, r_grad, 0.5)
            elif self.config["use_weight"] == "LD":
                new_weight = np.ones_like(self.r_pot)
                min_r = np.min(self.r_pot[self.pdf_ref["pair_11"] > 0])
                new_weight[self.r_pot > min_r] = 1 - (self.r_pot[self.r_pot > min_r] - min_r) / (self.r_pot[-1] - min_r)
            elif self.config["use_weight"] == "LI":
                new_weight = np.zeros_like(self.r_pot)
                min_r = np.min(self.r_pot[self.pdf_ref["pair_11"] > 0])
                new_weight[self.r_pot > min_r] = (self.r_pot[self.r_pot > min_r] - min_r) / (self.r_pot[-1] - min_r)
            elif self.config["use_weight"] == "grad_back":
                r_grad = np.min([self.config["grad_r1"], (self.config["grad_r1"] - self.config["grad_r0"]) * self.i_iter / self.config["grad_steps"] + self.config["grad_r0"]])
                new_weight = rweight_back(self.r_pot, r_grad, 0.5)
        else:
            if "use_SR" in self.config and self.config["use_SR"]:
                new_weight = np.ones_like(self.r_pot)
                new_weight[self.r_pot > self.config["use_SR"]] = 0
        # print(new_weight, flush=True)

        avg_tetra_cg = np.sum(self.property['tetra'][0]*self.property['tetra'][1])/np.sum(self.property['tetra'][1]) if 'tetra' in self.property.keys() else 0
        print(f'Avg tetra CG: {avg_tetra_cg}', flush=True)
        
        self.lambda_value *= (self.tetra_aa/avg_tetra_cg)**self.alpha_lambda

        error_PDF = {}
        perror_PDF = {}
        update_pdf = False if self.config['target']=='density' and (rho < 0.2) else True
        for key in self.property["RDF"].keys():
            if key not in self.pdf_ref.keys():
                raise ValueError(f"Key {key} not in reference RDF")
            error_PDF[key] = np.log((self.property["RDF"][key] + 1e-8)/(self.pdf_ref[key] + 1e-8))
            # # limite this error within np.log(100) and np.log(0.01)
            # error_PDF[key][error_PDF[key] > np.log(10)] = np.log(10)
            # error_PDF[key][error_PDF[key] < np.log(0.1)] = np.log(0.1)
            # error_PDF[key] = self._smooth_potential(self.r_pot, error_PDF[key])[0] * new_weight
            error_PDF[key] = error_PDF[key] * new_weight
            perror_PDF[key] = np.mean(np.abs(self.property["RDF"][key] - self.pdf_ref[key]))
            if update_pdf:
                self.e_pot[key] -= error_PDF[key] * self.alpha * self.config["temp"]
            self.e_pot[key] -= self.e_pot[key][-1]

            # # define new weight, which is the maximum of the RDF or RDF_ref
            # vec = np.vstack((np.mean(g1_list, axis=0), self.RDF_ref))
            # # new_weight = np.max(vec, axis=0)/1*self.r_pot/self.r_pot[-1]
            # if 'weight_normalize' in self.config and self.config['weight_normalize']:
            #     new_weight = new_weight * np.max(vec, axis=0)

            # if ("tetra" in self.)
            
            if (
                self.config["density_correction"]
                and self.i_iter % self.config["density_correction_freq"] == 0
            ):
                if self.config["PM"] == "rscale":
                    if len(self.rho) > 2:
                        tmp = (self.rho[-2]-self.density_ref)/(self.rho[-1]-self.density_ref) # tmp > 1, approaching density ref, good; 
                        if tmp > 0 and tmp < 1: # away from density ref
                            self.gamma = self.gamma*1.01
                        elif tmp < 0: # ocillation
                            self.gamma = self.gamma/1.01
                    print(f'rscale is on, gamma={self.gamma}', flush=True)
                    scale = (rho / self.density_ref) ** float(self.gamma)
                    # scale = 1 / scale
                    if scale > 1.01:
                        scale = 1.01
                    if scale < 0.99:
                        scale = 0.99
                    self.e_pot[key] = rscale_table_pot(self.r_pot, self.e_pot[key], scale)[1]

                elif self.config["PM"] == "hybrid":
                    if len(self.rho) > 2:
                        tmp = (self.rho[-2]-self.density_ref)/(self.rho[-1]-self.density_ref) # tmp > 1, approaching density ref, good; 
                        if tmp > 0 and tmp < 1: # away from density ref
                            self.gamma = self.gamma*1.01
                        elif tmp < 0: # ocillation
                            self.gamma = self.gamma/1.01
                    print(f'rscale is on, gamma={self.gamma}', flush=True)
                    scale = (rho / self.density_ref) ** float(self.config["gamma"])
                    # scale = 1 / scale
                    if scale > 1.01:
                        scale = 1.01
                    if scale < 0.99:
                        scale = 0.99
                    self.e_pot[key] = rscale_table_pot(self.r_pot, self.e_pot[key], scale)[1]
                    if self.turn_on_LC:
                        print('LC is on', flush=True)
                        scale = np.log(rho/self.density_ref) * self.config["A"]
                        if self.config['target'] == "pressure":
                            scale = -scale
                        self.e_pot[key] = linear_attra(self.r_pot, self.e_pot[key], scale, self.config["LC_rcut"])
                elif self.config["PM"] == "linear":
                    if len(self.rho) > 2:
                        tmp = (self.rho[-2]-self.density_ref)/(self.rho[-1]-self.density_ref) # tmp > 1, approaching density ref, good; 
                        if tmp > 0 and tmp < 1: # away from density ref
                            self.LC_A = self.LC_A*1.5
                        elif tmp < 0: # ocillation
                            self.LC_A = self.LC_A/1.5
                    print('LC is on', flush=True)
                    scale = np.log(rho/self.density_ref) * self.LC_A
                    if self.config['target'] == "pressure":
                        scale = -scale
                    if scale > 0.001:
                        scale = 0.001
                    if scale < -0.001:
                        scale = -0.001
                    self.e_pot[key] = linear_attra(self.r_pot, self.e_pot[key], scale, self.config["LC_rcut"])
                    print('linear correction scale=', scale, flush=True)
                elif self.config["PM"] == "postlinear":
                    self.LC_A = self.config["A"]
                    print('LC is on', flush=True)
                    scale = np.log(rho/self.density_ref) * self.LC_A
                    if self.config['target'] == "pressure":
                        scale = -scale
                    if scale > 0.01:
                        scale = 0.01
                    if scale < -0.01:
                        scale = -0.01
                    self.e_pot[key] = linear_attra(self.r_pot, self.e_pot[key], scale, self.config["LC_rcut"])
                    print('linear correction scale=', scale, flush=True)
                elif self.config["PM"] == "shift":
                    delta = (rho - self.density_ref) * self.config["delta"]
                    if delta > 0.1:
                        delta = 0.1
                    if delta < -0.1:
                        delta = -0.1
                    self.e_pot[key] = shift_table_pot(self.r_pot, self.e_pot[key], delta)[1]
                else:
                    raise ValueError("PM method not supported")

            # if rho < 0.2 and self.config['target'] == 'density':
            #     scale = (rho - self.density_ref) * self.config["A"]
            #     if scale > 0.01:
            #         scale = 0.01
            #     if scale < -0.01:
            #         scale = -0.01
            #     self.e_pot[key] = linear_attra(self.r_pot, self.e_pot[key], scale, 0)

            # if self.i_iter % self.config["smooth_freq"] == 0:
            #     if self.config['smooth_method'] == 'savitzky-golay':
            #         self.e_pot[key][self.r_pot>1] = savgol_filter(
            #             self.e_pot[key][self.r_pot>1],
            #             self.config["smooth_wl"],
            #             self.config["smooth_order"],
            #             mode="nearest"
            #         )
            #     if self.config['smooth_method'] == 'spline':
            #         self.e_pot[key] = UnivariateSpline(self.r_pot, self.e_pot[key], s=self.config['smooth_s'], k=3)(self.r_pot)
            #     if self.config['smooth_method'] == 'gaussian':
            #         self.e_pot[key] = gaussian_filter1d(self.e_pot[key], sigma=self.config['smooth_sigma'])
            #     if self.config['smooth_method'] == 'fft':
            #         self.e_pot[key] = lowpass(self.e_pot[key], self.r_pot, cutoff=self.config['fft_cutoff'])

            # new simplifeid smooth
            if self.i_iter % self.config["smooth_freq"] == 0:
                self.e_pot[key] = self._smooth_potential(self.r_pot, self.e_pot[key], sigma=self.smooth_sigma)[0]
            self.f_pot[key] = -np.gradient(self.e_pot[key], self.r_pot)

        if 'bl' in self.property.keys():
            error_bond = {}
            perror_bond = {}
            for key in self.property["bl"].keys():
                if key not in self.bond_length_ref.keys():
                    raise ValueError(f"Key {key} not in reference bond length")
                vec = np.vstack((self.property["bl"][key], self.bond_length_ref[key]))
                # new_weight = np.max(vec, axis=0) / np.max(vec)
                error_bond[key] = np.log((self.property["bl"][key] + 1e-8) / (self.bond_length_ref[key] + 1e-8)) #* new_weight
                perror_bond[key] = np.mean(np.abs(self.property["bl"][key] - self.bond_length_ref[key]))
                self.e_bond[key] -= error_bond[key] * self.alpha_bond * self.config["temp"]
                if self.config['smooth_bond'] and self.i_iter % self.config["smooth_freq"] == 0:
                    # self.e_bond[key] = gaussian_filter1d(self.e_bond[key], sigma=self.config['smooth_sigma'])
                    self.e_bond[key] = self._smooth_potential(self.r_bond, self.e_bond[key])[0]
                self.f_bond[key] = -np.gradient(self.e_bond[key], self.r_bond)

        if "angle" in self.property.keys():
            error_angle = {}
            perror_angle = {}
            for key in self.property["angle"].keys():
                if key not in self.angle_dist_ref.keys():
                    raise ValueError(f"Key {key} not in reference angle")
                vec = np.vstack((self.property["angle"][key], self.angle_dist_ref[key]))
                new_weight = np.max(vec, axis=0) / np.max(vec)
                error_angle[key] = np.log((self.property["angle"][key] + 1e-8) / (self.angle_dist_ref[key] + 1e-8)) * new_weight
                perror_angle[key] = np.mean(np.abs(self.property["angle"][key] - self.angle_dist_ref[key]))
                self.e_angle[key] -= error_angle[key] * self.alpha_angle * self.config["temp"]
                if self.config['smooth_angle'] and self.i_iter % self.config["smooth_freq"] == 0:
                    # self.e_angle[key] = gaussian_filter1d(self.e_angle[key], sigma=self.config['smooth_sigma'])
                    self.e_angle[key] = self._smooth_potential(self.r_angle, self.e_angle[key])[0]
                self.f_angle[key] = -np.gradient(self.e_angle[key], self.r_angle)

        # total_error = perror1 + perror2 + perror3 + density_perror
        # total_error = total_error / 4

        # self.error_list[iteration_number] = total_error
        # self.density_perror_list[iteration_number] = density_perror

        with open("log.txt", "a") as f:
            f.write(f"Iteration={self.i_iter} ")
            f.write(f"Target={self.property['target']:.6f} ")
            for key in error_PDF.keys():
                f.write(f"PDF error {key}={perror_PDF[key]:.6f} ")
            if 'bl' in self.property.keys():
                for key in perror_bond.keys():
                    f.write(f"Bond error {key}={perror_bond[key]:.6f} ")
            if 'angle' in self.property.keys():
                for key in perror_angle.keys():
                    f.write(f"Angle error {key}={perror_angle[key]:.6f} ")
            f.write(f"Alpha: {self.alpha:.6f} ")
            f.write("\n")

    def plot_results(self):
        
        # determine how many do we need to plot 
        # density, RDFs, bonds, angles
        n_subplots = 2
        if len(self.bond_types) > 0:
            n_subplots += 1
        if len(self.angle_types) > 0:
            n_subplots += 1
        fig, ax = plt.subplots(1, n_subplots, figsize=[2*n_subplots, 3], dpi=300)
        i_plot = 0
        # Plot target
        ax[i_plot].plot(np.arange(len(self.target)), self.target, label="Target")
        ax[i_plot].plot([0, len(self.target)], [self.density_ref, self.density_ref], "k--", lw=1)
        ax[i_plot].set_xlabel("Iteration")
        ax[i_plot].set_ylabel("Target")
        i_plot += 1

        # Plot RDFs
        for i, key in enumerate(self.property["RDF"].keys()):
            g = self.property["RDF"][key]
            ax[i_plot].plot(self.r_RDF_ref, g + 0.5*i, label=key)
            ax[i_plot].plot(self.r_RDF_ref, self.pdf_ref[key] + 0.5*i, "k--", lw=1)
        ax[i_plot].set_xlabel("r")
        ax[i_plot].set_ylabel("g(r)")
        i_plot += 1

        # plot bonds
        if 'bl' in self.property.keys():
            r_min, r_max = 10, 0
            for i, key in enumerate(self.property["bl"].keys()):
                bl = self.property["bl"][key]
                ax[i_plot].plot(self.r_bond, bl, label=key)
                ax[i_plot].plot(self.r_bond, self.bond_length_ref[key], "k--", lw=1)
                if self.r_bond[self.bond_length_ref[key] > 0][-1] > r_max:
                    r_max = self.r_bond[self.bond_length_ref[key] > 0][-1]
                if self.r_bond[self.bond_length_ref[key] > 0][0] < r_min:
                    r_min = self.r_bond[self.bond_length_ref[key] > 0][0]
            ax[i_plot].set_xlabel("l")
            ax[i_plot].set_ylabel("P(l)")
            ax[i_plot].set_xlim(r_min, r_max)
            i_plot += 1

        # plot angles
        if 'angle' in self.property.keys():
            for i, key in enumerate(self.property["angle"].keys()):
                angle = self.property["angle"][key]
                ax[i_plot].plot(self.r_angle, angle, label=key)
                ax[i_plot].plot(self.r_angle, self.angle_dist_ref[key], "k--", lw=1)
            ax[i_plot].set_xlabel("theta")
            ax[i_plot].set_ylabel("P(theta)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.directory, "results.jpg"))
        plt.close(fig)

        # plot tetrahedral order parameter
        if 'tetra' in self.property.keys():
            fig, ax = plt.subplots(1, 1, figsize=[3, 3], dpi=300)
            q_r, q_hist = self.property['tetra']
            ax.plot(q_r, q_hist, label='CG')
            ax.set_xlabel('q')
            ax.set_ylabel('P(q)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.directory, "tetra.jpg"))
            plt.close(fig)

        # Plot Potential Energy (van der Waals)
        n_subplots = 1
        if len(self.bond_types) > 0:
            n_subplots += 1
        if len(self.angle_types) > 0:
            n_subplots += 1
        
        fig, ax = plt.subplots(1, n_subplots, figsize=[2*n_subplots, 3], dpi=300)
        if n_subplots > 1:
            i_plot = 0
            y_min = 0
            for key in self.e_pot.keys():
                ax[i_plot].plot(self.r_pot, self.e_pot[key], label=key)
                if self.e_pot[key].min() < y_min:
                    y_min = self.e_pot[key].min()
            ax[i_plot].set_ylim(y_min, 2)
            ax[i_plot].legend(fontsize=5)
            ax[i_plot].set_xlabel("r")
            ax[i_plot].set_ylabel("E(r)")
            i_plot += 1

            # Plot Bond Potential
            if len(self.bond_types) > 0:
                for key in self.e_bond.keys():
                    ax[i_plot].plot(self.r_bond, self.e_bond[key], label=key)
                ax[i_plot].legend(fontsize=5)
                ax[i_plot].set_xlabel("l")
                ax[i_plot].set_ylabel("E(l)")
                i_plot += 1

            # Plot Angle Potential
            if len(self.angle_types) > 0:
                for key in self.e_angle.keys():
                    ax[i_plot].plot(self.r_angle, self.e_angle[key], label=key)
                ax[i_plot].legend(fontsize=5)
                ax[i_plot].set_xlabel("theta")
                ax[i_plot].set_ylabel("E(theta)")
        else:
            y_min = 0
            for key in self.e_pot.keys():
                ax.plot(self.r_pot, self.e_pot[key], label=key)
                if self.e_pot[key].min() < y_min:
                    y_min = self.e_pot[key].min()
            ax.set_ylim(y_min, 2)
            ax.legend(fontsize=5)
            ax.set_xlabel("r")
            ax.set_ylabel("E(r)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.directory, "potentials.jpg"))
        plt.close(fig)

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--start", type=int, default=1, help="Start iteration")
    args = parser.parse_args()
    if args.config:
        config_file = args.config
    if args.start:
        start = args.start
    simulation = IBISimulation(config_file)
    print(simulation.n_cpus, simulation.n_iter)
    simulation.run_simulation(start)
    # save simulation
    joblib.dump(simulation, "simulation.pkl")
