import math, itertools
import torch
from torch_geometric.data import InMemoryDataset, Data
from jarvis.core.atoms import Atoms
import pandas as pd
import numpy as np
from jarvis.core.specie import chem_data, get_node_attributes
from collections import OrderedDict
from ast import literal_eval

def get_edges(lattice, frac_coords, cutoff=8, max_neighbors=12, use_lattice=True):
    """
    lattice : np.array = atoms.lattice_mat
    frac_coords : np.array = atoms.frac_coords
    Construct k-NN edge list.
    """
    all_neighbors = get_all_neighbors(lattice, frac_coords, r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    while min_nbrs < max_neighbors:
        a,b,c = np.linalg.norm(lattice,axis=1).tolist()
        if cutoff < max(a, b, c):
            cutoff = max(a, b, c)
        else:
            cutoff = 2 * cutoff
        all_neighbors = get_all_neighbors(lattice, frac_coords, r=cutoff)
        min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    edge_index_j = []
    edge_index_i = []
    edge_length = []
    images_list =[]
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, distance, image in zip(ids, distances, images):
            edge_index_j.append(site_idx)
            edge_index_i.append(dst)
            edge_length.append(distance)
            images_list.append(image)

        if use_lattice:
            edge_index_j.extend([site_idx]*6)
            edge_index_i.extend([site_idx]*6)
            edge_length.append(np.linalg.norm(np.matmul(np.array([0, 0, 1]), lattice),axis=-1))
            edge_length.append(np.linalg.norm(np.matmul(np.array([0, 1, 0]), lattice),axis=-1))
            edge_length.append(np.linalg.norm(np.matmul(np.array([1, 0, 0]), lattice),axis=-1))
            edge_length.append(np.linalg.norm(np.matmul(np.array([0, 1, 1]), lattice),axis=-1))
            edge_length.append(np.linalg.norm(np.matmul(np.array([1, 0, 1]), lattice),axis=-1))
            edge_length.append(np.linalg.norm(np.matmul(np.array([1, 1, 0]), lattice),axis=-1))
            images_list.append(np.array([0, 0, 1]))
            images_list.append(np.array([0, 1, 0]))
            images_list.append(np.array([1, 0, 0]))
            images_list.append(np.array([0, 1, 1]))
            images_list.append(np.array([1, 0, 1]))
            images_list.append(np.array([1, 1, 0]))

    images_array = np.array(images_list)
    cart_coords = np.matmul(frac_coords, lattice)
    position_vectors = cart_coords[edge_index_j] - cart_coords[edge_index_i] - np.matmul(images_array, lattice)

    edge_index = torch.LongTensor([edge_index_j, edge_index_i])
    edge_length = torch.Tensor(edge_length).unsqueeze(-1)
    position_vectors = torch.Tensor(position_vectors)

    return edge_index, edge_length, position_vectors


def get_all_neighbors(lattice, frac_coords, r=5, bond_tol=0.15):
    """
    Get neighbors for each atom in the unit cell, out to a distance r.
    Contains [index_i, index_j, distance, image] array.
    Adapted from pymatgen.

    lattice : np.array = atoms.lattice_mat
    frac_coords : np.array = atoms.frac_coords
    """
    reciprocal_lattice = 2 * np.pi * np.linalg.inv(lattice).T
    recp_len = np.linalg.norm(reciprocal_lattice,axis=1)
    maxr = np.ceil((r + bond_tol) * recp_len / (2 * math.pi))
    nmin = np.floor(np.min(frac_coords, axis=0)) - maxr
    nmax = np.ceil(np.max(frac_coords, axis=0)) + maxr
    all_ranges = [np.arange(x, y) for x, y in zip(nmin, nmax)]
    matrix = lattice
    cart_coords = np.matmul(frac_coords, lattice)
    neighbors = [list() for _ in range(len(cart_coords))]
    coords_in_cell = cart_coords
    site_coords = cart_coords
    indices = np.arange(len(site_coords))
    for image in itertools.product(*all_ranges):
        coords = np.dot(image, matrix) + coords_in_cell
        z = (coords[:, None, :] - site_coords[None, :, :]) ** 2
        all_dists = np.sum(z, axis=-1) ** 0.5
        all_within_r = np.bitwise_and(all_dists <= r, all_dists > 1e-8)
        for (j, d, within_r) in zip(indices, all_dists, all_within_r):
            for i in indices[within_r]:
                if d[i] > bond_tol:
                    neighbors[i].append([i, j, d[i], image])
    return np.array(neighbors, dtype="object")



class species_atribute(object):
    def __init__(self):
        ###get cgcnn atribute
        max_z = max(v["Z"] for v in chem_data.values())
        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", "cgcnn")
        features = np.zeros((1 + max_z, len(template)))
        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, "cgcnn")
            if x is not None:
                features[z, :] = x
        self.feature_table = features

    def node_feature(self, species_numbers:list=[]):
        features = self.feature_table[species_numbers]
        return torch.Tensor(features)



class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, cutoff=8, max_neighbors=12, use_lattice=True, datatype='r'):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_lattice = use_lattice
        self.atom_attribute = species_atribute()
        self.datatype = datatype
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    def process(self):
        # Read data into huge `Data` list.

        df = pd.read_csv(self.raw_paths[0])

        properties = torch.Tensor([eval(i.replace('nan', 'np.nan')) for i in df.properties.values])
        if self.datatype == 'r':
            mean = properties.nanmean(dim=0)
            std = torch.Tensor(np.nanstd(properties.numpy(), axis=0))
        elif self.datatype == 'c':
            mean = torch.LongTensor([0])
            std = torch.LongTensor([1])
        torch.save({'mean': mean, 'std': std}, self.root+"/mean_std.pt")

        data_list = []
        for _, i in df.iterrows():
            material_id = i.material_id
            atoms = Atoms.from_dict(eval(i.atoms))
            frac_coords = atoms.frac_coords
            lattice = atoms.lattice_mat
            properties = i.properties.replace('nan','np.nan')
            edge_index, length, position_vector = get_edges(lattice, frac_coords,self.cutoff, self.max_neighbors, self.use_lattice)
            data = Data(pos=torch.Tensor(atoms.frac_coords), lattice=torch.Tensor(lattice), x=self.atom_attribute.node_feature(atoms.atomic_numbers),
                        edge_index=edge_index, edge_attr=length, pos_vec=position_vector, y=(torch.Tensor(eval(properties)).unsqueeze(0)-mean)/std,
                        id=material_id, z=torch.IntTensor(atoms.atomic_numbers).unsqueeze(-1))
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])