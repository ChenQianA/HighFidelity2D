import torch
import pandas as pd

#######################preprocess################################
"""Material Project dataset"""
import gzip, json, pickle

with gzip.open('C:/Users/IanTs/Desktop/桌面/异质结/数据/chi_chen/band_gap_no_structs.gz','rb') as f:
    data=json.load(f)
with open('C:/Users/IanTs/Desktop/桌面/异质结/数据/chi_chen/MP_structures_dict_old','r') as f:
    data_structures=pickle.load(f)

from pymatgen.core import Structure, IStructure
from pymatgen.io.jarvis import JarvisAtomsAdaptor

material_id_list = []
atoms_list = []
properties_list=[]
for i, j in data['pbe'].items():
    material_id_list.append(i)
    atoms_list.append(JarvisAtomsAdaptor.get_atoms(Structure.from_str(data_structures[i], fmt='cif')).to_dict())
    properties_list.append([j])
mp_pbe = pd.DataFrame({'material_id':material_id_list,'atoms':atoms_list,'properties':properties_list})
mp_pbe.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/mp_pbe/raw/data.csv', index=False)

material_id_list = []
atoms_list = []
properties_list=[]
for i, j in data['hse'].items():
    material_id_list.append(i)
    atoms_list.append(JarvisAtomsAdaptor.get_atoms(Structure.from_str(data_structures[i], fmt='cif')).to_dict())
    properties_list.append([j])
mp_hse = pd.DataFrame({'material_id':material_id_list,'atoms':atoms_list,'properties':properties_list})
mp_hse.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/mp_hse/raw/data.csv', index=False)

material_id_list = []
atoms_list = []
properties_list=[]
for i, j in data['gllb-sc'].items():
    material_id_list.append(i)
    atoms_list.append(JarvisAtomsAdaptor.get_atoms(Structure.from_str(data_structures[i], fmt='cif')).to_dict())
    properties_list.append([j])
mp_gllb_sc = pd.DataFrame({'material_id':material_id_list,'atoms':atoms_list,'properties':properties_list})
mp_gllb_sc.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/mp_gllb-sc/raw/data.csv', index=False)

material_id_list = []
atoms_list = []
properties_list=[]
for i, j in data['scan'].items():
    material_id_list.append(i)
    atoms_list.append(JarvisAtomsAdaptor.get_atoms(Structure.from_str(data_structures[i], fmt='cif')).to_dict())
    properties_list.append([j])
mp_scan = pd.DataFrame({'material_id':material_id_list,'atoms':atoms_list,'properties':properties_list})
mp_scan.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/mp_scan/raw/data.csv', index=False)
#################C2DB dataset (z normalized)##################################
from ase.db import connect
from pymatgen.core.periodic_table import Element

db = connect('E:/桌面/异质结/数据/db数据读取/c2db-2022-11-30-big.db')

rows = db.select()
material_id_list = []
atoms_list = []
properties_list=[]
for row in rows:
    if hasattr(row, 'gap'):
        lattice = row.data['structure.json']['1']['cell'].tolist()
        position = row.data['structure.json']['1']['positions']
        elements = row.data['structure.json']['1']['numbers']
        lattice[2][2] = 50.
        position = position.copy()
        position[:, 2] = (25 - position[:, 2].mean()) + position[:, 2]
        atoms = Atoms(lattice_mat=lattice, coords=position, elements=[Element.from_Z(i).symbol for i in elements],
                      cartesian=True)
        atoms_list.append(atoms.to_dict())
        material_id_list.append(row.uid)
        BS = []
        if hasattr(row, 'gap'):
            BS.append(row.gap)
        else:
            BS.append(np.nan)
        if hasattr(row, 'gap_dir'):
            BS.append(row.gap_dir)
        else:
            BS.append(np.nan)
        if hasattr(row, 'vbm') and hasattr(row, 'evac'):
            BS.append(row.vbm - row.evac)
        else:
            BS.append(np.nan)
        if hasattr(row, 'cbm') and hasattr(row, 'evac'):
            BS.append(row.cbm - row.evac)
        else:
            BS.append(np.nan)
        properties_list.append(BS)
c2db_pbe = pd.DataFrame({'material_id': material_id_list, 'atoms': atoms_list, 'properties': properties_list})
c2db_pbe.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/c2db_pbe/raw/data.csv', index=False)


rows = db.select()
material_id_list = []
atoms_list = []
properties_list=[]
for row in rows:
    if hasattr(row, 'gap_hse'):
        lattice = row.data['structure.json']['1']['cell'].tolist()
        position = row.data['structure.json']['1']['positions']
        elements = row.data['structure.json']['1']['numbers']
        lattice[2][2] = 50.
        position = position.copy()
        position[:, 2] = (25 - position[:, 2].mean()) + position[:, 2]
        atoms = Atoms(lattice_mat=lattice, coords=position, elements=[Element.from_Z(i).symbol for i in elements],
                      cartesian=True)
        atoms_list.append(atoms.to_dict())
        material_id_list.append(row.uid)
        BS = []
        if hasattr(row, 'gap_hse'):
            BS.append(row.gap_hse)
        else:
            BS.append(np.nan)
        if hasattr(row, 'gap_dir_hse'):
            BS.append(row.gap_dir_hse)
        else:
            BS.append(np.nan)
        if hasattr(row, 'vbm_hse') and hasattr(row, 'evac'):
            BS.append(row.vbm_hse - row.evac)
        else:
            BS.append(np.nan)
        if hasattr(row, 'cbm_hse') and hasattr(row, 'evac'):
            BS.append(row.cbm_hse - row.evac)
        else:
            BS.append(np.nan)
        properties_list.append(BS)
c2db_hse = pd.DataFrame({'material_id': material_id_list, 'atoms': atoms_list, 'properties': properties_list})
c2db_hse.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/c2db_hse/raw/data.csv', index=False)


rows = db.select()
material_id_list = []
atoms_list = []
properties_list=[]
for row in rows:
    if hasattr(row, 'gap_gw'):
        lattice = row.data['structure.json']['1']['cell'].tolist()
        position = row.data['structure.json']['1']['positions']
        elements = row.data['structure.json']['1']['numbers']
        lattice[2][2] = 50.
        position = position.copy()
        position[:, 2] = (25 - position[:, 2].mean()) + position[:, 2]
        atoms = Atoms(lattice_mat=lattice, coords=position, elements=[Element.from_Z(i).symbol for i in elements],
                      cartesian=True)
        atoms_list.append(atoms.to_dict())
        material_id_list.append(row.uid)
        BS = []
        if hasattr(row, 'gap_gw'):
            BS.append(row.gap_gw)
        else:
            BS.append(np.nan)
        if hasattr(row, 'gap_dir_gw'):
            BS.append(row.gap_dir_gw)
        else:
            BS.append(np.nan)
        if hasattr(row, 'vbm_gw') and hasattr(row, 'evac'):
            BS.append(row.vbm_gw - row.evac)
        else:
            BS.append(np.nan)
        if hasattr(row, 'cbm_gw') and hasattr(row, 'evac'):
            BS.append(row.cbm_gw - row.evac)
        else:
            BS.append(np.nan)
        properties_list.append(BS)
c2db_gw = pd.DataFrame({'material_id': material_id_list, 'atoms': atoms_list, 'properties': properties_list})
c2db_gw.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/c2db_gw/raw/data.csv', index=False)


rows = db.select()
material_id_list = []
atoms_list = []
properties_list=[]
for row in rows:
    if hasattr(row, 'emass_cb_dir1'):
        lattice = row.data['structure.json']['1']['cell'].tolist()
        position = row.data['structure.json']['1']['positions']
        elements = row.data['structure.json']['1']['numbers']
        lattice[2][2] = 50.
        position = position.copy()
        position[:, 2] = (25 - position[:, 2].mean()) + position[:, 2]
        atoms = Atoms(lattice_mat=lattice, coords=position, elements=[Element.from_Z(i).symbol for i in elements],
                      cartesian=True)
        atoms_list.append(atoms.to_dict())
        material_id_list.append(row.uid)
        emass = []
        if hasattr(row, 'emass_vb_dir1'):
            emass.append(row.emass_vb_dir1)
        else:
            emass.append(np.nan)
        if hasattr(row, 'emass_vb_dir2'):
            emass.append(row.emass_vb_dir2)
        else:
            emass.append(np.nan)
        if hasattr(row, 'emass_vb_dir3'):
            emass.append(row.emass_vb_dir3)
        else:
            emass.append(np.nan)
        if hasattr(row, 'emass_cb_dir1'):
            emass.append(row.emass_cb_dir1)
        else:
            emass.append(np.nan)
        if hasattr(row, 'emass_cb_dir2'):
            emass.append(row.emass_cb_dir2)
        else:
            emass.append(np.nan)
        if hasattr(row, 'emass_cb_dir3'):
            emass.append(row.emass_cb_dir3)
        else:
            emass.append(np.nan)
        properties_list.append(emass)
c2db_emass = pd.DataFrame({'material_id': material_id_list, 'atoms': atoms_list, 'properties': properties_list})
c2db_emass.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/c2db_emass/raw/data.csv', index=False)


rows = db.select()
material_id_list = []
atoms_list = []
properties_list=[]
for row in rows:
    if hasattr(row, 'thermodynamic_stability_level') and hasattr(row, 'dynamic_stability_phonons') and hasattr(row, 'dynamic_stability_stiffness'):
        lattice = row.data['structure.json']['1']['cell'].tolist()
        position = row.data['structure.json']['1']['positions']
        elements = row.data['structure.json']['1']['numbers']
        lattice[2][2] = 50.
        position = position.copy()
        position[:, 2] = (25 - position[:, 2].mean()) + position[:, 2]
        atoms = Atoms(lattice_mat=lattice, coords=position, elements=[Element.from_Z(i).symbol for i in elements],
                      cartesian=True)
        atoms_list.append(atoms.to_dict())
        material_id_list.append(row.uid)
        if row.thermodynamic_stability_level == 3 and row.dynamic_stability_phonons == 'high' and row.dynamic_stability_stiffness == 'high':
            properties_list.append([1])
        else:
            properties_list.append([0])
c2db_stability = pd.DataFrame({'material_id':material_id_list,'atoms':atoms_list,'properties':properties_list})
c2db_stability.to_csv('C:/Users/IanTs/Desktop/小样本学习/Adapted_Matformer/data/c2db_stability/raw/data.csv', index=False)


############################2dmatpedia##################################
import json
import numpy as np

data_list=[]
for line in  open('C:/Users/IanTs/Downloads/db.json','r'):
    data_list.append(json.loads(line))

material_id_list = []
atoms_list = []
properties_list=[]
for i in data_list:
    if 'bandstructure' in i.keys():
        material_id_list.append(i['material_id'])
        atoms_list.append(JarvisAtomsAdaptor.get_atoms(IStructure.from_dict(i['structure'])).to_dict())
        property = list(i['bandstructure'].values())[:3]
        for idp in range(len(property)):
            if property[idp] is None:
                property[idp] = np.nan
        properties_list.append(property)
matpedia_optb88 = pd.DataFrame({'material_id':material_id_list,'atoms':atoms_list,'properties':properties_list})
matpedia_optb88.to_csv('C:/Users/eee/Desktop/small_data_learning/SDL/data/matpedia_optb88/raw/data.csv', index=False)

############################jarvis##################################
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data

dft_2d = data(dataset='dft_2d')

material_id_list = []
atoms_list = []
properties_list=[]
for i in dft_2d:
    material_id_list.append(i['jid'])
    atoms_list.append(i['atoms'])
    properties_list.append([i['optb88vdw_bandgap']])
jarvis_optb88 = pd.DataFrame({'material_id':material_id_list,'atoms':atoms_list,'properties':properties_list})
jarvis_optb88.to_csv('C:/Users/eee/Desktop/small_data_learning/SDL/data/jarvis_optb88/raw/data.csv', index=False)

#######################material project##############################
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm

my_api_key = 'BTDlsgr7UTNTrXsA89D0'
api = MPRester(my_api_key)

data_list=[]
for i in tqdm(range(20)):
    entries = api.get_entries({"nelements":i},property_data=list(api.supported_properties))
    data_list.extend(entries)

material_id_list = []
atoms_list = []
properties_list=[]
for i in tqdm(data_list):
    if "e_above_hull" in i.data.keys() and "cif" in i.data.keys():
        material_id_list.append(i.entry_id)
        atoms_list.append(JarvisAtomsAdaptor.get_atoms(IStructure.from_str(i.data["cif"], fmt="cif")).to_dict())
        properties_list.append([i.data["e_above_hull"]])

mp_ehull = pd.DataFrame({'material_id':material_id_list,'atoms':atoms_list,'properties': [str(i).replace('None', 'nan') for i in properties_list]})
mp_ehull.to_csv('C:/Users/eee/Desktop/small_data_2nd_layer_embedding/SDL/data/mp_ehull/raw/data.csv', index=False, na_rep="nan")

with open("C:/Users/eee/Desktop/small_data_2nd_layer_embedding/SDL/data/mp_ehull/mp.pickle", "wb") as f:
    pickle.dump(data_list, f)
