from SDL.modulus.pyg_att import Matformer4Multi
from SDL.modulus.utils import RBFExpansion, GluLayer
import torch
from torch.nn import ModuleList, Sequential, Linear, SiLU

class MF(torch.nn.Module):
    def __init__(self, atom_input_features=92, node_features=128, edge_features=128, angle_lattice=False,
                 triplet_input_features=40, node_layer_head=4, conv_layers=5, fc_features=128, embedding_layer=2,
                 task=[['r','r'],[1,2]]):
        '''
         :param task: the dict descriobing the task, the first list describes the task type where 'r' means regression,
         'c' means classification, the second task discribes the dimension of the corresponding outputs
        '''
        super().__init__()

        self.task = task
        self.edge_rbf = RBFExpansion(vmin=0, vmax=8.0, bins=edge_features)
        self.model = Matformer4Multi(atom_input_features = atom_input_features, node_features = node_features, edge_features = edge_features,
                                     angle_lattice = angle_lattice, triplet_input_features = triplet_input_features, node_layer_head = node_layer_head,
                                     conv_layers = conv_layers, fc_features = fc_features, embedding_layer = embedding_layer, num_tasks = len(task[0]))
        self.read_out_list = ModuleList([Sequential(Linear(fc_features, fc_features), SiLU(), Linear(fc_features, i)) for i in task[1]])

    def forward(self, batch, task_id:int):
        edge_feat = batch.edge_attr.squeeze()
        batch.edge_attr = self.edge_rbf(edge_feat)
        features = self.model(batch, task_id)
        output = self.read_out_list[task_id](features)

        return output








