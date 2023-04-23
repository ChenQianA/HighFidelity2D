"""Implementation based on the template of ALIGNN."""

import torch
from torch import nn
from .utils import RBFExpansion
from torch_scatter import scatter
from .transformer import MatformerConv


class Matformer4Stem(nn.Module):
    """att pyg implementation."""

    def __init__(self, atom_input_features=92, node_features=128, edge_features=128, angle_lattice=False,
                 triplet_input_features=40, node_layer_head=4, conv_layers=5, fc_features=128):
        """Set up att modules."""
        super().__init__()
        self.atom_embedding = nn.Linear(atom_input_features, node_features)
        self.rbf = nn.Sequential(RBFExpansion(vmin=0, vmax=8.0, bins=edge_features),
                                 nn.Linear(edge_features, node_features),
                                 nn.Softplus(),
                                 nn.Linear(node_features, node_features))

        self.angle_lattice = angle_lattice
        if self.angle_lattice: ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(RBFExpansion(vmin=0, vmax=50, bins=edge_features*4),
                                             nn.Linear(edge_features*4, node_features),
                                             nn.Softplus(),
                                             nn.Linear(node_features, node_features))
            self.lattice_angle = nn.Sequential(RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
                                               nn.Linear(triplet_input_features, node_features),
                                               nn.Softplus(),
                                               nn.Linear(node_features, node_features))

            self.lattice_emb = nn.Sequential(nn.Linear(node_features * 6, node_features),
                                             nn.Softplus(),
                                             nn.Linear(node_features, node_features))

            self.lattice_atom_emb = nn.Sequential(nn.Linear(node_features * 2, node_features),
                                                  nn.Softplus(),
                                                  nn.Linear(node_features, node_features))

        self.att_layers = nn.ModuleList([MatformerConv(in_channels=node_features, out_channels=node_features,
                                                       heads=node_layer_head, edge_dim=node_features)
                                         for _ in range(conv_layers)])

        self.fc = nn.Sequential(nn.Linear(node_features, fc_features), nn.SiLU())


    def forward(self, data) -> torch.Tensor:
        # initial node features: atom feature network...
        lattice = data.lattice
        node_features = self.atom_embedding(data.x)
        edge_feat = data.edge_attr.squeeze()
        edge_features = self.rbf(edge_feat)

        if self.angle_lattice: ## module not used
            lattice_len = torch.norm(lattice, dim=-1) # batch * 3 * 1
            lattice_edge = self.lattice_rbf(lattice_len.view(-1)).view(-1, 3 * 128) # batch * 3 * 128
            cos1 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,0,:] * lattice[:,1,:], dim=-1) / (torch.norm(lattice[:,0,:], dim=-1) * torch.norm(lattice[:,1,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            cos2 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,0,:] * lattice[:,2,:], dim=-1) / (torch.norm(lattice[:,0,:], dim=-1) * torch.norm(lattice[:,2,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            cos3 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,1,:] * lattice[:,2,:], dim=-1) / (torch.norm(lattice[:,1,:], dim=-1) * torch.norm(lattice[:,2,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            lattice_emb = self.lattice_emb(torch.cat((lattice_edge, cos1, cos2, cos3), dim=-1))
            node_features = self.lattice_atom_emb(torch.cat((node_features, lattice_emb[data.batch]), dim=-1))
        
        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[4](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        if self.angle_lattice:
            features += lattice_emb
        
        features = self.fc(features)

        return features


class Matformer4Multi(nn.Module):
    """att pyg implementation."""

    def __init__(self, atom_input_features=92, node_features=128, edge_features=128, angle_lattice=False,
                 triplet_input_features=40, node_layer_head=4, conv_layers=5, fc_features=128, num_tasks=None,
                 embedding_layer=2):
        """Set up att modules."""
        super().__init__()
        self.atom_embedding = nn.Linear(atom_input_features, node_features)
        self.rbf = nn.Sequential(nn.Linear(edge_features, node_features),
                                 nn.Softplus(),
                                 nn.Linear(node_features, node_features))

        self.angle_lattice = angle_lattice
        if self.angle_lattice:  ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(RBFExpansion(vmin=0, vmax=50, bins=edge_features * 4),
                                             nn.Linear(edge_features * 4, node_features),
                                             nn.Softplus(),
                                             nn.Linear(node_features, node_features))
            self.lattice_angle = nn.Sequential(RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
                                               nn.Linear(triplet_input_features, node_features),
                                               nn.Softplus(),
                                               nn.Linear(node_features, node_features))

            self.lattice_emb = nn.Sequential(nn.Linear(node_features * 6, node_features),
                                             nn.Softplus(),
                                             nn.Linear(node_features, node_features))

            self.lattice_atom_emb = nn.Sequential(nn.Linear(node_features * 2, node_features),
                                                  nn.Softplus(),
                                                  nn.Linear(node_features, node_features))

        self.att_layers = nn.ModuleList([MatformerConv(in_channels=node_features, out_channels=node_features,
                                                       heads=node_layer_head, edge_dim=node_features)
                                         for _ in range(conv_layers)])

        self.task_embedding_atom = torch.nn.Parameter(torch.zeros(num_tasks, node_features))
        self.embedding_layer = embedding_layer

        self.fc = nn.Sequential(nn.Linear(node_features, fc_features), nn.SiLU())

    def forward(self, data, task_id) -> torch.Tensor:
        # initial node features: atom feature network...
        lattice = data.lattice
        node_features = self.atom_embedding(data.x)
        edge_features = self.rbf(data.edge_attr)

        if self.angle_lattice:  ## module not used
            lattice_len = torch.norm(lattice, dim=-1)  # batch * 3 * 1
            lattice_edge = self.lattice_rbf(lattice_len.view(-1)).view(-1, 3 * 128)  # batch * 3 * 128
            cos1 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 0, :] * lattice[:, 1, :], dim=-1) / (
                        torch.norm(lattice[:, 0, :], dim=-1) * torch.norm(lattice[:, 1, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            cos2 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 0, :] * lattice[:, 2, :], dim=-1) / (
                        torch.norm(lattice[:, 0, :], dim=-1) * torch.norm(lattice[:, 2, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            cos3 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 1, :] * lattice[:, 2, :], dim=-1) / (
                        torch.norm(lattice[:, 1, :], dim=-1) * torch.norm(lattice[:, 2, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            lattice_emb = self.lattice_emb(torch.cat((lattice_edge, cos1, cos2, cos3), dim=-1))
            node_features = self.lattice_atom_emb(torch.cat((node_features, lattice_emb[data.batch]), dim=-1))

        for i, att_layer in enumerate(self.att_layers):
            if i == self.embedding_layer:
                node_features = node_features + self.task_embedding_atom[task_id]
            node_features = att_layer(node_features, data.edge_index, edge_features)

        if len(self.att_layers) == self.embedding_layer:
            node_features = node_features + self.task_embedding_atom[task_id]

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        if self.angle_lattice:
            features += lattice_emb

        features = self.fc(features)

        return features


class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, atom_input_features=92, node_features=128, edge_features=128, angle_lattice=False,
                 triplet_input_features=40, node_layer_head=4, conv_layers=5, fc_features=128, output_features=1):
        """Set up att modules."""
        super().__init__()
        self.atom_embedding = nn.Linear(atom_input_features, node_features)
        self.rbf = nn.Sequential(RBFExpansion(vmin=0, vmax=8.0, bins=edge_features),
                                 nn.Linear(edge_features, node_features),
                                 nn.Softplus(),
                                 nn.Linear(node_features, node_features))

        self.angle_lattice = angle_lattice
        if self.angle_lattice:  ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(RBFExpansion(vmin=0, vmax=50, bins=edge_features * 4),
                                             nn.Linear(edge_features * 4, node_features),
                                             nn.Softplus(),
                                             nn.Linear(node_features, node_features))
            self.lattice_angle = nn.Sequential(RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
                                               nn.Linear(triplet_input_features, node_features),
                                               nn.Softplus(),
                                               nn.Linear(node_features, node_features))

            self.lattice_emb = nn.Sequential(nn.Linear(node_features * 6, node_features),
                                             nn.Softplus(),
                                             nn.Linear(node_features, node_features))

            self.lattice_atom_emb = nn.Sequential(nn.Linear(node_features * 2, node_features),
                                                  nn.Softplus(),
                                                  nn.Linear(node_features, node_features))

        self.att_layers = nn.ModuleList([MatformerConv(in_channels=node_features, out_channels=node_features,
                                                       heads=node_layer_head, edge_dim=node_features)
                                         for _ in range(conv_layers)])

        self.fc = nn.Sequential(nn.Linear(node_features, fc_features), nn.SiLU())

        self.read_out = nn.Linear(fc_features, output_features)

    def forward(self, data) -> torch.Tensor:
        # initial node features: atom feature network...
        lattice = data.lattice
        node_features = self.atom_embedding(data.x)
        edge_feat = data.edge_attr.squeeze()
        edge_features = self.rbf(edge_feat)

        if self.angle_lattice:  ## module not used
            lattice_len = torch.norm(lattice, dim=-1)  # batch * 3 * 1
            lattice_edge = self.lattice_rbf(lattice_len.view(-1)).view(-1, 3 * 128)  # batch * 3 * 128
            cos1 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 0, :] * lattice[:, 1, :], dim=-1) / (
                        torch.norm(lattice[:, 0, :], dim=-1) * torch.norm(lattice[:, 1, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            cos2 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 0, :] * lattice[:, 2, :], dim=-1) / (
                        torch.norm(lattice[:, 0, :], dim=-1) * torch.norm(lattice[:, 2, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            cos3 = self.lattice_angle(torch.clamp(torch.sum(lattice[:, 1, :] * lattice[:, 2, :], dim=-1) / (
                        torch.norm(lattice[:, 1, :], dim=-1) * torch.norm(lattice[:, 2, :], dim=-1)), -1, 1).unsqueeze(
                -1)).view(-1, 128)
            lattice_emb = self.lattice_emb(torch.cat((lattice_edge, cos1, cos2, cos3), dim=-1))
            node_features = self.lattice_atom_emb(torch.cat((node_features, lattice_emb[data.batch]), dim=-1))

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[4](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        if self.angle_lattice:
            features += lattice_emb

        features = self.fc(features)

        output = self.read_out(features)

        return output