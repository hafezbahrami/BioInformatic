import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim



import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.loader as loader
# ***********************************************
from torch_geometric import utils # to convert the edge index to actual adjacent matrix for some preliminary implementations (not a good practice)

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


import pytorch_lightning as pl
# Setting the seed
pl.seed_everything(42)


# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/test"

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./data"

num_workers = 6 # number of cpus to emply (for data loader function)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ***********************************************

# These are defined by myself for testing, and different that the internal classes defined in torch-geometric in:
# 1) We use the actual adjacent matrix, rather than the edge-index. from memory saving practice this is not good.
# But we do it for learning purposes
class GCNLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, node_feats, edge_index):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        adj_matrix = utils.to_dense_adj(edge_index)
        node_feats = node_feats.view(1, node_feats.shape[0], node_feats.shape[1]) # remove 1st dimension with is for old-way of representation of Batch
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats.squeeze() # remove the 1st dimension since it is for old-batching-style

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            in_channels - Dimensionality of input features
            out_channels - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert out_channels % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            out_channels = out_channels // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(in_channels, out_channels * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_channels))  # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, edge_index, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        # Things needs to be done to get the original adjacent matrix in full. This is not a good practice (from saving
        # memory perspective), but we will do it just to see how things work
        adj_matrix = utils.to_dense_adj(edge_index)
        node_feats = node_feats.view(1, node_feats.shape[0], node_feats.shape[1]) # remove 1st dimension with is for old-way of representation of Batch


        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False)  # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ],
            dim=-1)  # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats.squeeze() # remove the 1st dimension since it is for old-batching-style

# ***********************************************
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
    "myAttention": GATLayer,
    "myGCN": GCNLayer,
}

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            # ***********************************************
            if isinstance(l, geom_nn.MessagePassing) or isinstance(l, GATLayer) or isinstance(l, GCNLayer):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden,  # Not our prediction output yet!
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)  # Average pooling
        x = self.head(x)
        return x



class GraphLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-2,
                                weight_decay=0.0)  # High lr because of small dataset and small model
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)


def perform_steps():
    tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")

    print("Data object:", tu_dataset.data)
    print("Length:", len(tu_dataset))
    print(f"Average label: {tu_dataset.data.y.float().mean().item():4.2f}")

    torch.manual_seed(42)
    tu_dataset.shuffle()
    train_dataset = tu_dataset[:150]
    test_dataset = tu_dataset[150:]

    graph_train_loader = loader.DataLoader(train_dataset, batch_size=64, num_workers=num_workers, shuffle=True)
    graph_val_loader = loader.DataLoader(test_dataset, batch_size=64, num_workers=num_workers) # Additional loader if you want to change to a larger dataset
    graph_test_loader = loader.DataLoader(test_dataset, batch_size=64, num_workers=num_workers)

    batch = next(iter(graph_test_loader))
    print("Batch:", batch)
    print("Labels:", batch.y[:10])
    print("Batch indices:", batch.batch[:40])

    def train_graph_classifier(model_name, **model_kwargs):
        pl.seed_everything(42)

        # Create a PyTorch Lightning trainer with the generation callback
        root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
        os.makedirs(root_dir, exist_ok=True)

        trainer = pl.Trainer(default_root_dir=root_dir,
                             callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                             #                          gpus=1 if str(device).startswith("cuda") else 0,
                             devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                             max_epochs=500,
                             progress_bar_refresh_rate=0)

        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model, loading...")
            model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
        else:
            pl.seed_everything(42)
            model = GraphLevelGNN(c_in=tu_dataset.num_node_features,
                                  c_out=1 if tu_dataset.num_classes == 2 else tu_dataset.num_classes,
                                  **model_kwargs)
            trainer.fit(model, graph_train_loader, graph_val_loader)
            model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # Test best model on validation and test set
        train_result = trainer.test(model, graph_train_loader, verbose=False)
        test_result = trainer.test(model, graph_test_loader, verbose=False)
        result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
        return model, result


    model, result = train_graph_classifier(model_name="myAttention",
                                           c_hidden=256,
                                           layer_name="myAttention",
                                           num_layers=3,
                                           dp_rate_linear=0.5,
                                           dp_rate=0.0)

    # model, result = train_graph_classifier(model_name="GraphConv",
    #                                        c_hidden=256,
    #                                        layer_name="GraphConv",
    #                                        num_layers=3,
    #                                        dp_rate_linear=0.5,
    #                                        dp_rate=0.0)


if __name__ == "__main__":
    perform_steps()