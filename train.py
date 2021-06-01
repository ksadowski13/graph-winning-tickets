import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv


class GAT(nn.Module):
    def __init__(
        self,
        g: dgl.DGLGraph,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int,
        heads: list[int],
        activation: str,
        feats_dropout: float = 0,
        attention_dropout: float = 0,
        negative_slope: float = 0.2,
        residual: bool = False,
    ):
        super().__init__()
        self._g = g
        self._num_layers = num_layers

        if activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self._activation = nn.LeakyReLU()

        self._layers = self._create_layers(
            in_feats,
            hidden_feats,
            out_feats,
            heads,
            feats_dropout,
            attention_dropout,
            negative_slope,
            residual,
        )

    def _create_layers(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        heads: list[int],
        feats_dropout: float,
        attention_dropout: float,
        negative_slope: float,
        residual: bool,
    ) -> nn.ModuleList:
        layers = nn.ModuleList()

        layers.append(GATConv(
            in_feats,
            hidden_feats,
            heads[0],
            feats_dropout,
            attention_dropout,
            negative_slope,
            False,
            self._activation,
            allow_zero_in_degree=True,
        ))

        for i in range(1, self._num_layers):
            layers.append(GATConv(
                hidden_feats * heads[i - 1],
                hidden_feats,
                heads[i],
                feats_dropout,
                attention_dropout,
                negative_slope,
                residual,
                self._activation,
                allow_zero_in_degree=True,
            ))

        layers.append(GATConv(
            hidden_feats * heads[-2],
            out_feats,
            heads[-1],
            feats_dropout,
            attention_dropout,
            negative_slope,
            residual,
            None,
            allow_zero_in_degree=True,
        ))

        return layers

    def forward(self, inputs, get_attention: bool = False) -> list[torch.Tensor]:
        h = inputs

        for i in range(self._num_layers):
            h = self._layers[i](self._g, h).flatten(1)

        output_projection, attention = self._layers[-1](
            self._g, h, get_attention)
        logits = output_projection.mean(1)

        return logits, attention

class GCN(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        hidden_feats,
        out_feats,
        num_layers,
        activation,
        dropout,
    ):
        super().__init__()
        self._g = g

        if activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self._activation = nn.LeakyReLU()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, hidden_feats, activation=self._activation, allow_zero_in_degree=True))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_feats, hidden_feats, activation=self._activation, allow_zero_in_degree=True))
        # output layer
        self.layers.append(GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self._g, h)
        return h




def train(model, g, optimizer):
    pass


if __name__ == '__main__':
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    for round in range(300):
        model = GAT(
            g,
            in_feats=g.ndata['feat'].shape[-1],
            hidden_feats=8,
            out_feats=dataset.num_classes,
            num_layers=1,
            heads=[8, 1],
            activation='relu',
            feats_dropout=0,
            attention_dropout=0,
            negative_slope=0.2,
            residual=False,
        )

        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()

            logits, attention = model(g.ndata['feat'], get_attention=True)

            loss = F.cross_entropy(
                logits[train_mask], g.ndata['label'][train_mask])

            loss.backward()
            optimizer.step()

            _, indices = torch.max(logits[train_mask], dim=1)
            correct = torch.sum(indices == g.ndata['label'][train_mask])
            accuracy = correct.item() / len(g.ndata['label'][train_mask])

        attention = attention.view(-1)
        criterium, _ = torch.kthvalue(
            attention, int(attention.shape[0] * 0.01))
        eids = torch.nonzero(attention <= criterium).view(-1)

        num_edges = g.num_edges()

        g = dgl.remove_edges(g, eids)

        print(f'GAT # Round: {round + 1} Loss: {loss:.2f} Accuracy: {accuracy * 100:.2f} % Num edges: {num_edges} Sparsity: {0.99 ** round * 100:.2f} %')

        model = GCN(
            g,
            in_feats=g.ndata['feat'].shape[-1],
            hidden_feats=16,
            out_feats=dataset.num_classes,
            num_layers=1,
            activation='relu',
            dropout=0.5,
        )

        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()

            logits = model(g.ndata['feat'])

            loss = F.cross_entropy(
                logits[train_mask], g.ndata['label'][train_mask])

            loss.backward()
            optimizer.step()

            _, indices = torch.max(logits[train_mask], dim=1)
            correct = torch.sum(indices == g.ndata['label'][train_mask])
            accuracy = correct.item() / len(g.ndata['label'][train_mask])

        print(f'GCN # Round: {round + 1} Loss: {loss:.2f} Accuracy: {accuracy * 100:.2f} %')
