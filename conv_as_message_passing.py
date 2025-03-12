import torch
import torch_geometric


from torch_geometric.data import Batch

def relation_conv(H, W, Kh, Kw, ph, pw, sh, sw, i, j, ni, nj):
    """
    Computes the relation convolution between two pixels.

    Arguments:
    ----------
    Kh : int
        Height of the kernel.
    Kw : int
        Width of the kernel.
    ph : int
        Padding height.
    pw : int
        Padding width.
    sh : int
        Stride height.
    sw : int
        Stride width.
    i : int
        Row index of the first pixel.
    j : int
        Column index of the first pixel.
    ni : int
        Row index of the second pixel.
    nj : int
        Column index of the second pixel.

    Returns:
    --------
    bool
        Return True if the two verify Rconv defined in Question 2.
    """
    low_border = (i >= Kh // 2 - ph) and (j >= Kw // 2 - pw)
    up_border = (i < H + ph - Kh // 2) and (j < W + pw - Kw // 2)
    stride = ((i + Kh // 2 - ph) % sh == 0) and ((j + Kw // 2 - pw) % sw == 0)
    kernel_neighbor = (abs(i - ni) <= Kh // 2) and (abs(j - nj) <= Kw // 2) # Should always be True with the current definition of ni and nj

    return low_border and up_border and stride and kernel_neighbor

def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None, flow: str = "source_to_target"
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    COMPLETE

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    batch = True
    Kh, Kw = conv2d.kernel_size[0], conv2d.kernel_size[1]
    ph, pw = conv2d.padding[0], conv2d.padding[1]
    sh, sw = conv2d.stride[0], conv2d.stride[1]
    if conv2d is not None:
        assert Kh % 2 == 1 and Kw % 2 == 1, "Expected odd kernel sizes."
        assert ph <= Kh // 2 and pw <= Kw // 2, "Padding should be less than or equal to half of the kernel size, no need for more."
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        batch = False
    N, C, H, W = image.shape
    R_conv = lambda i, j, ni, nj: relation_conv(H, W, Kh, Kw, ph, pw, sh, sw, i, j, ni, nj)

    graphs = []

    for n in range(N):
        nodes = []
        edges = []
        edge_attrs = []

        for i in range(H):
            for j in range(W):
                node_index = i * W + j
                nodes.append(image[n, :, i, j])

                for di in range(-Kh // 2, Kh // 2 + 1):
                    for dj in range(-Kw // 2, Kw // 2 + 1):
                        ni, nj = i + di, j + dj
                        verify = ni >= 0 and ni < H and nj >= 0 and nj < W
                        if R_conv(i, j, ni, nj) :
                            if verify:
                                neighbor_index = ni * W + nj
                                edge_value = 1 + (ni - i + Kh // 2) * Kw + nj - j + Kw // 2 # 1 + (i'-i+\lfloor \frac{K_h}{2} \rfloor)K_w+j'-j+\lfloor \frac{K_w}{2} \rfloor
                                if flow == "source_to_target":
                                    edges.append([neighbor_index, node_index])
                                elif flow == "target_to_source":
                                    edges.append([node_index, neighbor_index])
                                else:
                                    raise ValueError("flow should be 'source_to_target' or 'target_to_source'")
                                edge_attrs.append(edge_value)
        image_size = torch.zeros(C)
        if C > 1:
            image_size[0] = H
            image_size[1] = W
        else:
            image_size[0] = H
        nodes.append(image_size)
        nodes = torch.stack(nodes)
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.long).unsqueeze(1)

        data = torch_geometric.data.Data(x=nodes, edge_index=edges, edge_attr=edge_attrs)
        graphs.append(data)
    if batch:
        print("Image to Graph - Batch image mode with {} images".format(N))
        batch_graphs = Batch.from_data_list(graphs)
    else:
        print("Image to Graph - Single image mode")
        batch_graphs = graphs[0]

    return batch_graphs


def graph_to_image(
    data, height: int | None = None, width: int | None = None, conv2d: torch.nn.Conv2d | None = None, verbose: bool = True
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    if conv2d is None:
        conv2d = torch.nn.Conv2d(C, C, kernel_size=3, padding=1, stride=1)
    Kh, Kw = conv2d.kernel_size[0], conv2d.kernel_size[1]
    ph, pw = conv2d.padding[0], conv2d.padding[1]
    sh, sw = conv2d.stride[0], conv2d.stride[1]
    if conv2d is not None:
        assert Kh % 2 == 1 and Kw % 2 == 1, "Expected odd kernel sizes."
        assert ph <= Kh // 2 and pw <= Kw // 2, "Padding should be less than or equal to half of the kernel size, no need for more."

    if isinstance(data, torch_geometric.data.Batch):
        N = data.num_graphs
        if verbose:
            print("Generating a batch of images with batch_size = ", N)
        C, H, W = data.x.shape[1], height, width
        list_of_graphs = data.to_data_list()
        if H is None or W is None:
            if C > 1:
                H, W = int(list_of_graphs[0].x[-1][0].item()), int(list_of_graphs[0].x[-1][1].item())
            else:
                H = int(list_of_graphs[0].x[-1].item())
                W = data.num_nodes // N // H
        images = torch.zeros((N, C, H, W))

        for n in range(N):
            graph = list_of_graphs[n]
            if H*W != graph.x.shape[0] - 1:
                raise ValueError("It seems that the batch of images does not have the same sizes.")
            for i in range(H):
                for j in range(W):
                    node_index = i * W + j
                    images[n, :, i, j] = graph.x[node_index]
    elif isinstance(data, torch_geometric.data.Data):
        if verbose:
            print("Generating a single image")
        C, H, W = data.x.shape[1], height, width
        if H is None or W is None:
            if C > 1:
                H, W = int(data.x[-1][0].item()), int(data.x[-1][1].item())
            else:
                H = int(data.x[-1].item())
                W = data.num_nodes // H
        images = torch.zeros((C, H, W))

        for i in range(H):
            for j in range(W):
                node_index = i * W + j
                images[:, i, j] = data.x[node_index]
    else:
        if verbose:
            print("Assuming Data.x has been provided -> No graph batch or single graph detected.")
        C, H, W = data.shape[1], height, width
        if H is None or W is None:
            if C > 1:
                H, W = int(data[-1][0].item()), int(data[-1][1].item())
            else:
                raise ValueError("Height and Width should be provided to tell wether a Tensor is a batch or a single image.")
        N = data.shape[0] // (H * W)
        if N != 1:
            if verbose:
                print("Assuming a batch of images has been provided with batch_size = ", N)
            images = torch.zeros((N, C, H, W))
            for n in range(N):
                for i in range(H):
                    for j in range(W):
                        node_index = i * W + j
                        images[n, :, i, j] = data[node_index + n * (H * W) + n]
        else:
            if verbose:
                print("Generating a single image")
            C, H, W = data.shape[1], height, width
            if H is None or W is None:
                H, W = int(data[-1][0].item()), int(data[-1][1].item())
            images = torch.zeros((C, H, W))

            for i in range(H):
                for j in range(W):
                    node_index = i * W + j
                    images[:, i, j] = data[node_index]

    return images

def compute_output_size(H, W, Kh, Kw, ph, pw, sh, sw):
    H_prime = (H + 2 * ph - Kh) // sh + 1
    W_prime = (W + 2 * pw - Kw) // sw + 1
    return H_prime, W_prime

def compute_output_index(i, j, Kh, Kw, ph, pw, sh, sw):
    i_prime = (i - Kh // 2 + ph) // sh
    j_prime = (j - Kw // 2 + pw) // sw
    return i_prime, j_prime

class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d, flow="source_to_target", implemented=False, direct=False):
        super().__init__(aggr='sum')  # "Add" aggregation.
        self.Kh, self.Kw = conv2d.kernel_size
        self.ph, self.pw = conv2d.padding
        self.sh, self.sw = conv2d.stride
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.weight = conv2d.weight # Shape: (out_channels, in_channels, Kh, Kw)
        self.bias = conv2d.bias # Shape: (out_channels)
        self.conv2d = conv2d
        self.H = None
        self.W = None
        self.H_prime = None
        self.W_prime = None
        self.edge_index = None
        self.num_graphs = None
        self.flow = flow
        self.implemented = implemented # If True, the propagation method is implemented, otherwise it is the basic propagate function.
        self.direct = direct # If True, the output is computed directly with formula given in Q2.
        self.compute_out_size = lambda H, W: compute_output_size(H, W, self.Kh, self.Kw, self.ph, self.pw, self.sh, self.sw)
        self.compute_out_index = lambda i, j: compute_output_index(i, j, self.Kh, self.Kw, self.ph, self.pw, self.sh, self.sw)

    def forward(self, data):
        if isinstance(data, torch_geometric.data.Batch):
            self.num_graphs = data.num_graphs
            list_of_graphs = data.to_data_list()
        else:
            self.num_graphs = 1
            list_of_graphs = [data]

        outputs = []
        for graph in list_of_graphs:
            x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
            self.edge_index = edge_index
            if self.in_channels > 1:
                H, W = int(x[-1][0].item()), int(x[-1][1].item())
            else:
                H = int(x[-1].item())
                W = graph.num_nodes // self.num_graphs // H
            self.H, self.W = H, W
            x = x[:-1]
            self.H_prime, self.W_prime = self.compute_out_size(H, W)
            if self.implemented and not self.direct:
                out = self.propagation(edge_index, x=x, edge_attr=edge_attr, flow=self.flow)
                out = self.transform_output(out)
            elif not self.implemented and not self.direct:
                out = self.propagate(edge_index, x=x, edge_attr=edge_attr, flow=self.flow)
                out = self.transform_output(out)
            else:
                out = self.compute_out_directly(x, edge_attr)
            outputs.append(out)
        if self.direct :
            if self.num_graphs == 1:
                return outputs[0]
            return torch.stack(outputs).view(self.num_graphs, outputs[0].shape[0], outputs[0].shape[1], outputs[0].shape[2]) # (num_graphs, out_channels, H', W')
        else :
            return torch.stack(outputs).view(self.num_graphs * outputs[0].shape[0], outputs[0].shape[1])

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the source node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size COMPLETE)
        """
        if self.bias is not None:
            if self.flow == "source_to_target":
                node_counts = torch.bincount(self.edge_index[1, :], minlength=self.edge_index.max().item() + 1)
                node_counts = node_counts[self.edge_index[1, :]]
            elif self.flow == "target_to_source":
                node_counts = torch.bincount(self.edge_index[0, :], minlength=self.edge_index.max().item() + 1)
                node_counts = node_counts[self.edge_index[0, :]]
        C_prime, nb_edges = self.out_channels, x_j.shape[0]
        phi = torch.zeros((nb_edges, C_prime))
        for idx in range(nb_edges):
            edge_value = int(edge_attr[idx]) - 1
            m, n = edge_value // self.Kw, edge_value % self.Kw
            for c_prime in range(C_prime):
                for c in range(self.in_channels):
                    phi[idx, c_prime] += x_j[idx, c] * self.weight[c_prime, c, m, n]
                if self.bias is not None:
                    phi[idx, c_prime] += self.bias[c_prime] * (1/node_counts[idx])
        return phi

    def transform_output(self, out):
        """
        Transforms the output of the message passing to the expected shape -> Indeed, we get (num_nodes, out_channels) as shape.
        But for convolution many of those nodes are useless to us. We want to get the output in the shape (H'*W', out_channels).
        """
        output = torch.zeros((self.H_prime * self.W_prime + 1, self.out_channels))
        if self.flow == "source_to_target":
            unique_nodes = torch.unique(self.edge_index[1, :])
        elif self.flow == "target_to_source":
            unique_nodes = torch.unique(self.edge_index[0, :])
        for idx in unique_nodes:
            i_prime, j_prime = self.compute_out_index(idx // self.W, idx % self.W)
            output[i_prime * self.W_prime + j_prime, :] = out[idx]
        if self.out_channels > 1:
            output[-1, :2] = torch.tensor([self.H_prime, self.W_prime])
        else:
            output[-1, :1] = torch.tensor([self.H_prime])
        return output
    
    def propagation(self, edge_index, x=None, edge_attr=None, flow="source_to_target"):
        """
        My own implemented propagate function : propagates messages along the edges of the graph.
        You can use it by changing the preset parameter 'implemented' to True.
        """
        if flow == "source_to_target":
            x_j = x[edge_index[0]]
            x_i = x[edge_index[1]]
        elif flow == "target_to_source":
            x_j = x[edge_index[1]]
            x_i = x[edge_index[0]]
        else:
            raise ValueError("Flow should be either 'source_to_target' or 'target_to_source'.")
        message = self.message(x_j, x_i, edge_attr)
        assert message.shape == (edge_index.shape[1], self.out_channels)
        out = torch.zeros((x.shape[0], self.out_channels))
        for idx in range(edge_index.shape[1]):
            i, j = int(edge_index[0, idx]), int(edge_index[1, idx])
            if flow == "source_to_target":
                out[j] += message[idx]
            elif flow == "target_to_source":
                out[i] += message[idx]
        return out
    
    def compute_out_directly(self, x: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the layer directly without message passing.
        """
        C_prime, H_prime, W_prime = self.out_channels, self.H_prime, self.W_prime
        Final = torch.zeros((C_prime, H_prime, W_prime))
        for idx in range(self.edge_index.shape[1]):
            if self.flow == "source_to_target":
                i, j = int(self.edge_index[1, idx]), int(self.edge_index[0, idx])
            elif self.flow == "target_to_source":
                i, j = int(self.edge_index[0, idx]), int(self.edge_index[1, idx])
            i_prime, j_prime = self.compute_out_index(i // self.W, i % self.W)
            edge_value = int(edge_attr[idx]) - 1
            m, n = edge_value // self.Kw, edge_value % self.Kw
            for c_prime in range(C_prime):
                for c in range(self.in_channels):
                    Final[c_prime, i_prime, j_prime] += x[j, c] * self.weight[c_prime, c, m, n]
                if self.bias is not None:
                    Final[c_prime, i_prime, j_prime] += self.bias[c_prime]

        return Final