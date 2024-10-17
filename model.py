import torch
import torch.nn as nn
import numpy as np
import torch_scatter

# Linear encoder of conserved quantity
class Encoder(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()
        self.L = nn.Linear(in_size, out_size, bias=False)

    def forward(self, x):
        y = self.L(x)
        return y
    
# Decoder of conserved, which is just the psuedoinverse of the encoder
class Decoder(torch.nn.Module):
    
    def __init__(self, encoder):
        super(Decoder, self).__init__()
        self.encoder = encoder 
        
    def forward(self, y):
        L_inv = torch.linalg.pinv(self.encoder.L.weight)
        x = torch.linalg.matmul(L_inv, y.T).T
        return x 
    
class NodeBlock(torch.nn.Module):

    def __init__(self, edges, custom_func=None):

        super(NodeBlock, self).__init__()
        self.edges = edges
        self.net = custom_func

    def forward(self, e, v):
        
        # Aggregate cell messages
        v1 = torch.zeros((v.shape[0], e.shape[1]), device=v.device)
        torch_scatter.scatter_add(e, self.edges[:,0], dim=0, out=v1)
        torch_scatter.scatter_add(e, self.edges[:,1], dim=0, out=v1)

        # Concatenate with current node values
        v2 = torch.cat([
            v,
            v1
        ], dim=1)
        
        # Apply an MLP (or other defined function)
        v3 = self.net(v2)
        return v3

# Create messages that are passed between cells (basically, higher bandwidth messages)
class CellMessageBlock(torch.nn.Module):
    def __init__(self, edges, custom_func=None):
        
        super(CellMessageBlock, self).__init__()
        self.edges = edges
        self.net = custom_func

    def forward(self, e, v):

        e_features = torch.cat([
            v[self.edges[:,0]],
            v[self.edges[:,1]],
            e
        ], dim=1)
        
        e = self.net(e_features)  

        return e

class FluxMessageBlock(torch.nn.Module):
    def __init__(self, edges, custom_func=None):
        
        super(FluxMessageBlock, self).__init__()
        self.edges = edges
        self.net = custom_func

    def forward(self, h, m_flux, v):
 
        # For permutation invariance, add h values
        m_features = torch.cat([
            h[self.edges[:,0]] +  h[self.edges[:,1]],
            m_flux,
            v[self.edges[:,0]],
            v[self.edges[:,1]],
        ], dim=1)

        # Apply an MLP (or custom function)
        m_flux = self.net(m_features)  

        return m_flux

    
def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = torch.nn.Sequential(
        torch.nn.Linear(in_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, out_size)
    )
    if lay_norm: return torch.nn.Sequential(module,  torch.nn.LayerNorm(normalized_shape=out_size))
    return module
    

class FluxGNN(torch.nn.Module):
    
    def __init__(
        self, 
        mesh, 
        vertex_in_size = 1,
        edge_in_size = 1, 
        hidden_size = 16,
        h_hidden_size = 8,
        message_passing_num = 5,
        device='cuda'
    ):
        super(FluxGNN, self).__init__()
        
        # Mesh properties 
        self.mesh = mesh 
        dual_vertices, dual_edges, edge_normals, edge_lens, edge_midpoints = mesh.get_dual_mesh()
        self.num_vertices = len(dual_vertices)
        self.num_edges = len(dual_edges)
        self.edges = torch.tensor(dual_edges, dtype=torch.int64, device=device)
        self.cell_areas = torch.tensor(mesh.cell_areas, dtype=torch.float32, device=device)
        self.edge_lens = torch.tensor(edge_lens, dtype=torch.float32, device=device)
        self.edge_normals = torch.tensor(edge_normals, dtype=torch.float32, device=device)

        # Feature sizes
        self.vertex_in_size = vertex_in_size 
        self.edge_in_size = edge_in_size 
        self.hidden_size = hidden_size 
        self.h_hidden_size = h_hidden_size
        
        # Conserved quantity encoder
        self.h_encoder = Encoder(1, h_hidden_size)
        self.h_decoder = Decoder(self.h_encoder)

        # Vertex and edge features encoder / decoders
        if self.edge_in_size > 0:
            self.e_encoder = build_mlp(
                edge_in_size, 
                hidden_size,
                hidden_size
            )
            
        if self.vertex_in_size > 0:
            self.v_encoder = build_mlp(
                 vertex_in_size,
                 hidden_size,
                 hidden_size
            )

        # Define message passing methods
        self.processer_list_nodes = []
        self.processer_list_edges = []
        self.processor_list_flux = []
        for _ in range(message_passing_num):
            self.processer_list_nodes.append(
                NodeBlock(
                    self.edges,
                    custom_func = build_mlp(
                        2*hidden_size,
                        hidden_size,
                        hidden_size
                ))
            )
            self.processer_list_edges.append(
                CellMessageBlock(
                    self.edges,
                    custom_func=build_mlp(
                        3*hidden_size,
                        hidden_size, 
                        hidden_size
                    ))
            )
            
            self.processor_list_flux.append(
                FluxMessageBlock(
                    self.edges,
                    custom_func=build_mlp(
                        3*h_hidden_size + 2*hidden_size,
                        2*h_hidden_size,
                        2*h_hidden_size
                    ))
            )
            
        self.processer_list_nodes = torch.nn.ModuleList(self.processer_list_nodes) 
        self.processer_list_edges = torch.nn.ModuleList(self.processer_list_edges) 
        self.processor_list_flux = torch.nn.ModuleList(self.processor_list_flux) 


    def forward(self, h, **kwargs):
        
        # Initialize edge features e
        if 'e' in kwargs:
            e = kwargs['e']
            e = self.e_encoder(e)
        else: 
            e = torch.zeros((self.num_nodes, self.hidden_size), device=h.device)
        
        # Initialize cell features v
        if 'v' in kwargs:
            v = kwargs['v']
            v = self.v_encoder(v)
        else:
            v = torch.zeros((self.num_edges, self.hidden_size), device=h.device)
            
        # Initialize flux messages
        m_flux = torch.zeros((len(self.edges),  2*self.h_hidden_size), dtype=torch.float32, device=h.device)
        
        # The conserved quantity is volume, so multiply by cell areas
        h = h*self.cell_areas.unsqueeze(1)
        h = self.h_encoder(h)
        
        # Perform message passing
        for vb, eb, fb in zip(self.processer_list_nodes,self.processer_list_edges, self.processor_list_flux):
            
            # Update edge features
            e = e + eb(e, v)  
            # Update cell features
            v = v + vb(e, v)    
            # Compute flux messages 
            m_flux = m_flux + fb(h, m_flux, v)
            m = m_flux.reshape(m_flux.shape[0],-1,2)
            
            qx = m[:,:,0]*self.edge_normals[:,None,0]
            qy = m[:,:,1]*self.edge_normals[:,None,1]
            q_n = qx + qy 
            q_n = q_n * self.edge_lens[:,None]
            
            torch_scatter.scatter_add(q_n, self.edges[:,0], dim=0, out=h)
            torch_scatter.scatter_add(-q_n, self.edges[:,1], dim=0, out=h)
                        
        h = self.h_decoder(h)
        h = h / self.cell_areas.unsqueeze(1)
        
        return h 