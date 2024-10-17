from mesh import Mesh 
import numpy as np 
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import FluxGNN
from data_loader import V, C0, C1

data_dir = 'data'
vertices = np.load(f'{data_dir}/coordinates.npy')
triangles = np.load(f'{data_dir}/triangles.npy')

# Normalize
c_mean = C0.mean()
c_std = C0.std()
# Concentration at t_n
C0 = (C0 - c_mean) / c_std
# Concentration at t_{n+1}
C1 = (C1 - c_mean) / c_std
# Velocity field
V = (V - V.mean()) / V.std()

# Get mesh features 
mesh = Mesh(vertices, triangles)
centroids, dual_edges, edge_normals, edge_lens, edge_midpoints = mesh.get_dual_mesh()
# Vectors pointing from first to second cell center
dx = centroids[dual_edges[:,0]] - centroids[dual_edges[:,1]]
d = np.linalg.norm(dx, axis=1)
# Divide cell offsets by average length
dx = dx / d.mean()

e = torch.tensor(dx, dtype=torch.float32, device='cuda')
cell_areas = torch.tensor(mesh.cell_areas, dtype=torch.float32, device='cuda')

dataset = TensorDataset(V, C0.unsqueeze(2), C1.unsqueeze(2))
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = FluxGNN(
    mesh,
    vertex_in_size=2,
    edge_in_size=2,
    hidden_size=16,
    h_hidden_size=16,
    message_passing_num=3
)

model = model.cuda()
#state_dict = torch.load('dev_experiments/fluxgnn/checkpoints/model_100')
#model.load_state_dict(state_dict)
model.train()

device = 'cuda'
num_epochs = 1000
optimizer = optim.Adam(model.parameters(), lr=5e-4)
save_epoch = 20 

# Training loop
for epoch in range(num_epochs):
    print('epoch', epoch)
    running_loss = 0.0
    for batch_idx, (v, c0, c1) in enumerate(train_loader):
        v = v[0].to(device)
        c0 = c0[0].to(device)
        c1 = c1[0].to(device)
        
        optimizer.zero_grad()
        # Cell features are velocity velocities at cell centers
        # Edge features are vectors offsets between cell centers
        # Conserved quantity is concentration
        y = model(c0, v=v, e=e).flatten()
      
        # Area weighted loss
        loss = ((y - c1[:,0])**2 * cell_areas).sum()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    if epoch % save_epoch == 0:
        model_path = f'data/checkpoints/model_{epoch}.pt'
        torch.save(model.state_dict(), model_path)

