# TGNNs
Temporal Graph Neural Networks (TGNNs) are an advanced approach to modeling time series data, particularly in scenarios where there are interdependencies among different time series. In the context of multi-product time series forecasting, TGNNs can effectively capture both temporal dynamics and cross-series dependencies, making them particularly useful for applications like retail supply-chain forecasting.

### Key Concepts

1. **Temporal Graphs**: These are graphs where the nodes represent entities (e.g., products) and edges represent relationships (e.g., interactions between products) that can change over time.

2. **Cross-Series Dependencies**: In a retail context, sales of one product may influence the sales of another. For example, if two products are often bought together, an increase in the sales of one may lead to an increase in the sales of the other.

3. **Temporal Dynamics**: This refers to how the relationships and properties in the graph change over time. For example, seasonality in retail sales can be captured through temporal dynamics.

### Example in Python

Here’s a simplified example to illustrate how you might implement a TGNN for multi-product time series forecasting. This example uses PyTorch and PyTorch Geometric (a library for deep learning on graphs).

#### Step 1: Install Required Libraries

Make sure you have the necessary libraries installed:

```bash
pip install torch torch-geometric pandas numpy
```

#### Step 2: Define the Temporal Graph Neural Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class TGNN(nn.Module):
 def __init__(self, in_channels, out_channels):
 super(TGNN, self).__init__()
 self.conv1 = GCNConv(in_channels, 16)
 self.conv2 = GCNConv(16, out_channels)

 def forward(self, x, edge_index):
 x = F.relu(self.conv1(x, edge_index))
 x = self.conv2(x, edge_index)
 return x
```

#### Step 3: Prepare Your Data

Assuming you have time series data for multiple products, you can format it into a temporal graph structure.

```python
# Sample data (number of products, features, and relationships)
num_products = 5
features = torch.rand(num_products, 3) # Random features for each product
# Example edge index representing connections between products
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

# Create a PyTorch Geometric Data object
data = Data(x=features, edge_index=edge_index)
```

#### Step 4: Train the Model

```python
model = TGNN(in_channels=3, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Example training loop
for epoch in range(100):
 model.train()
 optimizer.zero_grad()
 out = model(data.x, data.edge_index)
 # Assuming we have some target values
 target = torch.rand(num_products, 1)
 loss = F.mse_loss(out, target)
 loss.backward()
 optimizer.step()
 print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

#### Step 5: Make Predictions

After training, you can use the model to make predictions on unseen data by passing the new temporal graph structure through the model.

```python
model.eval()
with torch.no_grad():
 predictions = model(data.x, data.edge_index)
 print("Predictions:", predictions)
```

### Conclusion

In this example, we defined a simple Temporal Graph Neural Network using PyTorch and PyTorch Geometric. We prepared some sample data to represent multiple products and their relationships, trained the model, and made predictions. In real-world applications, you would need to preprocess your actual retail supply-chain data to create suitable features and relationships for the graph.
