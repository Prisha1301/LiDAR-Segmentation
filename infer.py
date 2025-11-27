import torch
from model import RandLANet
from data_utils import SemanticKITTIDataset

# Config
ckpt_path = 'randlanet_semantickitti.pth'
data_path = 'E:/LiDAR-Diffusion/datasets/semantic_kitti/dataset/sequences/'

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RandLANet(num_classes=20).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# Load and prepare data
points = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)[:, :3]
points = torch.from_numpy(points).float().unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    logits = model(points)
    preds = logits.max(dim=1)[1].cpu().numpy()

print("Predictions:", preds)