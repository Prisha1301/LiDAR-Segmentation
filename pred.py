import torch
import numpy as np
from pcdet.models import build_network
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

# Load configuration
cfg_file = "OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml"
cfg_from_yaml_file(cfg_file, cfg)

# Load PointPillars model
model = build_network(model_cfg=cfg.MODEL, num_class=cfg.CLASS_NAMES, dataset=KittiDataset)
model.load_state_dict(torch.load("pointpillars_kitti_202012221652utc.pth"))
model.cuda()
model.eval()

# Load test LiDAR point cloud (.bin format)
lidar_bin = "/kitti/test/000001.bin"
points = np.fromfile(lidar_bin, dtype=np.float32).reshape(-1, 4)

# Convert to PyTorch tensor
points_tensor = torch.from_numpy(points).cuda().unsqueeze(0)

# Run inference
with torch.no_grad():
    pred_dicts, _ = model(points_tensor)

# Print results (bounding boxes)
for box in pred_dicts[0]["pred_boxes"]:
    print(f"Detected object at: {box[:3]}, Size: {box[3:6]}, Rotation: {box[6]}")
