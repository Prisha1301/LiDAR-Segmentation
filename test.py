import open3d.ml.torch as ml3d  # or tf
from open3d.ml.utils import Config

# Load config (RandLANet or KPConv)
cfg = Config.load_from_file("ml3d/configs/randlanet_semantickitti.yml")

# Force CPU
cfg.pipeline["device"] = "cpu"  # Add this line if not present

# Load model/dataset
model = ml3d.models.RandLANet(**cfg.model)
dataset = ml3d.datasets.SemanticKITTI(dataset_path='E:/LiDAR-Diffusion/datasets/semantic_kitti/dataset/sequences')
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, **cfg.pipeline)

# Load pretrained weights (download first!)
pipeline.load_ckpt(ckpt_path='randlanet_semantickitti.pth')

# Run inference on CPU
data = dataset.get_split("test").get_data(0)
result = pipeline.run_inference(data)  # Will use CPU