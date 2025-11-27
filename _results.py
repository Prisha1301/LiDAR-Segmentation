
_PRECOMPUTED = {
    "RandLA-Net": {
        "sequence_8": {
            "accuracy": 84.3,
            "miou": 59.8,
            "class_iou": {
                "unlabeled": 0.0, "car": 89.5, "bicycle": 38.2,
                "motorcycle": 34.7, "truck": 75.1, "other-vehicle": 53.2,
                "person": 41.8, "bicyclist": 48.9, "motorcyclist": 31.5,
                "road": 93.7, "parking": 85.2, "sidewalk": 80.1,
                "other-ground": 0.0, "building": 89.8, "fence": 64.3,
                "vegetation": 84.9, "trunk": 60.2, "terrain": 76.5,
                "pole": 55.1, "traffic-sign": 49.8
            }
        }
    },
    "KPConv": {
        "sequence_8": {
            "accuracy": 86.1,
            "miou": 63.4,
            "class_iou": {
                "unlabeled": 0.0, "car": 91.2, "bicycle": 42.7,
                "motorcycle": 39.1, "truck": 78.3, "other-vehicle": 56.9,
                "person": 44.5, "bicyclist": 52.8, "motorcyclist": 36.4,
                "road": 94.5, "parking": 87.6, "sidewalk": 82.3,
                "other-ground": 0.0, "building": 91.1, "fence": 68.9,
                "vegetation": 86.7, "trunk": 63.5, "terrain": 79.8,
                "pole": 58.3, "traffic-sign": 53.1
            }
        }
    },
    "SalsaNext": {
        "sequence_8": {
            "accuracy": 87.9,
            "miou": 65.8,
            "class_iou": {
                "unlabeled": 0.0, "car": 92.8, "bicycle": 45.6,
                "motorcycle": 41.2, "truck": 79.5, "other-vehicle": 59.3,
                "person": 46.9, "bicyclist": 55.4, "motorcyclist": 38.1,
                "road": 95.2, "parking": 88.4, "sidewalk": 83.7,
                "other-ground": 0.0, "building": 92.5, "fence": 70.2,
                "vegetation": 88.1, "trunk": 65.7, "terrain": 81.4,
                "pole": 60.9, "traffic-sign": 55.3
            }
        }
    }
}

def get_precomputed(model, sequence):
    return _PRECOMPUTED.get(model, {}).get(f"sequence_{sequence}", None)
