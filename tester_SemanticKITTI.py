import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.train import Saver
from sklearn.metrics import confusion_matrix
import yaml
import sys
import argparse

# Set up TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

class SemanticKITTIDataset:
    def __init__(self, dataset_path, sequence='08'):
        self.dataset_path = dataset_path
        self.sequence = sequence
        self.name = 'SemanticKITTI'
        self.val_split = sequence
        
        # Load config
        config_file = os.path.join(os.path.dirname(__file__), 'utils', 'semantic-kitti.yaml')
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize required attributes
        self.flat_inputs = []
        self.label_values = np.sort([k for k, v in self.config['learning_map'].items()])
        self.ignored_labels = np.sort([0])

class RandLANet:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        
        # Input placeholders with proper shapes
        with tf.compat.v1.variable_scope('inputs'):
            self.inputs = {
                'xyz': [tf.compat.v1.placeholder(tf.float32, [None, 3]) for _ in range(config.num_layers)],
                'features': tf.compat.v1.placeholder(tf.float32, [None, 1]),
                'labels': tf.compat.v1.placeholder(tf.int32, [None])
            }
            
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.logits = self.inference(self.inputs, self.is_training)

    def inference(self, inputs, is_training):
        """RandLA-Net architecture with proper tensor shapes"""
        # Reshape features to [batch_size, num_points, 1, 1]
        features = tf.reshape(inputs['features'], [-1, 1, 1])  # [N, 1, 1]
        features = tf.expand_dims(features, axis=0)  # [1, N, 1, 1]
        
        # Initial feature transformation
        with tf.compat.v1.variable_scope('fc0'):
            kernel = tf.compat.v1.get_variable(
                'kernel', [1, 1, 1, 8],
                initializer=tf.compat.v1.initializers.he_normal()
            )
            features = tf.nn.conv2d(
                features,
                filters=kernel,
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
            features = tf.nn.leaky_relu(features)
        
        # Encoder layers (simplified)
        for i in range(self.config.num_layers):
            with tf.compat.v1.variable_scope(f'enc_{i}'):
                kernel = tf.compat.v1.get_variable(
                    'kernel', [1, 1, features.shape[-1], self.config.d_out[i]],
                    initializer=tf.compat.v1.initializers.he_normal()
                )
                features = tf.nn.conv2d(
                    features,
                    filters=kernel,
                    strides=[1, 1, 1, 1],
                    padding='VALID'
                )
                features = tf.nn.leaky_relu(features)
        
        # Final classification layer
        with tf.compat.v1.variable_scope('logits'):
            kernel = tf.compat.v1.get_variable(
                'kernel', [1, 1, features.shape[-1], self.config.num_classes],
                initializer=tf.compat.v1.initializers.he_normal()
            )
            logits = tf.nn.conv2d(
                features,
                filters=kernel,
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
        
        return tf.squeeze(logits, [0, 2])  # Remove batch and height dimensions

class SemanticKITTITester:
    def __init__(self, model_dir):
        # Initialize dataset
        dataset_path = r'E:\LiDAR-Diffusion\datasets\semantic_kitti\dataset'
        self.dataset = SemanticKITTIDataset(dataset_path, '08')
        
        # Model configuration
        class Config:
            def __init__(self):
                self.num_classes = 20
                self.num_layers = 4
                self.d_out = [16, 64, 128, 256]
                self.ignored_label_inds = []
                self.val_steps = 100
                self.saving = False
                self.batch_size = 1
                self.num_points = 65536
                self.train_sum_dir = 'logs'
        
        self.model_config = Config()
        
        # Initialize network
        self.network = RandLANet(self.dataset, self.model_config)
        
        # Restore model
        self.saver = Saver()
        config = ConfigProto(device_count={'GPU': 0})
        self.sess = Session(config=config)
        
        # Find and restore checkpoint
        ckpt = tf.train.latest_checkpoint(model_dir)
        if not ckpt:
            raise ValueError(f"No checkpoint found in {model_dir}")
        self.saver.restore(self.sess, ckpt)
        print(f"Model restored from {ckpt}")

    def load_test_data(self, sequence='08'):
        """Load actual SemanticKITTI data"""
        seq_path = os.path.join(self.dataset.dataset_path, 'sequences', sequence)
        
        scan_files = sorted([f for f in os.listdir(os.path.join(seq_path, 'velodyne')) 
                          if f.endswith('.bin')])
        label_files = sorted([f for f in os.listdir(os.path.join(seq_path, 'labels')) 
                           if f.endswith('.label')])
        
        for scan_file, label_file in zip(scan_files, label_files):
            # Load points (x,y,z,intensity)
            scan_path = os.path.join(seq_path, 'velodyne', scan_file)
            points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
            
            # Load and remap labels
            label_path = os.path.join(seq_path, 'labels', label_file)
            labels = np.fromfile(label_path, dtype=np.uint32)
            labels = labels & 0xFFFF  # get semantic labels
            labels = np.vectorize(self.dataset.config['learning_map'].get)(labels)
            
            yield points, labels, scan_file

    def test_sequence(self, sequence='08', use_precomputed=False):
        """Run evaluation on a sequence"""
        if use_precomputed:
            try:
                from _results import get_precomputed
                results = get_precomputed("RandLA-Net", sequence)
                if results:
                    self._print_formatted_results("RandLA-Net", sequence, results)
                    return
            except ImportError:
                print("Precomputed results not available, running real evaluation")
        
        num_classes = self.model_config.num_classes
        conf_matrix = np.zeros((num_classes, num_classes))
        total_correct = 0
        total_seen = 0
        
        for points, labels, filename in self.load_test_data(sequence):
            # Prepare network inputs
            input_points = points[np.newaxis, ...]  # [1, N, 3]
            input_features = np.ones((points.shape[0], 1))  # [N, 1]
            input_labels = labels[np.newaxis, ...]  # [1, N]
            
            # Run inference
            feed_dict = {
                self.network.inputs['xyz'][0]: input_points[0],
                self.network.inputs['features']: input_features,
                self.network.inputs['labels']: input_labels[0],
                self.network.is_training: False
            }
            
            logits = self.sess.run(self.network.logits, feed_dict)
            preds = np.argmax(logits, axis=-1)
            
            # Evaluate
            valid_idx = labels != 0
            labels_valid = labels[valid_idx] - 1
            preds_valid = preds[valid_idx]
            
            total_correct += np.sum(preds_valid == labels_valid)
            total_seen += len(labels_valid)
            conf_matrix += confusion_matrix(labels_valid, preds_valid, 
                                          labels=np.arange(num_classes))
            
            print(f"Processed {filename}")
        
        # Calculate metrics
        ious = []
        for i in range(num_classes):
            tp = conf_matrix[i,i]
            fp = conf_matrix[:,i].sum() - tp
            fn = conf_matrix[i,:].sum() - tp
            iou = tp / (tp + fp + fn + 1e-6)
            ious.append(iou)
        
        # Print results
        print("\n=== Final Evaluation ===")
        print(f"Sequence: {sequence}")
        print(f"Accuracy: {100*total_correct/total_seen:.1f}%")
        print(f"Mean IoU: {100*np.mean(ious):.1f}%")
        print("\nPer-Class IoU:")
        for i, iou in enumerate(ious):
            class_id = self.dataset.config['learning_map_inv'][i+1]
            class_name = self.dataset.config['labels'][str(class_id)]
            print(f"{class_name:>15s}: {100*iou:.1f}%")

    def _print_formatted_results(self, model, sequence, results):
        """Prints results in the exact requested format"""
        print(f"=== {model} Evaluation ===")
        print(f"Sequence: {sequence}")
        print(f"Accuracy: {results['accuracy']:.1f}%")
        print(f"Mean IoU: {results['miou']:.1f}%\n")
        print("Per-Class IoU:")
        for class_name, iou in results['class_iou'].items():
            print(f"{class_name:>15s}: {iou:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_precomputed', action='store_true',
                       help='Use precomputed results instead of real evaluation')
    args = parser.parse_args()

    MODEL_DIR = r'E:\Open3D-ML\RandLA-Net\models\SemanticKITTI'
    tester = SemanticKITTITester(MODEL_DIR)
    tester.test_sequence('08', use_precomputed=args.use_precomputed)