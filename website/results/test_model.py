# test_model.py
import os
import sys
import json
import zipfile
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import tensorflow as tf
keras = tf.keras
import matplotlib.pyplot as plt

# ============================================================================
# MODEL TESTING SCRIPT - BRAIN TUMOR CLASSIFICATION
# ============================================================================

class ResNetPreprocess(keras.layers.Layer):
    def call(self, inputs):
        return keras.applications.resnet.preprocess_input(inputs)


def load_classifier_model(model_path):
    """Load a model, restoring the uniquely named preprocessing Lambdas if needed."""
    if zipfile.is_zipfile(model_path):
        with zipfile.ZipFile(model_path) as model_archive:
            if 'config.json' in model_archive.namelist():
                model_config = json.loads(model_archive.read('config.json'))
            else:
                model_config = None

        if model_config is not None:
            preprocessing_layers = {}
            known_preprocessors = {
                'resnet_preprocess': keras.applications.resnet.preprocess_input,
                'densenet_preprocess': keras.applications.densenet.preprocess_input,
            }
            for layer in model_config.get('config', {}).get('layers', []):
                if layer.get('class_name') != 'Lambda':
                    continue
                layer_config = layer.get('config', {})
                function_config = layer_config.get('function')
                if isinstance(function_config, dict):
                    layer_name = layer_config.get('name')
                    function_name = function_config.get('config')
                    if function_name == 'preprocess_input' and layer_name in known_preprocessors:
                        preprocessing_layers[layer_name] = known_preprocessors[layer_name]

            lambda_layers = [
                layer for layer in model_config.get('config', {}).get('layers', [])
                if layer.get('class_name') == 'Lambda'
            ]
            if (lambda_layers and len(preprocessing_layers) == len(lambda_layers)):
                model_config.pop('compile_config', None)
                model = keras.models.model_from_json(
                    json.dumps(model_config),
                    custom_objects={
                        'preprocess_input': next(iter(preprocessing_layers.values()))
                    }
                )
                for layer_name, preprocess_function in preprocessing_layers.items():
                    model.get_layer(layer_name).function = preprocess_function
                model.load_weights(model_path)
                return model

    return keras.models.load_model(
        model_path,
        custom_objects={
            'ResNetPreprocess': ResNetPreprocess,
            'Custom>ResNetPreprocess': ResNetPreprocess,
            'brisc>ResNetPreprocess': ResNetPreprocess,
        },
        compile=False,
    )


def load_segmentation_model(model_directory):
    """Load the Keras segmentation model from the result folder."""
    model_path = os.path.join(model_directory, 'segmentation.keras')
    if not os.path.isfile(model_path):
        print(f"[WARN] No segmentation.keras found in: {model_directory}")
        return None

    model = keras.models.load_model(
        model_path,
        custom_objects={'brisc>ResNetPreprocess': ResNetPreprocess},
        compile=False,
    )
    print(f"[OK] Segmentation model loaded: {model_path}")
    return model


class BrainTumorPredictor:
    def __init__(self, model_path, config_path, class_names_path):
        """Load model and configuration"""
        print("Loading model...")
        self.model = load_classifier_model(model_path)
        self.segmentation_model = load_segmentation_model(
            os.path.dirname(os.path.abspath(model_path))
        )

        self.config = {}
        if config_path and os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        with open(class_names_path, 'r') as f:
            self.class_data = json.load(f)
        
        self.class_names = self.class_data['class_names']
        self.image_size = self.config.get(
            'image_size',
            self.class_data.get('image_size', self.model.input_shape[1]),
        )
        ground_truth_config = self.config.get('ground_truth', {})
        self.unknown_ground_truth = ground_truth_config.get(
            'unknown_label', 'unknown'
        )
        self.ground_truth_folder_aliases = {
            self._normalize_folder_name(folder): class_name
            for folder, class_name in ground_truth_config.get(
                'folder_aliases', {}
            ).items()
        }
        (
            self.gradcam_model,
            self.gradcam_layer_names,
            self.gradcam_logits_layer,
        ) = self._build_gradcam_model()
        print(f"[OK] Model loaded. Classes: {self.class_names}\n")
    
    def preprocess_image(self, image_path):
        """Load and preprocess single image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32)
            return img_array, img
        except Exception as e:
            print(f"[ERROR] Error loading image: {e}")
            return None, None

    def predict_tumor_mask(self, img_array):
        """Return the segmentation model's foreground probability map."""
        if self.segmentation_model is None:
            return None

        batch = np.expand_dims(img_array, axis=0).astype(np.float32)
        segmentation_shape = self.segmentation_model.input_shape
        segmentation_size = segmentation_shape[1:3]
        if tuple(batch.shape[1:3]) != tuple(segmentation_size):
            batch = tf.image.resize(batch, segmentation_size).numpy()

        probabilities = self.segmentation_model(batch, training=False)
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 4 or probabilities.shape[-1] != 1:
            raise ValueError(
                'Unsupported segmentation output shape: '
                f'{probabilities.shape}'
            )
        mask = probabilities[0, :, :, 0]
        if mask.shape != (self.image_size, self.image_size):
            mask = tf.image.resize(
                mask[np.newaxis, ..., np.newaxis],
                (self.image_size, self.image_size),
            ).numpy()[0, :, :, 0]
        return mask

    @staticmethod
    def _normalize_folder_name(folder_name):
        return folder_name.strip().lower().replace('-', '_').replace(' ', '_')

    def get_ground_truth(self, image_path):
        """Infer the label from the image's parent folder when it is recognized."""
        folder_name = self._normalize_folder_name(
            os.path.basename(os.path.dirname(os.path.abspath(image_path)))
        )
        if folder_name in self.ground_truth_folder_aliases:
            return self.ground_truth_folder_aliases[folder_name]
        if folder_name in self.class_names:
            return folder_name
        return self.unknown_ground_truth

    def _build_gradcam_model(self):
        """Expose the classifier output and its connected spatial feature maps."""
        classification_outputs = [
            output for output in self.model.outputs if len(output.shape) == 2
        ]
        if len(classification_outputs) != 1:
            raise ValueError(
                'Grad-CAM requires exactly one rank-2 classification output.'
            )

        output_history = classification_outputs[0]._keras_history
        output_layer = getattr(output_history, 'operation', None)
        if output_layer is None:
            output_layer = output_history.layer
        output_activation = getattr(output_layer, 'activation', None)
        is_softmax_activation = (
            output_activation is not None
            and keras.activations.serialize(output_activation) == 'softmax'
        )
        logits_layer = None
        if isinstance(output_layer, keras.layers.Dense) and is_softmax_activation:
            class_score_output = output_layer.input
            logits_layer = output_layer
        elif (
            isinstance(output_layer, keras.layers.Softmax)
            or (
                isinstance(output_layer, keras.layers.Activation)
                and is_softmax_activation
            )
        ):
            class_score_output = output_layer.input
        else:
            class_score_output = classification_outputs[0]

        feature_layers = []
        feature_maps = []
        for pooling_layer in self.model.layers:
            if not isinstance(
                pooling_layer, keras.layers.GlobalAveragePooling2D
            ):
                continue
            history = pooling_layer.input._keras_history
            feature_layer = getattr(history, 'operation', None)
            if feature_layer is None:
                feature_layer = history.layer
            if (not isinstance(feature_layer, keras.Model)
                    or isinstance(feature_layer, keras.Sequential)):
                raise ValueError(
                    f'Could not identify the convolutional backbone for '
                    f'{pooling_layer.name}.'
                )
            feature_outputs = feature_layer.output
            if isinstance(feature_outputs, (list, tuple)):
                feature_output = feature_outputs[history.tensor_index]
            else:
                feature_output = feature_outputs
            if len(feature_output.shape) != 4:
                raise ValueError(
                    f'Grad-CAM requires a spatial feature map from {feature_layer.name}.'
                )
            feature_layers.append(feature_layer)
            feature_maps.append(pooling_layer.input)

        if not feature_maps:
            raise ValueError(
                'Grad-CAM requires a GlobalAveragePooling2D layer connected to '
                'a spatial feature map in the classifier.'
            )

        gradcam_model = keras.Model(
            inputs=self.model.inputs,
            outputs=feature_maps + [
                classification_outputs[0],
                class_score_output,
            ],
            name='gradcam_model',
        )
        return (
            gradcam_model,
            [layer.name for layer in feature_layers],
            logits_layer,
        )

    def _predict_with_gradcam(self, batch):
        """Return probabilities and a Grad-CAM averaged across classifier branches."""
        model_input = tf.convert_to_tensor(batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            outputs = self.gradcam_model(model_input, training=False)
            feature_maps = outputs[:-2]
            predictions = outputs[-2]
            class_scores = outputs[-1]
            if self.gradcam_logits_layer is not None:
                class_scores = tf.linalg.matmul(
                    class_scores, self.gradcam_logits_layer.kernel
                )
                if self.gradcam_logits_layer.use_bias:
                    class_scores = tf.nn.bias_add(
                        class_scores, self.gradcam_logits_layer.bias
                    )
            class_index = tf.argmax(predictions[0])
            class_score = class_scores[:, class_index]

        gradients = tape.gradient(class_score, feature_maps)
        heatmaps = []
        for layer_name, feature_map, gradient in zip(
                self.gradcam_layer_names, feature_maps, gradients):
            if gradient is None:
                raise ValueError(
                    f'Could not calculate Grad-CAM gradients for layer {layer_name}.'
                )

            channel_weights = tf.reduce_mean(gradient, axis=(1, 2), keepdims=True)
            heatmap = tf.nn.relu(
                tf.reduce_sum(channel_weights * feature_map, axis=-1)
            )
            heatmap = tf.image.resize(
                heatmap[..., tf.newaxis],
                (self.image_size, self.image_size),
                method='bilinear'
            )[..., 0]
            heatmap = tf.math.divide_no_nan(
                heatmap,
                tf.reduce_max(heatmap, axis=(1, 2), keepdims=True)
            )
            heatmaps.append(heatmap)

        combined_heatmap = tf.reduce_mean(tf.stack(heatmaps, axis=0), axis=0)[0]
        combined_heatmap = tf.math.divide_no_nan(
            combined_heatmap, tf.reduce_max(combined_heatmap)
        )
        return predictions[0].numpy(), combined_heatmap.numpy()
    
    def predict(self, image_path):
        """Make prediction on single image"""
        img_array, img_pil = self.preprocess_image(image_path)
        
        if img_array is None:
            return None
        
        # Batch dimension
        batch = np.expand_dims(img_array, axis=0)
        
        pred_probs, heatmap = self._predict_with_gradcam(batch)
        pred_class_idx = np.argmax(pred_probs)
        pred_class = self.class_names[pred_class_idx]
        confidence = pred_probs[pred_class_idx]
        tumor_mask = self.predict_tumor_mask(img_array)
        return {
            'image': img_pil,
            'gradcam': heatmap,
            'tumor_mask': tumor_mask,
            'class': pred_class,
            'ground_truth': self.get_ground_truth(image_path),
            'confidence': confidence,
            'all_probs': {self.class_names[i]: float(pred_probs[i]) 
                         for i in range(len(self.class_names))},
            'pred_class_idx': pred_class_idx
        }
    
    def display_result(self, result):
        """Display prediction result"""
        if result is None:
            return
        
        print("\n" + "="*70)
        print(f"PREDICTION: {result['class'].upper()}")
        print(f"GROUND TRUTH: {result['ground_truth'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("="*70)
        
        print("\nAll probabilities:")
        for cls, prob in result['all_probs'].items():
            bar = "#" * int(prob * 40)
            print(f"  {cls:12s}: {prob:.4f}  {bar}")
        
        print("="*70 + "\n")
        print(
            "For Grad-CAM and LIME together, run "
            "test2_model.py with the image path."
        )
        
        panels = [
            ('Image', None, None, 'none'),
            (
                'Grad-CAM (classifier focus, not tumor mask)',
                result['gradcam'],
                'jet',
                'heatmap',
            ),
        ]
        if result['tumor_mask'] is not None:
            panels.append((
                'Tumor area estimate (segmentation model)',
                result['tumor_mask'],
                'Reds',
                'thresholded',
            ))
        panel_count = len(panels)
        fig, axes = plt.subplots(
            1, panel_count, figsize=(6 * panel_count, 6)
        )
        axes = np.atleast_1d(axes)
        for ax, (title, overlay, cmap, display_mode) in zip(axes, panels):
            ax.imshow(result['image'])
            if overlay is not None:
                if display_mode == 'thresholded':
                    overlay = np.ma.masked_where(overlay < 0.5, overlay)
                    ax.imshow(
                        overlay, cmap=cmap, vmin=0.5, vmax=1, alpha=0.55
                    )
                elif display_mode == 'probability':
                    ax.imshow(
                        overlay, cmap=cmap, vmin=0, vmax=1, alpha=0.7
                    )
                else:
                    ax.imshow(overlay, cmap=cmap, vmin=0, vmax=1, alpha=0.45)
            ax.set_title(title, fontsize=12, fontweight='bold')
        for ax in axes:
            ax.axis('off')
        fig.suptitle(
            f"Prediction: {result['class']} ({result['confidence']:.2%}) | "
            f"Ground truth: {result['ground_truth']}",
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

# ============================================================================
# FILE PATH FINDER
# ============================================================================

def find_model_files():
    """Search for model files in common locations"""
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),  # Directory containing this script
        '.',                                    # Current directory
        './models',                             # models folder
        '../models',                            # parent/models
        os.path.expanduser('~/Downloads'),     # Downloads
        os.path.expanduser('~/Desktop'),       # Desktop
    ]
    
    print("Searching for model files...\n")
    
    for path in search_paths:
        if not os.path.isdir(path):
            continue

        model_name = next((name for name in (
            'classifier.keras', 'best_model.keras',
            'efficientnetb0_baseline.keras'
        ) if os.path.isfile(os.path.join(path, name))), None)
        config_name = 'preprocessing_config.json'
        class_name = 'class_names.json'
        if model_name and os.path.isfile(os.path.join(path, class_name)):
            model_file = os.path.abspath(os.path.join(path, model_name))
            config_path = os.path.join(path, config_name)
            config_file = (
                os.path.abspath(config_path)
                if os.path.isfile(config_path) else None
            )
            class_file = os.path.abspath(os.path.join(path, class_name))
            print(f"[OK] Found model: {model_file}")
            if config_file:
                print(f"[OK] Found config: {config_file}")
            print(f"[OK] Found classes: {class_file}\n")
            return model_file, config_file, class_file

    print("\n[WARN] Could not find the model and class names in one folder.\n")
    return None, None, None

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def main():
    model_path, config_path, class_path = find_model_files()
    if not model_path:
        print("[ERROR] Could not find a classifier model and class_names.json.")
        sys.exit(1)
    
    try:
        predictor = BrainTumorPredictor(model_path, config_path, class_path)
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        sys.exit(1)
    
    print("="*70)
    print("BRAIN TUMOR MRI CLASSIFICATION - TESTING")
    print("="*70)

    while True:
        input_path = input("\nEnter path for image or folder: ").strip().strip('"\'')
        if input_path.lower() == 'exit':
            break

        if os.path.isdir(input_path):
            image_paths = [
                os.path.join(input_path, name)
                for name in sorted(os.listdir(input_path))
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if not image_paths:
                print(f"[ERROR] No JPG, JPEG, or PNG images found in: {input_path}")
                continue

            print(f"\nFound {len(image_paths)} images. Testing...\n")
            for image_path in image_paths:
                result = predictor.predict(image_path)
                if result is not None:
                    print(
                        f"{os.path.basename(image_path):40s} -> "
                        f"predicted: {result['class']:12s} "
                        f"ground truth: {result['ground_truth']:12s} "
                        f"({result['confidence']:.2%})"
                    )
        elif os.path.isfile(input_path):
            if not input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"[ERROR] Unsupported image format: {input_path}")
                continue
            result = predictor.predict(input_path)
            predictor.display_result(result)
        else:
            print(f"[ERROR] Image or folder not found: {input_path}")

if __name__ == "__main__":
    main()