import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tf_keras as keras
import numpy as np
import json

def load_compressed_model(config_path='model_config.json', weights_path='model_weights_f16.npz'):
    """Load model from compressed float16 weights + config."""
    
    # Rebuild model from config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = keras.Sequential.from_config(config)
    
    # Load compressed float16 weights
    data = np.load(weights_path)
    
    for layer in model.layers:
        layer_weights = []
        i = 0
        while f"{layer.name}_{i}" in data:
            w = data[f"{layer.name}_{i}"].astype(np.float32)  # cast back to float32
            layer_weights.append(w)
            i += 1
        if layer_weights:
            layer.set_weights(layer_weights)
    
    return model

# Usage in your Streamlit app:
# model = load_compressed_model('model_config.json', 'model_weights_f16.npz')
