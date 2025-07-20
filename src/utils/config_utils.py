import yaml
import importlib.util

def load_config(path):
    """
    Loads a config file. Supports both .py and .yaml files.
    """
    if path.endswith('.py'):
        spec = importlib.util.spec_from_file_location("config", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.config
    elif path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config file format. Use .py or .yaml.")

# if __name__ == '__main__':
#     # Example usage with the default python config
#     py_config = load_config('configs/default_config.py')
#     print("--- Loaded Python Config ---")
#     print(py_config)
    
#     # Example usage with a dummy yaml file
#     dummy_yaml_content = """
#     data:
#       path: data/another_dataset
#       batch_size: 8
#     training:
#       learning_rate: 5e-5
#     """
#     with open('configs/dummy_config.yaml', 'w') as f:
#         f.write(dummy_yaml_content)
        
#     yaml_config = load_config('configs/dummy_config.yaml')
#     print("\n--- Loaded YAML Config ---")
#     print(yaml_config)
    
#     # Clean up dummy file
#     import os
#     os.remove('configs/dummy_config.yaml') 