"""
The python file for taking the configuration and use it 
"""
import yaml

# Specify the path to your YAML file
yaml_file_path = "config.yaml"

# Open and read the YAML file
with open(yaml_file_path, "r") as yaml_file:
    try:
        yaml_data = yaml.safe_load(yaml_file)
    except yaml.YAMLError as e:
        print("Error reading YAML:", e)


print(yaml_data["models"][0])
print(yaml_data["parameters"]["learning_rate"])