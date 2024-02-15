
import yaml

def load_yaml(filename: str) -> dict:
    with open(filename, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

def save_yaml(filename: str, data: dict) -> None:
    with open(filename, 'w') as f:
        yaml.dump(data, f)

