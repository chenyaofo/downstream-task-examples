import os
import yaml

def get_config(path="config.yaml"):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

config = get_config(os.environ.get("CONFIG", "assets/config.yaml"))