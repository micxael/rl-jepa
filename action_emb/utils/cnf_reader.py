import yaml


def read_config(file: str) -> dict:
    """
    Helper function to read the configuration file (yml format).
    :param file: config file name
    :return: configuration dict
    """
    global cfg
    with open(file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg