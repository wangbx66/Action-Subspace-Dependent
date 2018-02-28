from os import path

import ipdb
debug = ipdb.set_trace

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))
