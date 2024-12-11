from icecream import ic
import numpy as np
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from firm_core import firm_core1
from create_data import generate_random, create_by_file
import networkx as nx
import matplotlib.pyplot as plt

# file_name = generate_random(6, 3, 1.1)
file_name = 'random_39'
MG, num_layers = create_by_file(f'synthetic/{file_name}')
ic(MG.number_of_edges(), MG.number_of_nodes())

firm_core1(MG, num_layers, 2)