
import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

import jax
import jax.numpy as jnp
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh

from src.fem.hyperElasticity import HyperElasticity


def criticTile(msh_file:str,):
    pass
