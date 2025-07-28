# Import some useful modules.
import jax
import jax.numpy as np
import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))
# Import JAX-FEM specific modules.

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh


# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad)
    # to obtain the 'P_fn' function.
    def get_tensor_map(self):

        def psi(F):
            E = 10.0
            nu = 0.3
            mu = E / (2.0 * (1.0 + nu))
            kappa = E / (3.0 * (1.0 - 2.0 * nu))
            J = np.linalg.det(F)
            Jinv = J ** (-2.0 / 3.0)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.0) * (Jinv * I1 - 3.0) + (kappa / 2.0) * (J - 1.0) ** 2.0
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress


# Specify mesh-related information (first-order hexahedron element).
ele_type = "HEX8"
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), "data")
Lx, Ly, Lz = 1.0, 1.0, 1.0
meshio_mesh = box_mesh_gmsh(
    Nx=20, Ny=20, Nz=20, domain_x=Lx, domain_y=Ly, domain_z=Lz, data_dir=data_dir, ele_type=ele_type
)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0.0, atol=1e-5)


def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)


# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.0


def dirichlet_val_x2(point):
    return (
        0.5
        + (point[1] - 0.5) * np.cos(np.pi / 3.0)
        - (point[2] - 0.5) * np.sin(np.pi / 3.0)
        - point[1]
    ) / 2.0


def dirichlet_val_x3(point):
    return (
        0.5
        + (point[1] - 0.5) * np.sin(np.pi / 3.0)
        + (point[2] - 0.5) * np.cos(np.pi / 3.0)
        - point[2]
    ) / 2.0


dirichlet_bc_info = [
    [left] * 3 + [right] * 3,
    [0, 1, 2] * 2,
    [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] + [zero_dirichlet_val] * 3,
]


# Create an instance of the problem.
problem = HyperElasticity(
    mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info
)


# Solve the defined problem.
sol_list = solver(problem, solver_options={"petsc_solver": {}})


# Store the solution to local file.
vtk_path = os.path.join(data_dir, f"vtk/u.vtu")
save_sol(problem.fes[0], sol_list[0], vtk_path)
