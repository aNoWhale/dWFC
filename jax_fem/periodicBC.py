import jax
import jax.numpy as np
import numpy as onp
import scipy


def periodic_boundary_conditions(periodic_bc_info, mesh, vec):
    """
    Construct the 'P' matrix
    Reference: https://fenics2021.com/slides/dokken.pdf
    """
    p_node_inds_list_A = []
    p_node_inds_list_B = []
    p_vec_inds_list = []

    location_fns_A, location_fns_B, mappings, vecs = periodic_bc_info
    for i in range(len(location_fns_A)):
        node_inds_A = onp.argwhere(jax.vmap(location_fns_A[i])(mesh.points)).reshape(-1)
        node_inds_B = onp.argwhere(jax.vmap(location_fns_B[i])(mesh.points)).reshape(-1)
        points_set_A = mesh.points[node_inds_A]
        points_set_B = mesh.points[node_inds_B]

        EPS = 1e-5
        node_inds_B_ordered = []
        for node_ind in node_inds_A:
            point_A = mesh.points[node_ind]
            dist = onp.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
            node_ind_B_ordered = node_inds_B[onp.argwhere(dist < EPS)].reshape(-1)
            node_inds_B_ordered.append(node_ind_B_ordered)

        node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(-1)
        vec_inds = onp.ones_like(node_inds_A, dtype=onp.int32) * vecs[i]

        p_node_inds_list_A.append(node_inds_A)
        p_node_inds_list_B.append(node_inds_B_ordered)
        p_vec_inds_list.append(vec_inds)
        assert len(node_inds_A) == len(node_inds_B_ordered)

    # For mutiple variables (e.g, stokes flow, u-p coupling), offset will be nonzero.
    offset = 0
    inds_A_list = []
    inds_B_list = []
    for i in range(len(p_node_inds_list_A)):
        inds_A_list.append(onp.array(p_node_inds_list_A[i] * vec + p_vec_inds_list[i] + offset, dtype=onp.int32))
        inds_B_list.append(onp.array(p_node_inds_list_B[i] * vec + p_vec_inds_list[i] + offset, dtype=onp.int32))

    inds_A = onp.hstack(inds_A_list)
    inds_B = onp.hstack(inds_B_list)

    num_total_nodes = len(mesh.points)
    num_total_dofs = num_total_nodes * vec
    N = num_total_dofs
    M = num_total_dofs - len(inds_B)

    # The use of 'reduced_inds_map' seems to be a smart way to construct P_mat
    reduced_inds_map = onp.ones(num_total_dofs, dtype=onp.int32)
    reduced_inds_map[inds_B] = -(inds_A + 1)
    reduced_inds_map[reduced_inds_map == 1] = onp.arange(M)

    I = []
    J = []
    V = []
    for i in range(num_total_dofs):
        I.append(i)
        V.append(1.)
        if reduced_inds_map[i] < 0:
            J.append(reduced_inds_map[-reduced_inds_map[i] - 1])
        else:
            J.append(reduced_inds_map[i])
 
    P_mat = scipy.sparse.csr_array((onp.array(V), (onp.array(I), onp.array(J))), shape=(N, M))

    return P_mat

