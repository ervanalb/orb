#!/usr/bin/env python3

import geodesic as geo
import numpy as np
import itertools
import collections
import matplotlib.pyplot as plt
import random
random.seed(0)
import uuid
import re

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

KICAD_FLIP = np.array([1, -1]) # Kicad Y is positive down

# Overall radius of sphere
radius = 50

# Flat pattern: Offset each polygon inward slightly to add clearance for board edges
offset_dist = -0.05 # mm
circle_r = 0.65 # mm

(vs, es, fs) = geo.sphere(u=3, v=0)

#vms = geo.vertex_matrices(vs)
#ems = geo.edge_matrices(es, vs)
#fms = geo.face_matrices(fs, vs)
#ves = geo.vertex_edges(es)
#fes = geo.face_edges(fs, es)
#els = geo.edge_lengths(es, vs)

def dual(vs, es, fs):
    def faces_from_edge(e):
        assert len(result) == 2

    # Get a list of each dual vertex (parallel to faces)
    def face_centroid(f):
        return np.mean(vs[np.array(f)], axis=0)

    dvs = np.array([face_centroid(f) for f in fs])

    # Get a list of each dual face (parallel to vertices)
    def face_indices_from_vertex_index(vi):
        result = [
            fi for (fi, f) in enumerate(fs)
            if vi in f
        ]
        return result

    dfs = [face_indices_from_vertex_index(vi) for vi in range(len(vs))]

    # Get a list of each dual edge (parallel to edges)
    def dual_edge_from_edge(e):
        # Find the two face indices that this edge is the intersection of
        e = frozenset(e)
        result = [fi for (fi, f) in enumerate(fs) if 
            frozenset((f[0], f[1])) == e
            or frozenset((f[0], f[2])) == e
            or frozenset((f[1], f[2])) == e
        ]
        assert len(result) == 2
        return result

    des = [dual_edge_from_edge(e) for e in es]

    return (dvs, des, dfs)

(dvs, des, dfs) = dual(vs, es, fs)


def nudge_vertices(vs, es):
    vs = np.array(vs)

    def adjacent_vis(vi):
        vis = []
        for (vi1, vi2) in es:
            if vi == vi1:
                vis.append(vi2)
            elif vi == vi2:
                vis.append(vi1)
        return set(vis)

    def nudge_vertex(vi):
        adjacent_1 = adjacent_vis(vi)
        #adjacent_2 = {vi2 for vi1 in adjacent_1 for vi2 in adjacent_vis(vi1)}
        #adjacent_2 = adjacent_2 - adjacent_1 - {vi}
        #new_v = np.mean(vs[np.array(adjacent_1)], axis=0)
        new_v = np.mean(vs[list(adjacent_1)], axis=0)
        nudge = new_v - vs[vi]
        return vs[vi] - nudge ## Apply backwards--weird

    return [nudge_vertex(vi) for vi in range(len(vs))]

def face_planes(vs, fs):
    vs = np.array(vs)

    def face_plane(f):
        f = np.array(f)
        expected_dir = np.mean(vs[f], axis=0)
        # Find the average normal of all vertex triples
        vi_triples = np.array(list(itertools.combinations(f, 3)))

        def normal(v1, v2, v3):
            n = np.cross(v2 - v1, v3 - v1)
            if np.dot(n, expected_dir) < 0:
                n = -n
            return n / np.linalg.norm(n)

        normals = [normal(*vs[vi_triple]) for vi_triple in vi_triples]
        avg_normal = np.mean(np.array(normals), axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        avg_dist = np.mean(np.dot(vs[f], avg_normal), axis=0)

        return avg_normal, avg_dist

    face_normals, face_ds = zip(*[face_plane(f) for f in fs])

    return face_normals, face_ds

def flatten_faces(vs, fs, face_normals, face_ds):
    vs = np.array(vs)

    # Recompute each vertex as the intersection of its face planes
    def intersection_of_faces(vi):
        fis = [
            fi for (fi, f) in enumerate(fs)
            if vi in f
        ]

        assert len(fis) == 3
        a = np.array([face_normals[f] for f in fis])
        b = np.array([face_ds[f] for f in fis])
        pt = np.linalg.solve(a, b)
        return pt

    return [intersection_of_faces(vi) for vi in range(len(vs))]

def flatten_pts_onto_faces(face_normals, face_ds, pts):
    def flatten_pt(pt):
        chosen_fi = None
        for fi, (n, d) in enumerate(zip(face_normals, face_ds)):
            proj_d = np.dot(n, pt)
            if proj_d > d - 1e-5:
                # Project onto plane
                pt *= d / proj_d
                chosen_fi = fi
        assert chosen_fi is not None
        return pt, chosen_fi

    (pts, fis) = zip(*[flatten_pt(pt) for pt in pts])
    return pts, fis

dvs = nudge_vertices(dvs, des)
dvs = geo.project_points_to_sphere(dvs)

face_normals, face_ds = face_planes(dvs, dfs)
dvs = flatten_faces(dvs, dfs, face_normals, face_ds)

## Draw the result
## Only select a few faces
#
#allow_face_indices = []
#
#allow_vertices = [vi for vi in range(len(dvs)) if any(vi in dfs[fi] for fi in allow_face_indices)]
#
#for (dvi1, dvi2) in des:
#    #if dvi1 in allow_vertices and dvi2 in allow_vertices:
#    if dvs[dvi1][2] > 0.5:
#        ax.plot(
#            [dvs[dvi1][0], dvs[dvi2][0]],
#            [dvs[dvi1][1], dvs[dvi2][1]],
#            [dvs[dvi1][2], dvs[dvi2][2]],
#        )

(vs, es, fs) = geo.sphere(u=4, v=3, base=vs)

vs = np.array(vs) * radius
dvs = np.array(dvs) * radius

print(len(vs))

face_normals, face_ds = face_planes(dvs, dfs)
vs, v_fis = flatten_pts_onto_faces(face_normals, face_ds, vs)
vs = np.array(vs)
v_fis = np.array(v_fis)

#cond = vs[:, 2] > 0.5
#vs = vs[cond]
#v_fis = v_fis[cond]
#ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2], c=v_fis)

# Route data from face to face

def route_data_coarse(es, fs, first_face, last_face, heuristic):
    face_connections = [set() for _ in range(len(fs))]
    for (ei, (vi1, vi2)) in enumerate(es):
        # Find faces that contain this edge
        faces = set(fi for fi in range(len(fs))
            if vi1 in fs[fi] and vi2 in fs[fi])

        # Connect them to each other
        for fi in faces:
            face_connections[fi] |= {(fi, ei) for fi in faces - {fi}}

    # Search over faces with a heuristic
    # to generate a path that visits all of them

    to_visit = [([], [], first_face)]

    heuristic_key = lambda s: heuristic(*s)

    while to_visit:
        (f_history, e_history, f) = to_visit.pop()
        # Check termination condition
        if f == last_face and len(f_history) + 1 == len(fs):
            f_history = f_history + [f]

            # The search is done--
            # but first, add in a fake edge at the beginning for overall data input
            # and a fake edge at the end for overall data output.
            # Do this by artifically extending the search one step before the first and one step beyond the last
            # and taking whichever edge produces the best heuristic value.

            possible_pre_first_steps = [([fi], [], f_history[0]) for (fi, ei) in face_connections[f_history[0]] if ei != e_history[0]]
            ((pre_first_face,), _, _) = max(possible_pre_first_steps, key=heuristic_key)
            (first_half_edge,) = (ei for (fi, ei) in face_connections[f_history[0]] if fi == pre_first_face)

            possible_post_last_steps = [([f_history[-1]], [], fi) for (fi, ei) in face_connections[f_history[-1]] if ei != e_history[-1] and ei != first_half_edge]
            (_, _, post_last_face) = max(possible_post_last_steps, key=heuristic_key)
            (last_half_edge,) = (ei for (fi, ei) in face_connections[f_history[-1]] if fi == post_last_face)

            e_history = [first_half_edge] + e_history + [last_half_edge]

            return f_history, e_history

        # Enumerate possible next steps
        possible_next_steps = [(f_history + [f], e_history + [ei], nf) for nf, ei in face_connections[f] if nf not in f_history]
        # Sort possible next steps according to heuristic
        possible_next_steps = sorted(possible_next_steps, key=heuristic_key)
        to_visit.extend(possible_next_steps)

    assert False, "Search did not find any paths"

topness = lambda fi: face_normals[fi][2]

def circulation(fi1, fi2):
    c = np.cross(face_normals[fi1], face_normals[fi2])
    c = c / np.linalg.norm(c)
    return c[2]

first_face = max(range(len(dfs)), key=topness)
last_face = min(range(len(dfs)), key=topness)

def heuristic(fh, eh, fi):
    t = topness(fi)
    c = circulation(fh[-1], fi)
    return t + 0.5 * c

face_order, edge_order = route_data_coarse(des, dfs, first_face, last_face, heuristic=heuristic)

print(face_order)

def route_data_fine(vs, es, v_coarse_fis, coarse_vs, coarse_es, coarse_fs, coarse_face_order, coarse_edge_order):
    vs = np.array(vs)
    coarse_vs = np.array(coarse_vs)

    def route_data_fine(include_vertices, first_vertex, last_vertex, heuristic):
        vertex_connections = [set() for _ in range(len(vs))]
        for (ei, (vi1, vi2)) in enumerate(es):
            if vi1 in include_vertices and vi2 in include_vertices:
                vertex_connections[vi1].add((vi2, ei))
                vertex_connections[vi2].add((vi1, ei))

        # Search over vertices with a heuristic
        # to generate a path that visits all of them

        to_visit = [([], [], first_vertex)]

        heuristic_key = lambda s: heuristic(*s)

        while to_visit:
            (v_history, e_history, v) = to_visit.pop()
            # Check termination condition
            if v == last_vertex and len(v_history) + 1 == len(include_vertices):
                return v_history + [v], e_history

            # Enumerate possible next steps
            possible_next_steps = [(v_history + [v], e_history + [ei], nv) for nv, ei in vertex_connections[v] if nv not in v_history]
            # Sort possible next steps according to heuristic
            possible_next_steps = sorted(possible_next_steps, key=heuristic_key)
            to_visit.extend(possible_next_steps)

        assert False, "Search did not find any paths"

    def route_one_coarse_face(coarse_fi):
        include_vertices = [vi for vi, v_coarse_fi in enumerate(v_coarse_fis) if v_coarse_fi == coarse_fi]

        first_vertex = None
        last_vertex = None

        # Trivial case: there is only one LED which is both the first and last vertex
        if len(face_order) == 1:
            first_vertex = 0
            last_vertex = 0

        # Pick first and last based on proximity to coarse face input/output
        position_in_order = face_order.index(coarse_fi)
        if first_vertex is None:
            input_ei = edge_order[position_in_order]
            input_edge_pt = np.mean(coarse_vs[coarse_es[input_ei]], axis=0)

            first_vertex = min(include_vertices, key=lambda vi: np.linalg.norm(input_edge_pt - vs[vi]))

        if last_vertex is None:
            output_ei = edge_order[position_in_order + 1]
            output_edge_pt = np.mean(coarse_vs[coarse_es[output_ei]], axis=0)

            last_vertex = min(set(include_vertices) - {first_vertex}, key=lambda vi: np.linalg.norm(output_edge_pt - vs[vi]))

        ## If either input or output was missing, set the other one as far away as possible
        #if first_vertex is None and last_vertex is not None:
        #    first_vertex = max(set(include_vertices) - {last_vertex}, key=lambda vi: np.linalg.norm(vs[last_vertex] - vs[vi]))
        #if last_vertex is None and first_vertex is not None:
        #    last_vertex = max(set(include_vertices) - {first_vertex}, key=lambda vi: np.linalg.norm(vs[first_vertex] - vs[vi]))

        assert first_vertex is not None
        assert last_vertex is not None

        last_vertex_pos = np.array(vs[last_vertex])

        def heuristic(v_history, e_history, v):
            # Farther from the last vertex is better
            return np.linalg.norm(np.array(vs[v]) - last_vertex_pos)

        return route_data_fine(include_vertices, first_vertex, last_vertex, heuristic)

    return [route_one_coarse_face(coarse_fi) for coarse_fi in range(len(coarse_fs))]

face_fine_routing = route_data_fine(vs, es, v_fis, dvs, des, dfs, face_order, edge_order)

# Come up with a transformation matrix for each LED
def led_placement(vs, face_fine_routing, v_fis, coarse_vs, coarse_es, coarse_fs, coarse_face_normals, coarse_face_order, coarse_edge_order, footprint_data_direction_vector):
    coarse_vs = np.array(coarse_vs)
    def place_one_led(vi):
        # Find center of face that this LED lies on
        coarse_fi = v_fis[vi]
        center = np.mean(coarse_vs[coarse_fs[coarse_fi]], axis=0)

        m = np.eye(4)

        # Translation = LED position
        t = vs[vi]
        # Z = normal to face
        z = coarse_face_normals[coarse_fi]
        # First guess at Y = away from center
        y_options = [t - center, np.array([1, 0, 0]), np.array([0, 1, 0])]
        for y in y_options:
            if np.linalg.norm(np.cross(y, z)) > 1e-5:
                break
        else:
            assert False, "Found no good Y basis direction"
        # X = Y cross Z
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        # Fixed Y = Z cross X
        y = np.cross(z, x)
        y /= np.linalg.norm(y)

        m[0:3, 0] = x
        m[0:3, 1] = y
        m[0:3, 2] = z
        m[0:3, 3] = t

        # Flip increments of 90 degrees if it fits routing better
        (fine_routing, _) = face_fine_routing[coarse_fi]
        i = fine_routing.index(vi)
        prev_pt = vs[fine_routing[i - 1]] if i > 0 else None
        next_pt = vs[fine_routing[i + 1]] if i < len(fine_routing) - 1 else None

        if prev_pt is None:
            # Route from center of input edge
            coarse_ei = coarse_edge_order[coarse_face_order.index(coarse_fi)]
            prev_pt = np.mean(coarse_vs[coarse_es[coarse_ei]], axis=0)

        if next_pt is None:
            # Route to center of output edge
            coarse_ei = coarse_edge_order[coarse_face_order.index(coarse_fi) + 1]
            next_pt = np.mean(coarse_vs[coarse_es[coarse_ei]], axis=0)

        routing_vector = next_pt - prev_pt

        rot_90 = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        possible_rotations = [
            np.eye(4),
            rot_90,
            rot_90 @ rot_90,
            rot_90 @ rot_90 @ rot_90,
        ]

        possible_orientations = [m @ r for r in possible_rotations]
        def aligned_with_routing(m):
            vec = (m @ np.array([footprint_data_direction_vector[0], footprint_data_direction_vector[1], 0, 0]))[0:3]
            # Return how aligned the data direction vector of the footprint is with the data line routing
            return np.dot(vec, routing_vector)

        final_orientation = max(possible_orientations, key=aligned_with_routing)

        return final_orientation

    return [place_one_led(vi) for vi in range(len(vs))]

led_matrices = led_placement(vs, face_fine_routing, v_fis, dvs, des, dfs, face_normals, face_order, edge_order, footprint_data_direction_vector=[1, 1])

# Draw the result

#for (dvi1, dvi2) in des:
#    ax.plot(
#        [dvs[dvi1][0], dvs[dvi2][0]],
#        [dvs[dvi1][1], dvs[dvi2][1]],
#        [dvs[dvi1][2], dvs[dvi2][2]],
#    )
#
#path = np.array([face_ds[fi] * face_normals[fi] for fi in face_order])
#ax.plot(path[:, 0], path[:, 1], path[:, 2])
#
#for (vertex_order, _) in face_fine_routing:
#    path = np.array([vs[vi] for vi in vertex_order])
#    ax.plot(path[:, 0], path[:, 1], path[:, 2])
#
#plt.show()
#assert False

# Unfold the orb

def unfold_matrix(fn1, fd1, fn2, fd2):
    # To get the matrix that "un-does" the fold between f1 and f2,
    # we will first generate a basis
    # whose origin lies on the edge between f1 and f2,
    # and whose Z axis is the normal of f1,
    # and whose Y axis is along the edge connecting f1 and f2.

    # First we find the intersection line in Plucker coordinates
    d = np.cross(fn1, fn2)
    norm_factor = 1 / np.linalg.norm(d)
    d *= norm_factor
    m = fd2 * fn1 - fd1 * fn2
    m *= norm_factor

    b1 = np.eye(4)

    # Take an arbitrary point on the line to serve as the b1 translation component
    b1[0:3, 3] = np.cross(d, m)
    b1[0:3, 2] = fn1
    b1[0:3, 1] = d
    b1[0:3, 0] = np.cross(d, fn1)

    # Now, build a matrix to perform the unfold in basis b1
    # (which is now just a rotation along the Y axis)
    sin_theta = -np.linalg.norm(np.cross(fn1, fn2))
    cos_theta = np.dot(fn1, fn2)
    r = np.eye(4)
    r[0,0] = cos_theta
    r[0,2] = sin_theta
    r[2,0] = -sin_theta
    r[2,2] = cos_theta

    # Return R in the basis of b1
    return b1 @ r @ np.linalg.inv(b1)


def unfold(es, fs, first_face, include_faces):
    face_connections = [set() for _ in range(len(fs))]
    for (ei, (vi1, vi2)) in enumerate(es):
        # Find faces that contain this edge
        faces = set(fi for fi in range(len(fs))
            if vi1 in fs[fi] and vi2 in fs[fi])

        # Connect them to each other
        for fi in faces:
            face_connections[fi] |= {(fi, ei) for fi in faces - {fi}}

    # Breadth-first search of faces
    # to generate a spanning tree
    # and compute unfold matrices along the way

    strategy = "bfs"
    unvisited = set(include_faces)
    matrices = [np.eye(4) for _ in range(len(fs))]
    shared_edges = set()

    to_visit = [(None, first_face, None)]

    while to_visit:
        (f1, f2, ei) = to_visit.pop()
        if f2 in unvisited:
            if f1 is not None:
                matrices[f2] = matrices[f1] @ unfold_matrix(face_normals[f1], face_ds[f1], face_normals[f2], face_ds[f2])
                shared_edges.add(ei)
            unvisited.remove(f2)
            for nf, ei in face_connections[f2]:
                if strategy == "dfs":
                    to_visit.append((f2, nf, ei))
                elif strategy == "bfs":
                    to_visit.insert(0, (f2, nf, ei))
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

    assert not unvisited

    return matrices, shared_edges

# Split into multiple PCBs and unfold

def split_and_unfold(vs, es, fs, face_splits):
    vs = np.array(vs)

    unfolded_matrices = [None] * len(fs)
    shared_edges = set()
    root_faces = []

    flat_board_offset = [200, 0]
    offset_m = np.eye(4)
    offset_m[0:2, 3] = flat_board_offset
    cur_offset_m = np.eye(4)

    for include_faces in face_splits:

        # Pick first face by which is closest to the center of all included faces
        center_of_each_included_face = np.array([np.mean(vs[fs[fi]], axis=0) for fi in include_faces])
        center_of_all_included_faces = np.mean(center_of_each_included_face, axis=0)
        first_face = include_faces[np.argmin(np.linalg.norm(center_of_each_included_face - center_of_all_included_faces, axis=1))]

        partial_unfolded_matrices, partial_shared_edges = unfold(es, fs, first_face, include_faces)

        # For each separate board,
        # move the first face to the origin
        # and then offset it by some amount

        t = np.mean(vs[fs[first_face]], axis=0)
        z = t / np.linalg.norm(t)
        y_options = [np.array([0, 1, 0]), np.array([1, 0, 0])]
        for y in y_options:
            if np.linalg.norm(np.cross(y, z)) > 1e-5:
                break
        else:
            assert False, "Found no good Y basis direction"
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)

        root_m = np.eye(4)
        root_m[0:3, 0] = x
        root_m[0:3, 1] = y
        root_m[0:3, 2] = z
        root_m[0:3, 3] = t
        root_m = np.linalg.inv(root_m)

        for (fi, m) in enumerate(partial_unfolded_matrices):
            if fi in include_faces:
                unfolded_matrices[fi] = cur_offset_m @ root_m @ m

        cur_offset_m = offset_m @ cur_offset_m

        shared_edges |= partial_shared_edges
        root_faces.append(first_face)

    return root_faces, unfolded_matrices, shared_edges

face_splits = [
    face_order[:len(face_order) // 2],
    face_order[len(face_order) // 2:],
]

root_faces, unfolded_matrices, shared_edges = split_and_unfold(dvs, des, dfs, face_splits)

# Designate each unused edge with a connector pads for + or -
def route_power(es, fs, edge_order, shared_edges):
    edge_labels = [None] * len(es)

    # See which edges are joined by the PCB
    for ei in shared_edges:
        assert edge_labels[ei] is None
        edge_labels[ei] = "shared"

    # See which edges already carry data
    for ei in edge_order:
        if edge_labels[ei] is None:
            edge_labels[ei] = "data"
        elif edge_labels[ei] == "shared":
            edge_labels[ei] = "shared,data"

    # Greedily assign + and - to the rest:
    # Find the least balanced of the least valence faces

    face_connections = [set() for _ in range(len(fs))]
    for (ei, (vi1, vi2)) in enumerate(es):
        # Find faces that contain this edge
        faces = set(fi for fi in range(len(fs))
            if vi1 in fs[fi] and vi2 in fs[fi])

        # Connect them to each other
        for fi in faces:
            face_connections[fi].add(ei)

    def face_valence_and_balance(fi):
        eis = face_connections[fi]
        valence = sum(edge_labels[ei] is None for ei in eis)
        pos = sum(edge_labels[ei] == "pos" for ei in eis)
        neg = sum(edge_labels[ei] == "neg" for ei in eis)
        return (valence, pos - neg)

    def edge_valence_and_balance(ei):
        assert edge_labels[ei] is None
        fis = [fi for (fi, eis) in enumerate(face_connections) if ei in eis]
        assert len(fis) == 2
        face_valence_and_balance_values = [face_valence_and_balance(fi) for fi in fis]
        valences, balances = zip(*face_valence_and_balance_values)
        assert min(valences) > 0 # At least this edge should be an option
        return (min(valences), max(balances, key=abs))

    while True:
        eis = [ei for (ei, label) in enumerate(edge_labels) if label is None]
        if not eis:
            break

        # Sort first by lowest valence, then by largest imbalance
        def heuristic_key(ei):
            (valence, balance) = edge_valence_and_balance(ei)
            return -valence + abs(balance)

        target_ei = max(eis, key=heuristic_key)
        (valence, balance) = edge_valence_and_balance(target_ei)
        if balance > 0:
            edge_labels[target_ei] = "neg"
        elif balance < 0:
            edge_labels[target_ei] = "pos"
        else:
            edge_labels[target_ei] = random.choice(("pos", "neg"))

    worst_balance = max((face_valence_and_balance(fi)[0] for fi in range(len(fs))), key=abs)

    return edge_labels, worst_balance

edge_labels, worst_balance = route_power(des, dfs, edge_order, shared_edges)

# Make sure power distribution is somewhat balanced
assert abs(worst_balance) < 2

# Draw flat pattern

def xf_for_face(fi):
    def xf(v):
        a = np.array([0., 0., 0., 1.])
        a[0:3] = v
        a = unfolded_matrices[fi] @ a
        a[2] = 0
        return a[0:3]

    return xf

#for fi in range(len(dfs)):
#    allow_vertices = [vi for vi in range(len(dvs)) if vi in dfs[fi]]
#
#    xf = xf_for_face(fi)
#
#    for (ei, (dvi1, dvi2)) in enumerate(des):
#        if ei not in shared_edges and dvi1 in allow_vertices and dvi2 in allow_vertices:
#            ax.plot(
#                *zip(xf(dvs[dvi1]), xf(dvs[dvi2])),
#                c="black",
#            )
#
#            txt, c = {
#                "pos": ("+", "red"),
#                "neg": ("-", "black"),
#            }.get(edge_labels[ei], ("", None))
#            if txt:
#                ax.text(*xf(0.5 * (dvs[dvi1] + dvs[dvi2])), s=txt, zdir=None, color=c)
#
#    # Draw LEDs
#    for vi in range(len(vs)):
#        if v_fis[vi] != fi:
#            continue
#
#        led_size = 2.5
#        pts = np.array([
#            [-1, -1, 0, 1],
#            [-1, 1, 0, 1],
#            #[1, 1, 0, 1],
#            [1, 0, 0, 1],
#            #[1,-1, 0, 1],
#            [-1, -1, 0, 1],
#            [-1, 1, 0, 1],
#            [1, 1, 0, 1],
#            [1,-1, 0, 1],
#            [-1, -1, 0, 1],
#        ]) * np.array([0.5 * led_size, 0.5 * led_size, 0.5 * led_size, 1])
#
#        pts = (led_matrices[vi] @ pts.T).T
#
#        ax.plot(
#            *zip(*(xf(pt[0:3]) for pt in pts)),
#            c="black",
#        )

def draw_routing(vs, es, fs, fine_vs, face_order, edge_order):
    vs = np.array(vs)

    input_edge_order = edge_order[:-1]
    output_edge_order = edge_order[1:]

    # Inputs
    for fi, ei in zip(face_order, input_edge_order):
        if ei is not None:
            edge_pt = np.mean(vs[es[ei]], axis=0)
            #face_pt = np.mean(vs[fs[fi]], axis=0)
            vertex_order, _ = face_fine_routing[fi]
            face_pt = fine_vs[vertex_order[0]]
            xf = xf_for_face(fi)
            ax.plot(*zip(xf(edge_pt), xf(face_pt)), color="green")

    # Outputs
    for fi, ei in zip(face_order, output_edge_order):
        if ei is not None:
            edge_pt = np.mean(vs[es[ei]], axis=0)
            #face_pt = np.mean(vs[fs[fi]], axis=0)
            vertex_order, _ = face_fine_routing[fi]
            face_pt = fine_vs[vertex_order[-1]]
            xf = xf_for_face(fi)
            ax.plot(*zip(xf(edge_pt), xf(face_pt)), color="red")

    # Fine routing
    for fi in face_order:
        vertex_order, _ = face_fine_routing[fi]
        xf = xf_for_face(fi)
        path = np.array([fine_vs[vi] for vi in vertex_order])
        ax.plot(*zip(*[xf(pt) for pt in path]), color="gray")

#draw_routing(dvs, des, dfs, vs, face_order, edge_order)
#plt.show()

def make_uuid():
    return str(uuid.uuid4())

def export_board_preamble():
    return """(kicad_pcb
    (version 20240108)
    (generator "pcbnew")
    (generator_version "8.0")
    (general
        (thickness 1.6)
        (legacy_teardrops no)
    )
    (paper "A4")
    (layers
        (0 "F.Cu" signal)
        (31 "B.Cu" signal)
        (32 "B.Adhes" user "B.Adhesive")
        (33 "F.Adhes" user "F.Adhesive")
        (34 "B.Paste" user)
        (35 "F.Paste" user)
        (36 "B.SilkS" user "B.Silkscreen")
        (37 "F.SilkS" user "F.Silkscreen")
        (38 "B.Mask" user)
        (39 "F.Mask" user)
        (40 "Dwgs.User" user "User.Drawings")
        (41 "Cmts.User" user "User.Comments")
        (42 "Eco1.User" user "User.Eco1")
        (43 "Eco2.User" user "User.Eco2")
        (44 "Edge.Cuts" user)
        (45 "Margin" user)
        (46 "B.CrtYd" user "B.Courtyard")
        (47 "F.CrtYd" user "F.Courtyard")
        (48 "B.Fab" user)
        (49 "F.Fab" user)
        (50 "User.1" user)
        (51 "User.2" user)
        (52 "User.3" user)
        (53 "User.4" user)
        (54 "User.5" user)
        (55 "User.6" user)
        (56 "User.7" user)
        (57 "User.8" user)
        (58 "User.9" user)
    )
    (setup
        (pad_to_mask_clearance 0)
        (allow_soldermask_bridges_in_footprints no)
        (pcbplotparams
            (layerselection 0x00010fc_ffffffff)
            (plot_on_all_layers_selection 0x0000000_00000000)
            (disableapertmacros no)
            (usegerberextensions no)
            (usegerberattributes yes)
            (usegerberadvancedattributes yes)
            (creategerberjobfile yes)
            (dashed_line_dash_ratio 12.000000)
            (dashed_line_gap_ratio 3.000000)
            (svgprecision 4)
            (plotframeref no)
            (viasonmask no)
            (mode 1)
            (useauxorigin no)
            (hpglpennumber 1)
            (hpglpenspeed 20)
            (hpglpendiameter 15.000000)
            (pdf_front_fp_property_popups yes)
            (pdf_back_fp_property_popups yes)
            (dxfpolygonmode yes)
            (dxfimperialunits yes)
            (dxfusepcbnewfont yes)
            (psnegative no)
            (psa4output no)
            (plotreference yes)
            (plotvalue yes)
            (plotfptext yes)
            (plotinvisibletext no)
            (sketchpadsonfab no)
            (subtractmaskfromsilk no)
            (outputformat 1)
            (mirror no)
            (drillshape 1)
            (scaleselection 1)
            (outputdirectory "")
        )
    )
"""

def generate_board_nets(vs, es, fs, fine_vs, face_order, edge_order):
    vs = np.array(vs)
    input_edge_order = edge_order[:-1]
    output_edge_order = edge_order[1:]

    nets = []

    # Global
    nets.append("")
    nets.append("GND")
    nets.append("VCC")

    # Inputs
    for fi, ei in zip(face_order, input_edge_order):
        if ei is not None:
            edge_pad = f"PAD{ei}-{fi}"
            vertex_order, _ = face_fine_routing[fi]
            led = f"LED{vertex_order[0]}"
            nets.append(f"({edge_pad})({led}-DI)")

    # Outputs
    for fi, ei in zip(face_order, output_edge_order):
        if ei is not None:
            edge_pad = f"PAD{ei}-{fi}"
            #face_pt = np.mean(vs[fs[fi]], axis=0)
            vertex_order, _ = face_fine_routing[fi]
            led = f"LED{vertex_order[-1]}"
            nets.append(f"({led}-DO)({edge_pad})")

    # Fine routing
    for fi in face_order:
        vertex_order, _ = face_fine_routing[fi]
        for (vi1, vi2) in zip(vertex_order[:-1], vertex_order[1:]):
            led1 = f"LED{vi1}"
            led2 = f"LED{vi2}"
            nets.append(f"({led1}-DO)({led2}-DI)")

    # Assign numbers
    nets = {net: f'{i} "{net}"' for i, net in enumerate(nets)}

    return nets

def export_board_nets(nets):
    export = ""
    for net_name in nets.values():
        export += f"""    (net {net_name})
"""
    return export

def net_name_for_pad(nets, pad):
    for net, net_name in nets.items():
        if f"({pad})" in net:
            return net_name

def export_board_outline(vs, es, fs, nets):
    # Draw lines on the Edge.Cuts layer
    # in the shape of the boards

    # Convert flattened 3D lines into closed 2D polygons

    # First, determine the topology
    flat_edges_points = []

    for fi in range(len(fs)):
        allow_vertices = [vi for vi in range(len(vs)) if vi in fs[fi]]

        xf = xf_for_face(fi)

        for (ei, (vi1, vi2)) in enumerate(es):
            if ei not in shared_edges and vi1 in allow_vertices and vi2 in allow_vertices:
                start = xf(vs[vi1])[0:2] * KICAD_FLIP
                end = xf(vs[vi2])[0:2] * KICAD_FLIP
                flat_edges_points.append((start, end))

    flat_vertices = []
    flat_edges = []

    for (fv1, fv2) in flat_edges_points:
        eis = []
        for fv in (fv1, fv2): 
            for (i, test_v) in enumerate(flat_vertices):
                if np.linalg.norm(test_v - fv) < 1e-5:
                    fvi = i
                    break # Already in list
            else:
                fvi = len(flat_vertices)
                flat_vertices.append(fv)
            eis.append(fvi)
        flat_edges.append(tuple(eis))

    # Now, walk each polygon to get connectivity and orient each edge correctly
    vertices_to_visit = set(range(len(flat_vertices)))
    polygons = []
    while vertices_to_visit:
        first_vertex = next(iter(vertices_to_visit))
        cur_vertex = first_vertex
        polygon = []
        while True:
            possible_next_vertices = []
            for vi1, vi2 in flat_edges:
                if vi1 == cur_vertex and vi2 in vertices_to_visit:
                    possible_next_vertices.append(vi2)
                elif vi2 == cur_vertex and vi1 in vertices_to_visit:
                    possible_next_vertices.append(vi1)
            if not possible_next_vertices:
                assert False, "No outgoing edge found"

            # Avoid going back to the start if there's a second option
            if len(possible_next_vertices) > 1 and first_vertex in possible_next_vertices:
                possible_next_vertices.remove(first_vertex)

            next_vertex = possible_next_vertices.pop()

            vertices_to_visit.remove(next_vertex)
            polygon.append(next_vertex)
            if next_vertex == first_vertex:
                break
            cur_vertex = next_vertex

        # See if the polygon is wound forwards or backwards
        # by measuring its area (negative area = backwards)
        area = 0
        for (vi1, vi2) in zip(polygon, polygon[1:] + [polygon[0]]):
            (v1x, v1y) = flat_vertices[vi1]
            (v2x, v2y) = flat_vertices[vi2]
            area += (v2y - v1y) * 0.5 * (v1x + v2x)

        if area < 0:
            polygon = list(reversed(polygon))

        polygons.append(polygon) 

    #def offset_v(vi_prev, vi, vi_next):
    #    rot90 = np.array([[0, 1], [-1, 0]])

    #    vp = np.array(flat_vertices[vi_prev])
    #    v = np.array(flat_vertices[vi])
    #    vn = np.array(flat_vertices[vi_next])

    #    offset_p = rot90 @ (v - vp)
    #    offset_p = offset_dist * offset_p / np.linalg.norm(offset_p)
    #    offset_n = rot90 @ (vn - v)
    #    offset_n = offset_dist * offset_n / np.linalg.norm(offset_n)

    #    def line_through(pt1, pt2):
    #        # Line through 2 points represented in homogeneous line coordinates
    #        n = rot90 @ (pt2 - pt1)
    #        n = n / np.linalg.norm(n)
    #        d = np.dot(pt1, n)
    #        l = np.array(n.tolist() + [-d])
    #        return l

    #    lp = line_through(vp + offset_p, v + offset_p)
    #    ln = line_through(v + offset_n, vn + offset_n)

    #    # Find the intersection of the two offset lines
    #    pt = np.cross(lp, ln)
    #    pt = pt[0:2] / pt[2]

    #    return pt

    #offset_flat_vertices = [None] * len(flat_vertices)
    #for polygon in polygons:
    #    for (vi_prev, vi, vi_next) in zip([polygon[-1]] + polygon[:-1], polygon, polygon[1:] + [polygon[0]]):
    #        assert offset_flat_vertices[vi] is None
    #        offset_flat_vertices[vi] = offset_v(vi_prev, vi, vi_next)

    export = ""

    angular_offset = np.asin(offset_dist / circle_r)

    for polygon in polygons:
        for (vi1, vi2, vi3) in zip([polygon[-1]] + polygon[:-1], polygon, polygon[1:] + [polygon[0]]):
            circle_center = flat_vertices[vi2]
            circle_start_vector = flat_vertices[vi1] - flat_vertices[vi2]
            circle_start_angle = np.atan2(circle_start_vector[1], circle_start_vector[0]) + angular_offset
            circle_end_vector = flat_vertices[vi3] - flat_vertices[vi2]
            circle_end_angle = np.atan2(circle_end_vector[1], circle_end_vector[0]) - angular_offset
            circle_swept_angle = (((circle_start_angle - circle_end_angle) % (2*np.pi)) + (2*np.pi)) % (2*np.pi)
            circle_mid_angle = circle_end_angle + 0.5 * circle_swept_angle

            circle_start_pt = circle_center + circle_r * np.array([np.cos(circle_start_angle), np.sin(circle_start_angle)])
            circle_end_pt = circle_center + circle_r * np.array([np.cos(circle_end_angle), np.sin(circle_end_angle)])
            circle_mid_pt = circle_center + circle_r * np.array([np.cos(circle_mid_angle), np.sin(circle_mid_angle)])

            next_circle_center = flat_vertices[vi3]
            next_circle_start_vector = flat_vertices[vi2] - flat_vertices[vi3]
            next_circle_start_angle = np.atan2(next_circle_start_vector[1], next_circle_start_vector[0]) + angular_offset
            next_circle_start_pt = next_circle_center + circle_r * np.array([np.cos(next_circle_start_angle), np.sin(next_circle_start_angle)])

            export += f"""        (gr_arc
            (start {circle_start_pt[0]:.3f} {circle_start_pt[1]:.3f})
            (mid {circle_mid_pt[0]:.3f} {circle_mid_pt[1]:.3f})
            (end {circle_end_pt[0]:.3f} {circle_end_pt[1]:.3f})
            (stroke
                (width 0.05)
                (type default)
            )
            (layer "Edge.Cuts")
            (uuid "{make_uuid()}")
        )
        (gr_line
            (start {circle_end_pt[0]:.3f} {circle_end_pt[1]:.3f})
            (end {next_circle_start_pt[0]:.3f} {next_circle_start_pt[1]:.3f})
            (stroke
                (width 0.05)
                (type default)
            )
            (layer "Edge.Cuts")
            (uuid "{make_uuid()}")
        )
"""

    # Add filled zones
    (gnd_num, gnd_name) = nets["GND"].split(" ")
    (vcc_num, vcc_name) = nets["VCC"].split(" ")

    for polygon in polygons:
        vs_in_poly = np.array(flat_vertices)[polygon]
        (xmin, ymin) = np.min(vs_in_poly, axis=0)
        (xmax, ymax) = np.max(vs_in_poly, axis=0)

        export += f"""    (zone
        (net {vcc_num})
        (net_name {vcc_name})
        (layer "F.Cu")
        (uuid "07f8a733-4d93-41dd-8ef6-34681b09ab16")
        (hatch edge 0.5)
        (connect_pads
            (clearance 0.2)
        )
        (min_thickness 0.2)
        (filled_areas_thickness no)
        (fill
            (thermal_gap 0.3)
            (thermal_bridge_width 0.2)
        )
        (polygon
            (pts
                (xy {xmin} {ymin}) (xy {xmax} {ymin}) (xy {xmax} {ymax}) (xy {xmin} {ymax})
            )
        )
    )
    (zone
        (net {gnd_num})
        (net_name {gnd_name})
        (layer "B.Cu")
        (uuid "07f8a733-4d93-41dd-8ef6-34681b09ab16")
        (hatch edge 0.5)
        (connect_pads
            (clearance 0.2)
        )
        (min_thickness 0.2)
        (filled_areas_thickness no)
        (fill
            (thermal_gap 0.3)
            (thermal_bridge_width 0.2)
        )
        (polygon
            (pts
                (xy {xmin} {ymin}) (xy {xmax} {ymin}) (xy {xmax} {ymax}) (xy {xmin} {ymax})
            )
        )
    )
"""

    return export

def export_board_folds(vs, es, fs):
    # Draw lines on the F.SilkS layer
    # and B.SilkS layer
    # indicating folds

    export = ""

    remaining_shared_edges = set(shared_edges)

    for fi in range(len(fs)):
        allow_vertices = [vi for vi in range(len(vs)) if vi in fs[fi]]

        xf = xf_for_face(fi)

        for (ei, (vi1, vi2)) in enumerate(es):
            if ei in remaining_shared_edges and vi1 in allow_vertices and vi2 in allow_vertices:
                remaining_shared_edges.remove(ei)

                start = xf(vs[vi1])[0:2] * KICAD_FLIP
                end = xf(vs[vi2])[0:2] * KICAD_FLIP

                tangent = end - start
                tangent = circle_r * tangent / np.linalg.norm(tangent)

                # Shrink to avoid circle edge cuts
                line_start = start + tangent
                line_end = end - tangent

                width = 0.1

                normal = np.array([[0, -1], [1, 0]]) @ (line_end - line_start)
                normal = 0.5 * width * normal / np.linalg.norm(normal)

                rect_1 = line_start - normal
                rect_2 = line_end - normal
                rect_3 = line_end + normal
                rect_4 = line_start + normal

                export += f"""    (gr_line
        (start {line_start[0]:.3f} {line_start[1]:.3f})
        (end {line_end[0]:.3f} {line_end[1]:.3f})
        (stroke
            (width 0.12)
			(type dash)
        )
        (layer "F.SilkS")
        (uuid "{make_uuid()}")
    )
    (gr_line
        (start {line_start[0]:.3f} {line_start[1]:.3f})
        (end {line_end[0]:.3f} {line_end[1]:.3f})
        (stroke
            (width 0.12)
			(type dash)
        )
        (layer "B.SilkS")
        (uuid "{make_uuid()}")
    )
	(zone
		(net 0)
		(net_name "")
		(layers "F&B.Cu")
		(uuid "{make_uuid()}")
		(hatch edge 0.5)
		(connect_pads
			(clearance 0)
		)
		(min_thickness 0.25)
		(filled_areas_thickness no)
		(keepout
			(tracks not_allowed)
			(vias not_allowed)
			(pads not_allowed)
			(copperpour allowed)
			(footprints allowed)
		)
		(fill
			(thermal_gap 0.5)
			(thermal_bridge_width 0.5)
		)
		(polygon
			(pts
				(xy {rect_1[0]:.3f} {rect_1[1]:.3f})
				(xy {rect_2[0]:.3f} {rect_2[1]:.3f})
				(xy {rect_3[0]:.3f} {rect_3[1]:.3f})
				(xy {rect_4[0]:.3f} {rect_4[1]:.3f})
			)
		)
	)
"""

    return export

def export_one_footprint(footprint_file_content, m, ref, pad_nets, other_replacements=None):
    if other_replacements is None:
        other_replacements = {}

    pos = m[0:2, 3] * KICAD_FLIP
    xvec = (np.linalg.inv(m) @ np.array([1, 0, 0, 0]))[0:2] * KICAD_FLIP
    angle = np.rad2deg(np.atan2(xvec[1], xvec[0]))
    angle = ((angle % 360) + 360) % 360

    footprint_lines = footprint_file_content.split("\n")
    footprint_lines = [l.replace("\t", "    ") for l in footprint_lines]
    footprint_lines = ["    " + line for line in footprint_lines]
    footprint_lines.insert(1, f'        (uuid "{make_uuid()}")')
    footprint_lines.insert(2, f'        (at {pos[0]:.3f} {pos[1]:.3f})')

    # Write pad & text angles
    # Add angle=0 if angle is omitted
    footprint_lines = [
        re.sub(r'\(at ([^ ]+) ([^ ]+)\)', rf'(at \1 \2 0)', line)
        for line in footprint_lines
    ]

    # Add rotation to existing angle
    def adjust_angle(line):
        match = re.search(r'\(at ([^ ]+) ([^ ]+) ([^ ]+)\)', line)
        if not match:
            return line
        (x, y, orig_angle) = match.groups()
        line = line[:match.start()] + f"(at {x} {y} {float(orig_angle) + angle:.3f})" + line[match.end():]
        return line

    footprint_lines = [
        adjust_angle(line)
        for line in footprint_lines
    ]

    # Write reference
    footprint_lines = [
        re.sub(r'\(property "Reference" "([^"]*)"', rf'(property "Reference" "{ref}"', line)
        for line in footprint_lines
    ]

    # Write pad nets
    for i, line in enumerate(footprint_lines):
        match = re.search(r'\(pad "([^"]*)"', line)
        if match is not None:
            (pad,) = match.groups()
            net = pad_nets.get(pad)
            if net:
                footprint_lines[i] = line + f"\n            (net {net})"

    # We added lines in a hacky way, so re-split
    footprint_lines = "\n".join(footprint_lines).split("\n")

    # Perform other replacements
    for (search_for, replace_with) in other_replacements.items():
        footprint_lines = [
            line.replace(search_for, replace_with)
            for line in footprint_lines
        ]

    return "\n".join(footprint_lines)

def export_board_leds(coarse_vs, coarse_fs, unfolded_matrices, v_fis, led_matrices, nets, footprint_file, pad_functions):
    with open(footprint_file) as f:
        footprint_file_content = f.read()

    export = ""

    for coarse_fi in range(len(coarse_fs)):
        for vi in range(len(led_matrices)):
            if v_fis[vi] != coarse_fi:
                continue

            m = unfolded_matrices[coarse_fi] @ led_matrices[vi]

            ref = f"LED{vi}"

            def net_for_function(pad_function):
                if pad_function == "DI":
                    return net_name_for_pad(nets, f"{ref}-DI")
                if pad_function == "DO":
                    return net_name_for_pad(nets, f"{ref}-DO")
                return nets[pad_function]

            pad_nets = {pad: net_for_function(pad_function) for (pad, pad_function) in pad_functions.items()}

            export += export_one_footprint(footprint_file_content, m, ref=ref, pad_nets=pad_nets)

    return export

def export_board_seam_pads(vs, es, fs, fine_vs, unfolded_matrices, edge_labels, v_fis, face_order, edge_order, nets, footprint_file):
    with open(footprint_file) as f:
        footprint_file_content = f.read()

    export = ""

    for fi, f in enumerate(fs):
        xf = xf_for_face(fi)

        face_center = np.mean(vs[np.array(f)], axis=0)
        face_center = xf(face_center)

        prior_faces = face_order[:face_order.index(fi)]
        prior_leds_to_this_face = [fine_vi for fine_vi in range(len(fine_vs)) if v_fis[fine_vi] in prior_faces]
        leds_on_this_face = [fine_vi for fine_vi in range(len(fine_vs)) if v_fis[fine_vi] == fi]

        input_to_led = len(prior_leds_to_this_face)
        output_to_led = input_to_led + len(leds_on_this_face)

        for ei, (vi1, vi2) in enumerate(es):
            if vi1 not in fs[fi] or vi2 not in fs[fi]:
                continue

            v1 = xf(vs[vi1])
            v2 = xf(vs[vi2])

            # Build a basis centered on this edge
            t = 0.5 * (v1 + v2)
            z = np.array([0, 0, 1])
            y = v2 - v1

            # Edges are in arbitrary order, so flip if necessary
            if np.dot(np.cross(face_center - t, y), z) < 0:
                y = -y

            y = y / np.linalg.norm(y)
            x = np.cross(y, z)
            m = np.eye(4)
            m[0:3, 0] = x
            m[0:3, 1] = y
            m[0:3, 2] = z
            m[0:3, 3] = t

            ref = f"PAD{ei}-{fi}"

            # Figure out what net this pad is
            label = edge_labels[ei]
            pad_nets = {}
            desc = "NC"
            if "data" in label:
                pad_nets["1"] = net_name_for_pad(nets, ref)

                # Figure out if this is an input or output
                foi = face_order.index(fi)
                if edge_order[foi] == ei:
                    desc = f"↑{input_to_led}↑"
                elif edge_order[foi+1] == ei:
                    desc = f"↓{output_to_led}↓"
                # It's possible neither of these are true
                # if this is the start or end

            elif label == "pos":
                pad_nets["1"] = nets["VCC"]
                desc = "+"
            elif label == "neg":
                pad_nets["1"] = nets["GND"]
                desc = "-"
            # if label is "shared" then there is no net

            export += export_one_footprint(footprint_file_content, m, ref=ref, pad_nets=pad_nets, other_replacements={"${DESCRIPTION}": desc})

    return export


    # Inputs
    for fi, ei in zip(face_order, input_edge_order):
        if ei is not None:
            edge_pad = f"PAD{ei}-{fi}"
            vertex_order, _ = face_fine_routing[fi]
            led = f"LED{vertex_order[0]}"
            nets.append(f"({edge_pad})({led}-DI)")

    # Outputs
    for fi, ei in zip(face_order, output_edge_order):
        if ei is not None:
            edge_pad = f"PAD{ei}-{fi}"
            #face_pt = np.mean(vs[fs[fi]], axis=0)
            vertex_order, _ = face_fine_routing[fi]
            led = f"LED{vertex_order[-1]}"
            nets.append(f"({led}-DO)({edge_pad})")


def export_board_postamble():
    return """)
"""

def export_board(coarse_vs, coarse_es, coarse_fs, vs, face_order, edge_order, unfolded_matrices, v_fis, led_matrices, edge_labels, filename):
    nets = generate_board_nets(coarse_vs, coarse_es, coarse_fs, vs, face_order, edge_order)
    content = "".join([
        export_board_preamble(),
        export_board_nets(nets),
        export_board_outline(coarse_vs, coarse_es, coarse_fs, nets),
        export_board_folds(coarse_vs, coarse_es, coarse_fs),
        export_board_leds(coarse_vs, coarse_fs, unfolded_matrices, v_fis, led_matrices, nets,
            footprint_file="lib.pretty/LED_SK6805_EC10_1111.kicad_mod",
            pad_functions={"1": "GND", "2": "DI", "3": "VCC", "4": "DO"},
        ),
        export_board_seam_pads(coarse_vs, coarse_es, coarse_fs, vs, unfolded_matrices, edge_labels, v_fis, face_order, edge_order, nets,
            footprint_file="lib.pretty/flex_seam_pad_small.kicad_mod",
        ),
        export_board_postamble(),
    ])
    with open(filename, "w") as f:
        f.write(content)

export_board(dvs, des, dfs, vs, face_order, edge_order, unfolded_matrices, v_fis, led_matrices, edge_labels, filename="kicad/orb/orb_generated.kicad_pcb")
