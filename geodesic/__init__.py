#!/usr/bin/env python3

import numpy as np
import itertools
from scipy.spatial import ConvexHull

phi = 0.5 * (1 + np.sqrt(5))
EPSILON = 1e-6

icosahedron_vertices = np.array([
    [0, 1, phi],
    [0, 1, -phi],
    [0, -1, phi],
    [0, -1, -phi],
    [1, phi, 0],
    [1, -phi, 0],
    [-1, phi, 0],
    [-1, -phi, 0],
    [phi, 0, 1],
    [phi, 0, -1],
    [-phi, 0, 1],
    [-phi, 0, -1],
]) / np.linalg.norm([1, phi])

# Rotate the vertices so that a vertex lies on top
c = phi / np.sqrt(1 + phi**2)
s = -1 / np.sqrt(1 + phi**2)
rotation_matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
icosahedron_vertices = np.dot(rotation_matrix, icosahedron_vertices.T).T

tetrahedron_vertices = np.array([
    [1, 0, -np.sqrt(0.5)],
    [-1, 0, -np.sqrt(0.5)],
    [0, 1, np.sqrt(0.5)],
    [0, -1, np.sqrt(0.5)],
]) / np.linalg.norm([1, np.sqrt(0.5)])

# Rotate the vertices so that a vertex lies on top
c = np.sqrt(1. / 3.)
s = -np.sqrt(2. / 3.)
rotation_matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
tetrahedron_vertices = np.dot(rotation_matrix, tetrahedron_vertices.T).T

octahedron_vertices = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
]) / np.linalg.norm([1, 1])

def field_z(pt):
    return np.array([1, 0, 0])

def field_radial(pt):
    return np.array(pt)

def field_from_vertices(vertices):
    def _field(pt):
        vectors = pt - vertices
        mags = np.maximum(np.linalg.norm(vectors, axis=1), 1e-5)
        return np.sum(vectors / mags[:,None] ** 2, axis=0)
    return _field

def field_from_faces(faces, vertices):
    face_centers = np.array([(vertices[i1] + vertices[i2] + vertices[i3]) / 3 for i1, i2, i3 in faces])

    def _field(pt):
        vectors = pt - face_centers
        mags = np.maximum(np.linalg.norm(vectors, axis=1), 1e-5)
        directions = np.cross(face_centers, pt)
        directions = directions / np.maximum(np.linalg.norm(directions, axis=1)[:,None], 1e-5)
        return np.sum(directions / mags[:,None], axis=0)
    return _field

def field_from_polyhedron(faces, vertices, curl_factor=1e-3):
    ffv = field_from_vertices(vertices)
    fff = field_from_faces(faces, vertices)
    return lambda pt: ffv(pt) + curl_factor * fff(pt)

def edges_from_faces(faces):
    edges = set()
    for f in faces:
        edges.add(frozenset((f[0], f[1])))
        edges.add(frozenset((f[0], f[2])))
        edges.add(frozenset((f[1], f[2])))

    return [list(e) for e in edges]

#def faces_from_points(points):
#    faces = []
#    for (i, j, k) in itertools.combinations(range(len(points)), 3):
#        o = points[i]
#        normal = np.cross(points[j] - o, points[k] - o)
#        sides = np.dot(points - o, normal)
#        if np.all(sides < EPSILON) or np.all(sides > -EPSILON):
#            faces.append([i, j, k])
#
#    return faces

def faces_from_points(points):
    return ConvexHull(points).simplices

def orient_edges(edges, points, field=field_z):
    """ Flips edges so that they align with the given vector field
    """
    def flip_edge(e):
        pt1, pt2 = [points[i] for i in e]
        midpoint = 0.5 * (pt1 + pt2)
        direction = np.dot(pt2 - pt1, field(midpoint))
        return direction < 0

    return [e[::-1] if flip_edge(e) else e for e in edges]

def orient_faces(faces, points, field=field_radial):
    """ Flips triangles so that they are as close as possible to isosceles in the ABA representation,
    and wound so that their normal aligns with the given vector field.
    """
    def sort_triangle(f):
        (a, b, c) = f
        vec1 = points[b] - points[a]
        vec2 = points[c] - points[b]
        centroid = (points[a] + points[b] + points[c]) / 3
        flip_winding = np.dot(np.cross(vec1, vec2), field(centroid)) < 0
        triangle = (c, b, a) if flip_winding else (a, b, c)
        # The middle point is the one that is abnormally close or abnormally far from the centroid
        distance_to_centroid = np.array([np.linalg.norm(points[i] - centroid) for i in triangle])
        middle_point = np.argmax(np.abs(distance_to_centroid - np.mean(distance_to_centroid)))
        triangle = (triangle * 3)[middle_point + 2:middle_point + 5]
        return list(triangle)

    return [sort_triangle(f) for f in faces]

def subdivide_triangle(pt1, pt2, pt3, u=0, v=0):
    assert u != 0 or v != 0
    inv_det = 1 / (u ** 2 + u * v + v ** 2)
    a = (pt2 - pt1) * inv_det
    b = (pt3 - pt1) * inv_det
    uhat = a * (u + v) - b * v
    vhat = a * v + b * u

    # Generate a superset of the points we need, in UV space
    uv_pts = np.array([[i, j] for i in range(-v, u + v + 1) for j in range(0, u + v + 1 - i)])

    # Discard points that lie outside of the triangle in UV space
    def inside_triangle(uv):
        a = np.array([u, v])
        b = np.array([-v, u + v])
        # We're working with integers so no epsilon needed!
        return np.cross(uv, a) <= 0 and np.cross(uv, b) >= 0 and np.cross(uv - a, b - a) <= 0

    return [pt1 + uhat * uv[0] + vhat * uv[1] for (uv) in uv_pts if inside_triangle(uv)]

def deduplicate_points(points):
    new_points = np.empty(shape=(0, 3))
    for point in points:
        if not np.any(np.linalg.norm(new_points - point, axis=1) < EPSILON):
            new_points = np.vstack((new_points, point))
    return new_points

def subdivide_faces(faces, points, u=0, v=0):
    new_points = [pt for (i1, i2, i3) in faces for pt in subdivide_triangle(points[i1], points[i2], points[i3], u=u, v=v)]
    return deduplicate_points(new_points)

def project_points_to_sphere(points):
    return points / np.linalg.norm(points, axis=1)[:, None]

def matrix_for_vertex(point, field=field_z):
    z = point
    x = field(point)

    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    if np.linalg.norm(y) < EPSILON:
        x = np.array([1, 0, 0])
        y = np.cross(z, x)
        if np.linalg.norm(y) < EPSILON:
            x = np.array([0, 1, 0])
            y = np.cross(z, x)
            assert np.linalg.norm(y) >= EPSILON
    y = y / np.linalg.norm(y)
    x = np.cross(z, y)

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = point
    return result

def matrix_for_edge(v1, v2):
    translation = 0.5 * (v1 + v2)
    x = v2 - v1

    x = x / np.linalg.norm(x)
    y = np.cross(translation, x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = translation
    return result

def matrix_for_face(v1, v2, v3):
    translation = (v1 + v2 + v3) / 3
    y = v2 - 0.5 * (v1 + v3)
    x = v1 - v3

    x = x / np.linalg.norm(x)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = translation
    return result

def vertex_matrices(points, field=field_z):
    return [matrix_for_vertex(point, field).tolist() for point in points]

def edge_matrices(edges, points):
    return [matrix_for_edge(points[i1], points[i2]).tolist() for i1, i2 in edges]

def face_matrices(faces, points):
    return [matrix_for_face(points[i1], points[i2], points[i3]).tolist() for i1, i2, i3 in faces]

def edge_lengths(edges, points):
    """ Returns a list parallel to edges where each entry is the length of that edge
    """
    return np.array([np.linalg.norm(points[i2] - points[i1]) for i1, i2 in edges])

def vertex_edges(edges):
    """ Returns a list parallel to vertices where each entry is a list of edge indices
    for edges incident on that point.
    """
    n_points = max(max(edges, key=max)) + 1
    result = [set() for _ in range(n_points)]
    for (ei, (vi1, vi2)) in enumerate(edges):
        result[vi1].add(ei)
        result[vi2].add(ei)
    return [list(edges) for edges in result]

def vertex_faces(faces):
    """ Returns a list parallel to vertices where each entry is a list of face indices
    for faces containing that point.
    """
    n_points = max(max(faces, key=max)) + 1
    result = [set() for _ in range(n_points)]
    for (ei, (vi1, vi2, vi3)) in enumerate(faces):
        result[vi1].add(ei)
        result[vi2].add(ei)
        result[vi3].add(ei)
    return [list(faces) for faces in result]

def face_edges(faces, edges):
    """ Returns a list parallel to faces where each entry is a list of edge indices
    for edges bordering that face.
    """
    result = []
    for f in faces:
        sf = set(f)
        aes = [i for (i, e) in enumerate(edges) if set(e).issubset(sf)]
        result.append(aes)
    return result

def face_triangles_2d(faces, points, face_matrices):
    """ Returns the three points of a face triangle in the coordinate frame of the face transform.
    These points correspond to the face when drawn in the XY plane and transformed by the face matrix.
    """
    def _tri(face, matrix):
        tri_pts = points[face]
        tri_pts = np.vstack((tri_pts.T, np.ones(3)))
        tri_pts = np.dot(np.linalg.inv(matrix), tri_pts)
        tri_pts = tri_pts[0:2,:].T
        return tri_pts
    return [_tri(face, matrix) for face, matrix in zip(faces, face_matrices)]

def filter_vertices(criterion, points, edges=None, faces=None):
    """ Applies the function `criterion` to each point in `points`, and keeps it if the function returns True.
    Discarded points are removed from the set, and the indices in `edges` and `faces` are adjusted accordingly.
    Returns a tuple containing the new points, edges, and faces (if edges and faces were given)
    """

    keep = np.array([criterion(pt) for pt in points])

    new_points = points[keep]
    result = [new_points]

    lookup = np.cumsum(keep) - 1
    keep_indices = np.arange(len(points))[keep]
    if edges is not None:
        new_edges = np.array([[lookup[i1], lookup[i2]] for (i1, i2) in edges if i1 in keep_indices and i2 in keep_indices])
        result.append(new_edges)

    if faces is not None:
        new_faces = np.array([[lookup[i1], lookup[i2], lookup[i3]] for (i1, i2, i3) in faces if i1 in keep_indices and i2 in keep_indices])
        result.append(new_faces)

    return tuple(result)

def sphere(u=2, v=0, base=None):
    """ Returns the vertices, edges, and faces of a geodesic sphere.
    """
    vs = base if base is not None else icosahedron_vertices
    fs = faces_from_points(vs)
    fs = orient_faces(fs, vs)
    field = field_from_polyhedron(fs, vs)

    vs = subdivide_faces(fs, vs, u=u, v=v)
    vs = project_points_to_sphere(vs)
    fs = faces_from_points(vs)
    fs = orient_faces(fs, vs)
    es = edges_from_faces(fs)
    es = orient_edges(es, vs, field=field)

    return (vs, es, fs)
