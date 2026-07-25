#!/usr/bin/env python3
"""Generate the normalized level-1 Loop-subdivided Killeroo OBJ asset."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path


EXPECTED_CONTROL_VERTICES = 4290
EXPECTED_CONTROL_FACES = 8316
EXPECTED_VERTICES = 16899
EXPECTED_FACES = 33264


def parse_array(text: str, declaration: str) -> list[float]:
    match = re.search(re.escape(declaration) + r"\s*\[([^]]*)\]", text, re.S)
    if not match:
        raise ValueError(f"missing PBRT array: {declaration}")
    return [float(value) for value in match.group(1).split()]


def parse_control_mesh(path: Path):
    text = path.read_text(encoding="utf-8")
    raw_points = parse_array(text, '"point3 P"')
    raw_indices = parse_array(text, '"integer indices"')
    if len(raw_points) % 3 or len(raw_indices) % 3:
        raise ValueError("PBRT mesh arrays are not triangle-aligned")
    points = [tuple(raw_points[i : i + 3]) for i in range(0, len(raw_points), 3)]
    faces = [tuple(int(v) for v in raw_indices[i : i + 3]) for i in range(0, len(raw_indices), 3)]
    if len(points) != EXPECTED_CONTROL_VERTICES or len(faces) != EXPECTED_CONTROL_FACES:
        raise ValueError(
            f"unexpected control mesh: {len(points)} vertices, {len(faces)} faces"
        )
    return points, faces


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(v):
    length = math.sqrt(sum(component * component for component in v))
    if not math.isfinite(length) or length <= 1e-20:
        raise ValueError("cannot normalize zero or non-finite vector")
    return tuple(component / length for component in v)


def topology(vertex_count, faces):
    neighbors = [set() for _ in range(vertex_count)]
    edge_opposites = defaultdict(list)
    for a, b, c in faces:
        for u, v, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            key = (min(u, v), max(u, v))
            edge_opposites[key].append(opposite)
            neighbors[u].add(v)
            neighbors[v].add(u)
    boundary_neighbors = [set() for _ in range(vertex_count)]
    for (u, v), opposites in edge_opposites.items():
        if len(opposites) == 1:
            boundary_neighbors[u].add(v)
            boundary_neighbors[v].add(u)
        elif len(opposites) != 2:
            raise ValueError(f"non-manifold edge {(u, v)} has {len(opposites)} faces")
    return neighbors, boundary_neighbors, edge_opposites


def weighted_vertex(point, ring, points, beta):
    result = mul(point, 1.0 - beta * len(ring))
    for index in ring:
        result = add(result, mul(points[index], beta))
    return result


def loop_subdivide_once(points, faces):
    neighbors, boundary, edges = topology(len(points), faces)
    refined = []
    for index, point in enumerate(points):
        if boundary[index]:
            ring = sorted(boundary[index])
            if len(ring) != 2:
                raise ValueError(f"boundary vertex {index} has {len(ring)} boundary neighbors")
            refined.append(weighted_vertex(point, ring, points, 1.0 / 8.0))
        else:
            valence = len(neighbors[index])
            beta = 3.0 / 16.0 if valence == 3 else 3.0 / (8.0 * valence)
            refined.append(weighted_vertex(point, neighbors[index], points, beta))

    edge_vertex = {}
    for edge in sorted(edges):
        u, v = edge
        opposites = edges[edge]
        if len(opposites) == 1:
            point = mul(add(points[u], points[v]), 0.5)
        else:
            point = add(
                mul(add(points[u], points[v]), 3.0 / 8.0),
                mul(add(points[opposites[0]], points[opposites[1]]), 1.0 / 8.0),
            )
        edge_vertex[edge] = len(refined)
        refined.append(point)

    refined_faces = []
    for a, b, c in faces:
        ab = edge_vertex[(min(a, b), max(a, b))]
        bc = edge_vertex[(min(b, c), max(b, c))]
        ca = edge_vertex[(min(c, a), max(c, a))]
        refined_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
    return refined, refined_faces


def project_to_limit(points, faces):
    neighbors, boundary, _ = topology(len(points), faces)
    limit = []
    for index, point in enumerate(points):
        if boundary[index]:
            ring = sorted(boundary[index])
            limit.append(weighted_vertex(point, ring, points, 1.0 / 5.0))
        else:
            valence = len(neighbors[index])
            beta = 3.0 / 16.0 if valence == 3 else 3.0 / (8.0 * valence)
            gamma = 1.0 / (valence + 3.0 / (8.0 * beta))
            limit.append(weighted_vertex(point, neighbors[index], points, gamma))
    return limit


def smooth_normals(points, faces):
    accum = [(0.0, 0.0, 0.0) for _ in points]
    for a, b, c in faces:
        face_normal = cross(sub(points[b], points[a]), sub(points[c], points[a]))
        accum[a] = add(accum[a], face_normal)
        accum[b] = add(accum[b], face_normal)
        accum[c] = add(accum[c], face_normal)
    return [normalize(value) for value in accum]


def normalize_mesh(points):
    mins = [min(point[axis] for point in points) for axis in range(3)]
    maxs = [max(point[axis] for point in points) for axis in range(3)]
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    longest = max(maxs[axis] - mins[axis] for axis in range(3))
    scale = 2.0 / longest
    normalized = [mul(sub(point, center), scale) for point in points]
    return normalized, center, scale


def validate(points, normals, faces):
    if len(points) != EXPECTED_VERTICES or len(faces) != EXPECTED_FACES:
        raise ValueError(f"unexpected output mesh: {len(points)} vertices, {len(faces)} faces")
    if len(normals) != len(points):
        raise ValueError("normal count does not match vertex count")
    for face in faces:
        if len(set(face)) != 3 or min(face) < 0 or max(face) >= len(points):
            raise ValueError(f"invalid face indices: {face}")
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("non-finite vertex coordinate")
    for normal in normals:
        length = math.sqrt(sum(value * value for value in normal))
        if not all(math.isfinite(value) for value in normal) or abs(length - 1.0) > 1e-5:
            raise ValueError("invalid unit normal")
    extents = [
        max(point[axis] for point in points) - min(point[axis] for point in points)
        for axis in range(3)
    ]
    if abs(max(extents) - 2.0) > 1e-5:
        raise ValueError(f"normalized longest extent is {max(extents)}, expected 2.0")


def write_obj(path, source, points, normals, faces, center, scale):
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write("# Killeroo model courtesy of headus (http://headus.com.au)\n")
        output.write("# Source: pbrt-v4-scenes/killeroos/geometry/killeroo3.pbrt\n")
        output.write("# Generated by tools/generate_killeroo_gold.py: Loop level 1 + limit projection\n")
        output.write(f"# source_bbox_center {center[0]:.9g} {center[1]:.9g} {center[2]:.9g}\n")
        output.write(f"# source_to_obj_scale {scale:.9g}\n")
        output.write(f"# source_path {source.as_posix()}\n")
        for point in points:
            output.write(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        for normal in normals:
            output.write(f"vn {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n")
        for a, b, c in faces:
            output.write(f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}\n")


def check_obj(path):
    points, normals, faces = [], [], []
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            points.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "vn":
            normals.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f":
            faces.append(tuple(int(value.split("/")[0]) - 1 for value in fields[1:4]))
    validate(points, normals, faces)
    print(f"validated {path}: {len(points)} vertices, {len(faces)} triangles")


def main():
    script_dir = Path(__file__).resolve().parent
    default_source = script_dir.parent.parent / "local/external/software_rt/pbrt-v4-scenes/killeroos/geometry/killeroo3.pbrt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=script_dir.parent / "killeroo_gold.obj")
    parser.add_argument("--check", action="store_true", help="validate an existing output OBJ")
    args = parser.parse_args()
    if args.check:
        check_obj(args.output)
        return

    points, faces = parse_control_mesh(args.source)
    points, faces = loop_subdivide_once(points, faces)
    points = project_to_limit(points, faces)
    points, center, scale = normalize_mesh(points)
    normals = smooth_normals(points, faces)
    validate(points, normals, faces)
    write_obj(args.output, args.source, points, normals, faces, center, scale)
    print(f"wrote {args.output}: {len(points)} vertices, {len(faces)} triangles")
    print(f"source bbox center={center}, source_to_obj_scale={scale:.12g}")


if __name__ == "__main__":
    main()
