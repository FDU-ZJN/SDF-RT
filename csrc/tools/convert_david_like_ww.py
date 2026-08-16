#!/usr/bin/env python3
"""Convert the David bust to the ww.obj camera, scale, and OBJ conventions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_obj(path: Path):
    positions = []
    normals = []
    faces = []
    with path.open("r", encoding="ascii") as obj:
        for line_number, raw in enumerate(obj, 1):
            fields = raw.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                positions.append(tuple(map(float, fields[1:4])))
            elif fields[0] == "vn":
                normals.append(tuple(map(float, fields[1:4])))
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(f"{path}:{line_number}: expected a triangular face")
                face = []
                for reference in fields[1:]:
                    indices = reference.split("/")
                    if len(indices) != 3 or not indices[0] or not indices[2]:
                        raise ValueError(f"{path}:{line_number}: expected v/vt/vn or v//vn")
                    vi = int(indices[0])
                    ni = int(indices[2])
                    vi = vi - 1 if vi > 0 else len(positions) + vi
                    ni = ni - 1 if ni > 0 else len(normals) + ni
                    face.append((vi, ni))
                faces.append(face)
    if not positions or not normals or not faces:
        raise ValueError(f"{path}: OBJ must contain positions, normals, and faces")
    return positions, normals, faces


def bounds(points):
    lower = [min(point[axis] for point in points) for axis in range(3)]
    upper = [max(point[axis] for point in points) for axis in range(3)]
    center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
    extent = [upper[axis] - lower[axis] for axis in range(3)]
    return lower, upper, center, extent


def rotate_front_to_camera(value):
    # David's front is +X; the renderer camera looks from +Z toward the origin.
    # This is a proper -90 degree rotation around Y, so winding is preserved.
    x, y, z = value
    return -z, y, x


def unit(value):
    length = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(length) or length <= 1.0e-20:
        raise ValueError("source OBJ contains an invalid normal")
    return tuple(component / length for component in value)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=script_dir.parent / "david.obj")
    parser.add_argument("--reference", type=Path, default=script_dir.parent / "ww.obj")
    parser.add_argument("--output", type=Path, default=script_dir.parent / "david_ww.obj")
    args = parser.parse_args()

    source_positions, source_normals, source_faces = parse_obj(args.input)
    reference_positions, _, _ = parse_obj(args.reference)
    rotated_positions = [rotate_front_to_camera(position) for position in source_positions]
    _, _, source_center, source_extent = bounds(rotated_positions)
    reference_lower, reference_upper, reference_center, reference_extent = bounds(reference_positions)
    source_max_edge = max(source_extent)
    target_max_edge = max(reference_extent)
    scale = target_max_edge / source_max_edge

    def transform_position(position):
        rotated = rotate_front_to_camera(position)
        return tuple(
            reference_center[axis] + (rotated[axis] - source_center[axis]) * scale
            for axis in range(3)
        )

    def transform_normal(normal):
        return unit(rotate_front_to_camera(normal))

    pair_to_index = {}
    output_positions = []
    output_normals = []
    output_faces = []
    for face in source_faces:
        output_face = []
        for vi, ni in face:
            if not 0 <= vi < len(source_positions) or not 0 <= ni < len(source_normals):
                raise ValueError("source OBJ face index is out of range")
            pair = (vi, ni)
            if pair not in pair_to_index:
                pair_to_index[pair] = len(output_positions) + 1
                output_positions.append(transform_position(source_positions[vi]))
                output_normals.append(transform_normal(source_normals[ni]))
            output_face.append(pair_to_index[pair])
        output_faces.append(tuple(output_face))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="\n") as obj:
        obj.write("# David aligned to ww.obj: front +Z, Y-up, centered on ww bbox\n")
        obj.write(f"# target_max_edge {target_max_edge:.12g}\n")
        obj.write(f"# uniform_scale {scale:.12g}\n")
        for position in output_positions:
            obj.write("v {:.8f} {:.8f} {:.8f}\n".format(*position))
        for normal in output_normals:
            obj.write("vn {:.8f} {:.8f} {:.8f}\n".format(*normal))
        for face in output_faces:
            obj.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(*face))

    output_lower, output_upper, output_center, output_extent = bounds(output_positions)
    print(f"wrote {args.output}: vertices={len(output_positions)} normals={len(output_normals)} faces={len(output_faces)}")
    print(f"source center after rotation: {tuple(source_center)}")
    print(f"uniform scale: {scale:.12g}")
    print(f"bounds min: {tuple(output_lower)}")
    print(f"bounds max: {tuple(output_upper)}")
    print(f"center: {tuple(output_center)}")
    print(f"extent: {tuple(output_extent)}; max edge: {max(output_extent):.12g}")
    print(f"ww reference center: {tuple(reference_center)}; max edge: {target_max_edge:.12g}")


if __name__ == "__main__":
    main()
