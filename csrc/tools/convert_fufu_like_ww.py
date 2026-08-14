#!/usr/bin/env python3
"""Convert fufu to the ww axes/center with an exact target maximum edge."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_obj(path: Path):
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    faces: list[list[tuple[int, int]]] = []

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
                face: list[tuple[int, int]] = []
                for reference in fields[1:]:
                    indices = reference.split("/")
                    if len(indices) != 3 or not indices[0] or not indices[2]:
                        raise ValueError(
                            f"{path}:{line_number}: expected vertex//normal face references"
                        )
                    vertex_index = int(indices[0])
                    normal_index = int(indices[2])
                    vertex_index = vertex_index - 1 if vertex_index > 0 else len(positions) + vertex_index
                    normal_index = normal_index - 1 if normal_index > 0 else len(normals) + normal_index
                    face.append((vertex_index, normal_index))
                faces.append(face)

    if not positions or not normals or not faces:
        raise ValueError(f"{path}: OBJ must contain positions, normals, and faces")
    return positions, normals, faces


def bounds(points: list[tuple[float, float, float]]):
    lower = [min(point[axis] for point in points) for axis in range(3)]
    upper = [max(point[axis] for point in points) for axis in range(3)]
    center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
    extent = [upper[axis] - lower[axis] for axis in range(3)]
    return lower, upper, center, extent


def rotate_from_z_up(vector: tuple[float, float, float]):
    # fufu is Z-up and faces -Y; ww is Y-up and faces +Z.
    return vector[0], vector[2], -vector[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("csrc/fufu.obj"))
    parser.add_argument("--reference", type=Path, default=Path("csrc/ww.obj"))
    parser.add_argument("--output", type=Path, default=Path("csrc/fufu_norm_front.obj"))
    parser.add_argument("--max-edge", type=float, default=1.0)
    args = parser.parse_args()

    source_positions, source_normals, source_faces = parse_obj(args.input)
    reference_positions, _, _ = parse_obj(args.reference)

    rotated_positions = [rotate_from_z_up(position) for position in source_positions]
    _, _, source_center, source_extent = bounds(rotated_positions)
    reference_lower, reference_upper, reference_center, _ = bounds(reference_positions)
    source_max_extent = max(source_extent)
    if not math.isfinite(args.max_edge) or args.max_edge <= 0.0:
        raise ValueError("--max-edge must be finite and positive")
    if not math.isfinite(source_max_extent) or source_max_extent <= 0.0:
        raise ValueError("source OBJ has an invalid bounding box")
    scale = args.max_edge / source_max_extent

    def transform_position(position: tuple[float, float, float]):
        rotated = rotate_from_z_up(position)
        return tuple(
            reference_center[axis] + (rotated[axis] - source_center[axis]) * scale
            for axis in range(3)
        )

    def transform_normal(normal: tuple[float, float, float]):
        rotated = rotate_from_z_up(normal)
        length = math.sqrt(sum(component * component for component in rotated))
        if not math.isfinite(length) or length <= 1.0e-20:
            raise ValueError("source OBJ contains an invalid normal")
        return tuple(component / length for component in rotated)

    pair_to_index: dict[tuple[int, int], int] = {}
    output_positions: list[tuple[float, float, float]] = []
    output_normals: list[tuple[float, float, float]] = []
    output_faces: list[tuple[int, int, int]] = []
    for face in source_faces:
        output_face = []
        for pair in face:
            if pair not in pair_to_index:
                vertex_index, normal_index = pair
                if not 0 <= vertex_index < len(source_positions):
                    raise ValueError(f"face vertex index {vertex_index + 1} is out of range")
                if not 0 <= normal_index < len(source_normals):
                    raise ValueError(f"face normal index {normal_index + 1} is out of range")
                pair_to_index[pair] = len(output_positions) + 1
                output_positions.append(transform_position(source_positions[vertex_index]))
                output_normals.append(transform_normal(source_normals[normal_index]))
            output_face.append(pair_to_index[pair])
        output_faces.append(tuple(output_face))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="\n") as obj:
        obj.write("# fufu normalized to the ww.obj axes and center\n")
        obj.write(f"# target_max_edge {args.max_edge:.12g}\n")
        obj.write(f"# uniform_scale {scale:.12g}\n")
        for position in output_positions:
            obj.write("v {:.8f} {:.8f} {:.8f}\n".format(*position))
        for normal in output_normals:
            obj.write("vn {:.8f} {:.8f} {:.8f}\n".format(*normal))
        for face in output_faces:
            obj.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(*face))

    output_lower, output_upper, _, output_extent = bounds(output_positions)
    print(f"wrote {args.output}: vertices={len(output_positions)} faces={len(output_faces)}")
    print(f"uniform scale: {scale:.12g}")
    print(f"bounds min: {tuple(output_lower)}")
    print(f"bounds max: {tuple(output_upper)}")
    print(f"extent: {tuple(output_extent)}; max edge: {max(output_extent):.12g}")
    print(f"reference min: {tuple(reference_lower)}")
    print(f"reference max: {tuple(reference_upper)}")


if __name__ == "__main__":
    main()
