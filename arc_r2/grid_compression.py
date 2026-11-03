"""Lossless compact encoding for ARC grids and exact search-state fingerprints.

The codec uses palette-indexed run-length encoding only when it is smaller
than raw cells. ARC grids are limited to 30x30. Decoding rejects malformed
and non-canonical payloads and always reconstructs the exact integer grid.
"""
from __future__ import annotations

_MAGIC = b"AGC1"
_MAX_SIDE = 30


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while offset < len(data) and shift <= 28:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid or truncated run length")


def encode_grid(grid: tuple[tuple[int, ...], ...]) -> bytes:
    """Encode a non-empty rectangular ARC grid losslessly."""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be non-empty and rectangular")
    height, width = len(grid), len(grid[0])
    if height > _MAX_SIDE or width > _MAX_SIDE:
        raise ValueError("ARC grid dimensions must not exceed 30x30")
    cells = [cell for row in grid for cell in row]
    if any(type(cell) is not int or not 0 <= cell <= 9 for cell in cells):
        raise ValueError("grid cells must be integers from 0 to 9")
    header = _MAGIC + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    raw = bytes(cells)
    palette = sorted(set(cells))
    indexes = {color: index for index, color in enumerate(palette)}
    runs = bytearray()
    start = 0
    while start < len(cells):
        color = cells[start]
        end = start + 1
        while end < len(cells) and cells[end] == color:
            end += 1
        runs.append(indexes[color])
        runs.extend(_varint(end - start))
        start = end
    packed = bytes([len(palette)]) + bytes(palette) + bytes(runs)
    if len(packed) < len(raw):
        return header + b"\x01" + packed
    return header + b"\x00" + raw


def decode_grid(encoded: bytes) -> tuple[tuple[int, ...], ...]:
    """Decode a canonical encode_grid payload, rejecting malformed data."""
    if not isinstance(encoded, bytes) or len(encoded) < 9 or encoded[:4] != _MAGIC:
        raise ValueError("invalid grid encoding header")
    height = int.from_bytes(encoded[4:6], "big")
    width = int.from_bytes(encoded[6:8], "big")
    mode = encoded[8]
    if not 1 <= height <= _MAX_SIDE or not 1 <= width <= _MAX_SIDE or mode not in (0, 1):
        raise ValueError("invalid grid dimensions or encoding mode")
    payload = encoded[9:]
    count = height * width
    if mode == 0:
        if len(payload) != count or any(value > 9 for value in payload):
            raise ValueError("invalid raw grid payload")
        cells = list(payload)
    else:
        if not payload:
            raise ValueError("missing palette")
        palette_size = payload[0]
        if not 1 <= palette_size <= 10 or len(payload) < 1 + palette_size:
            raise ValueError("invalid palette")
        palette = payload[1:1 + palette_size]
        if (
            any(value > 9 for value in palette)
            or tuple(palette) != tuple(sorted(set(palette)))
        ):
            raise ValueError("palette must contain sorted unique ARC colors")
        cells = []
        offset = 1 + palette_size
        previous_index = None
        while offset < len(payload):
            index = payload[offset]
            offset += 1
            run, offset = _read_varint(payload, offset)
            if (
                index >= palette_size or run < 1
                or index == previous_index
                or len(cells) + run > count
            ):
                raise ValueError("invalid palette run")
            cells.extend([palette[index]] * run)
            previous_index = index
        if len(cells) != count:
            raise ValueError("decoded cell count does not match dimensions")

    grid = tuple(tuple(cells[r * width:(r + 1) * width]) for r in range(height))
    if encode_grid(grid) != encoded:
        raise ValueError("non-canonical grid encoding")
    return grid
