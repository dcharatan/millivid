import os
from io import BytesIO
from pathlib import Path

HEADER_BYTES = 8


def save_blobs(path: Path, blobs: list[bytes]):
    bytes_io = BytesIO()

    # Write the header to memory.
    offset = 0
    for blob in blobs:
        offset += len(blob)
        bytes_io.write(offset.to_bytes(HEADER_BYTES, byteorder="little", signed=False))

    # Write the body to memory.
    for blob in blobs:
        bytes_io.write(blob)

    # Use a single write to transfer everything to disk.
    with path.open("wb") as f:
        f.write(bytes_io.getbuffer())


def load_blobs(
    path: Path,
    num_blobs: int,
    start_blob_index: int,  # inclusive
    end_blob_index: int,  # exclusive
) -> tuple[bytes, ...]:
    valid = (
        (start_blob_index < end_blob_index)
        and (0 <= start_blob_index < num_blobs)
        and (0 < end_blob_index <= num_blobs)
    )
    if not valid:
        raise ValueError("Invalid blob range!")

    header_len = HEADER_BYTES * num_blobs
    fd = os.open(path, os.O_RDONLY)

    try:
        # Read the header. This will contain the offsets for all frames.
        header = os.pread(fd, header_len, 0)

        first_blob_start = HEADER_BYTES * start_blob_index
        if start_blob_index == 0:
            first_start_offset = 0
        else:
            first_start_offset = int.from_bytes(
                header[first_blob_start - HEADER_BYTES : first_blob_start],
                byteorder="little",
                signed=False,
            )

        last_blob_start = HEADER_BYTES * (end_blob_index - 1)
        last_end_offset = int.from_bytes(
            header[last_blob_start : last_blob_start + HEADER_BYTES],
            byteorder="little",
            signed=False,
        )

        # Load the entire video in memory once (only one syscall).
        full_video = os.pread(
            fd,
            last_end_offset - first_start_offset,
            header_len + first_start_offset,
        )

        # Now split it up.
        blobs = []
        ptr = 0
        for blob_index in range(start_blob_index, end_blob_index):
            current_start = HEADER_BYTES * blob_index
            if blob_index == 0:
                current_start_offset = 0
            else:
                current_start_offset = int.from_bytes(
                    header[current_start - HEADER_BYTES : current_start],
                    byteorder="little",
                    signed=False,
                )
            current_end_offset = int.from_bytes(
                header[current_start : current_start + HEADER_BYTES],
                byteorder="little",
                signed=False,
            )
            frame_len = current_end_offset - current_start_offset
            blobs.append(full_video[ptr : ptr + frame_len])
            ptr += frame_len

        return tuple(blobs)

    finally:
        os.close(fd)
