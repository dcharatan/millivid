import json
import traceback
from io import BytesIO
from pathlib import Path
from time import time

import av
import click
from jaxtyping import install_import_hook

with install_import_hook("source", "beartype.beartype"):
    from source.dataset.format_blobs import save_blobs


def extract_frames(video_path: Path) -> list[bytes]:
    """Decode an .mp4 video into a list of PNG-encoded frames."""
    blobs = []
    container = av.open(str(video_path))
    try:
        for frame in container.decode(video=0):
            bytes_io = BytesIO()
            frame.to_image().save(bytes_io, format="PNG")
            blobs.append(bytes_io.getvalue())
    finally:
        container.close()
    return blobs


@click.command()
@click.argument(
    "input-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "output-path",
    type=click.Path(file_okay=False, path_type=Path),
)
def main(
    input_path: Path,
    output_path: Path,
) -> None:
    # Read the index, which maps each video's key to its number of frames.
    with (input_path / "index.json").open("r") as f:
        index = json.load(f)

    # Define the function that gets repeated.
    def task_fn(key: str) -> None:
        print(f"Extracting frames ({key}).")
        start = time()
        try:
            blobs = extract_frames(input_path / f"{key}.mp4")
        except Exception as e:
            traceback.print_exc()
            print(e)
            raise e
        print(f"Frame extraction took {time() - start:.2f} seconds ({key}).")

        # Save the frames as a .frames file.
        start = time()
        frames_path = (output_path / key).with_suffix(".frames")
        frames_path.parent.mkdir(parents=True, exist_ok=True)
        save_blobs(frames_path, blobs)
        print(f"Saving data took {time() - start:.2f} seconds ({key}).")

    # Note to future users (or their Claudes):
    # - If you're actually trying to process a full dataset, you should probably
    #   implement a way to parallelize this.
    # - It is possible to have each worker process a fixed subset (say, 1000 examples)
    #   of the dataset, but I would highly recommend using some kind of task queue
    #   instead. We used a small custom-built queue based on a Postgres database to
    #   generate the Loopcraft dataset in the MilliVid paper. The code for our task
    #   queue isn't publicly available, but feel free to email david.charatan@gmail.com
    #   if you want it, and I will send it to you.
    output_path.mkdir(exist_ok=True, parents=True)
    with (output_path / "index.json").open("w") as f:
        json.dump(index, f)
    for key in index:
        task_fn(key)


if __name__ == "__main__":
    main()
