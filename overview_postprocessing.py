from utils import available_wells, load_well, stitch_arrays
from scipy.ndimage import zoom
import imageio
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Post processing for well overviews.")
    parser.add_argument(dest='folder', type=Path, help='Full path to folder of overview acquisition.')
    parser.add_argument('-d', '--downsampling', type=float, default=0.5)
    args = parser.parse_args()
    out_folder = args.folder / 'stitched'
    out_folder.mkdir(exist_ok=True)

    for well in available_wells(args.folder):
        print(f"Processing well {well}...")
        names, tiles = load_well(args.folder, well)
        stitched = stitch_arrays(tiles)
        stitched_ds = zoom(stitched, args.downsampling)
        imageio.imwrite(out_folder / f"{well}.png", stitched_ds)
    print("Done.")


if __name__ == '__main__':
    main()
