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
    parser.add_argument('-c', '--channel', type=str, default='C02')
    parser.add_argument('--alert', action='store_true')
    args = parser.parse_args()
    out_folder = args.folder / 'stitched'
    out_folder.mkdir(exist_ok=True)

    for well in available_wells(args.folder):
        print(f"Processing well {well}...")
        out_fn = out_folder / f"{well}.png"
        if out_fn.exists():
            print(f"Processed already.")
            print()
            continue
        print(f'Loading files...')
        names, tiles = load_well(args.folder, well, channel=args.channel)
        print(f'stitching...')
        stitched = stitch_arrays(tiles)
        print(f'downsampling...')
        stitched_ds = zoom(stitched, args.downsampling)
        print(f'writing file {out_fn.name}...')
        print()
        imageio.imwrite(out_fn, stitched_ds)
    if args.alert:
        import winsound
        winsound.MessageBeep()
    print("Done.")


if __name__ == '__main__':
    main()
