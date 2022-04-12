import logging
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom
import csv
import argparse
from utils import available_wells, load_well, stitch_arrays, unstitch_arrays, plot_results
from processing_methods import find_objects_by_threshold, \
    find_objects_by_template_matching, \
    find_objects_by_multiple_template_matching, \
    find_objects_by_manual_annotation

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(r'processing.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="SearchFirst imaging of wells based on matching a template stitched_ds. ")
    parser.add_argument(dest='folder', type=str, help='Full path to folder of first pass.')
    parser.add_argument('-d', '--downsampling', type=float, default=0.25,
                        help='Downsampling ratio to speed up processing (default 0.25).')
    parser.add_argument('-n', '--n_objects_per_site', type=int, default=6)
    parser.add_argument('-p', '--plot_output', type=bool, default=True,
                        help='Whether to generate plots of the results.')
    parser.add_argument('-m', '--method', type=str, default='template',
                        help="""Method to use for object detection. Either `template`, 'multi-template', `threshold` or 'manual'. Make sure you
                        specify the required arguments for the method you chose.""")
    parser.add_argument('-ch', '--channel', type=str, default='C02', help='Channel of the overview acquisition.')

    # Template machting arguments
    parser.add_argument('-ot', '--object_threshold', type=float, default=0.5,
                        help='Threshold [0.0 - 1.0] for rejecting objects (default 0.5).')
    parser.add_argument('-t', '--template_path', type=str, default=None,
                        help="""Full path to template stitched_ds. Default is to search for `template.tif` in the folder.
                         If method 'multi-template' is used, all tiff files in the folder will be used as template.""")

    # Thresholding arguments
    parser.add_argument('-s', '--sigma', type=float, default=7,
                        help='Sigma of the gaussian filter to apply before thresholding.')
    parser.add_argument('-mos', '--minimum_object_size', type=int, default=1000,
                        help='Minimum object size in pixels in the downsampled stitched_ds.')

    args = parser.parse_args()

    fld = Path(args.folder)
    if not fld.is_dir():
        raise NotADirectoryError(f"Directory {fld.as_posix()} does not exist!")

    logging.info(f'processing folder {fld.as_posix()}')
    for well in list(available_wells(fld)):
        logging.info(f'processing well {well}...')
        logging.info('loading images...')
        names, imgs = load_well(fld, well, channel=args.channel)
        logging.info(f'processing {len(imgs)} images...')
        stitched = stitch_arrays(imgs, ny=5, nx=4)

        stitched_ds = zoom(stitched, args.downsampling)

        if args.method == 'template':
            objects, non_objects = find_objects_by_template_matching(stitched_ds,
                                                                     object_threshold=args.object_threshold,
                                                                     template_path=args.template_path,
                                                                     downsampling=args.downsampling,
                                                                     n_objects_per_site=args.n_objects_per_site,
                                                                     well=well,
                                                                     )
        elif args.method == 'multi-template':
            objects, non_objects = find_objects_by_multiple_template_matching(stitched_ds,
                                                                              object_threshold=args.object_threshold,
                                                                              template_path=args.template_path,
                                                                              downsampling=args.downsampling,
                                                                              n_objects_per_site=args.n_objects_per_site,
                                                                              well=well,
                                                                              )
        elif args.method == 'threshold':
            objects, non_objects = find_objects_by_threshold(stitched_ds,
                                                             sigma=args.sigma,
                                                             minimum_object_size=args.minimum_object_size,
                                                             )
        elif args.method == 'manual':
            objects, non_objects = find_objects_by_manual_annotation(stitched_ds,
                                                                     )
        else:
            raise NotImplementedError(f"Method `{args.method}` is not available. Use either `template`, 'multi-template' or `threshold`.")

        if args.plot_output:
            plot_results(stitched_ds, objects, non_objects, out_file=fld / f'plot_{well}.png')

        site_objects = unstitch_arrays(objects, ny=5, nx=4)

        for name, site_object in zip(names, site_objects):
            ys, xs = np.where(site_object)
            if len(xs) > 0:
                cv_name = name.parent / f'{name.stem}.csv'
                with open(cv_name, 'w', newline='') as f:
                    c = csv.writer(f)
                    for i, (x, y) in enumerate(zip(xs, ys)):
                        c.writerow([i + 1, int(x / args.downsampling), int(y / args.downsampling)])





if __name__ == '__main__':
    main()
