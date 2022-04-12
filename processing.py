import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(r'processing.log'),
        logging.StreamHandler()
    ]
)

try:
    import imageio
    from pathlib import Path
    import numpy as np
    from skimage.feature import match_template
    from skimage.morphology import local_maxima, remove_small_objects
    from scipy.ndimage import zoom
    from skimage import filters
    from skimage.measure import label, regionprops
    import matplotlib.pyplot as plt
    import seaborn as sns
    import csv
    import napari
    import argparse
    from utils import available_wells, load_well, stitch_arrays, unstitch_arrays
    import random
except Exception as import_exception:
    logging.error(f'{import_exception}')


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


def find_objects_by_template_matching(stitched_ds, object_threshold, template_path, downsampling, well,
                                      n_objects_per_site):
    if template_path is None:
        template_path = Path(r'C:\Users\CVUser\Documents\Python\searchFirst\templates\template_ZE_9x.tif')
    logging.info(f"loading template from {template_path}...")
    template = imageio.imread(template_path)
    template_ds = zoom(template, downsampling)

    match = match_template(stitched_ds, template_ds, pad_input=True, mode='constant', constant_values=100)
    match_thresholded = np.where(match > object_threshold, match, 0)
    if np.sum(match_thresholded) == 0:
        logging.warning(
            f"no matches found in {well}! Try lowering the `object_threshold` if you expected to find matches in this well.")
        return np.zeros_like(stitched_ds), np.zeros_like(stitched_ds)
    maxima = local_maxima(match_thresholded)
    n_objects = np.sum(maxima)
    logging.info(f'{n_objects} objects found...')
    score = match[np.where(maxima)]
    n_actual = n_objects_per_site
    if n_objects < n_objects_per_site:
        logging.warning(f"only {n_objects} objects found instead of {n_objects_per_site}")
        n_actual = n_objects
    nth_largest_score = -np.partition(-score, n_actual - 1)[n_actual - 1]
    weighted_maxima = np.where(maxima, match, 0)
    selected_objects = np.where(weighted_maxima >= nth_largest_score, weighted_maxima, 0)
    unselected_objects = np.where(np.logical_and(weighted_maxima > 0, weighted_maxima < nth_largest_score),
                                  weighted_maxima, 0)
    return selected_objects, unselected_objects


def find_objects_by_multiple_template_matching(stitched_ds, object_threshold,
                                               template_path, downsampling,
                                               well, n_objects_per_site):
    if template_path is None:
        template_path = Path((r'C:\Users\CVUser\Documents\Python\searchFirst'
                              '\templates\template_ZE_9x.tif'))
    logging.info(f"loading template from {template_path}...")

    # get list of template files
    template_path = Path(template_path)
    template_files = template_path.glob('*.tif')

    # initialize arrays for object matches from all templates combined
    all_selected = np.empty(np.shape(stitched_ds))
    all_unselected = np.empty(np.shape(stitched_ds))

    # iterate over templates
    for fyle in template_files:
        template = imageio.imread(fyle)
        template_ds = zoom(template, downsampling)

        match = match_template(stitched_ds, template_ds, pad_input=True,
                               mode='constant', constant_values=100)
        match_thresholded = np.where(match > object_threshold, match, 0)
        if np.sum(match_thresholded) == 0:
            logging.warning(
                f"no matches found in {well} for template {fyle}! '"
                f"'Try lowering the `object_threshold` if you expected to'"
                f"' find matches in this well.")
            continue
        maxima = local_maxima(match_thresholded)
        n_objects = np.sum(maxima)
        logging.info(f'{n_objects} objects found for template {fyle}...')
        score = match[np.where(maxima)]
        n_actual = n_objects_per_site
        if n_objects < n_objects_per_site:
            logging.warning(f"only {n_objects} objects found instead of '"
                            f"'{n_objects_per_site}")
            n_actual = n_objects

        nth_largest_score = -np.partition(-score, n_actual - 1)[n_actual - 1]
        weighted_maxima = np.where(maxima, match, 0)
        selected_objects = np.where(weighted_maxima >= nth_largest_score,
                                    weighted_maxima, 0)
        unselected_objects = np.where(
            np.logical_and(weighted_maxima > 0,
                           weighted_maxima < nth_largest_score),
            weighted_maxima, 0)

        all_selected += selected_objects
        all_unselected += unselected_objects

    return all_selected, all_unselected


def find_objects_by_threshold(stitched_ds, sigma, minimum_object_size):
    # Normalize stitched_ds
    img = stitched_ds / np.amax(stitched_ds)
    # initialize canvas of zeroes
    selected_objects = np.zeros(img.shape)

    gaussian = filters.gaussian(img, sigma=sigma)

    threshold_gaussian = filters.threshold_otsu(gaussian)

    binary_gaussian = gaussian >= threshold_gaussian

    masked = remove_small_objects(binary_gaussian, minimum_object_size)

    labeled_blobs = label(masked)

    props = regionprops(labeled_blobs)

    for props in props:
        a = props.centroid
        selected_objects[int(a[0]), int(a[1])] = 1

    return selected_objects, np.zeros_like(selected_objects)


def find_objects_by_manual_annotation(stitched_ds):
    viewer = napari.Viewer()
    viewer.add_image(stitched_ds)
    # rescale stitched image
    low, high = np.quantile(stitched_ds, [0.0001, 0.9999])
    viewer.layers['stitched_ds'].contrast_limits = [low, high]
    viewer.add_points(None)
    viewer.layers['Points'].mode = 'add'
    napari.run()

    # after the viewer is closed, the following will be executed:
    coords = viewer.layers['Points'].data
    n_objects = len(coords)
    if n_objects == 0:
        logging.warning('no coordinates were annotated...')
    else:
        logging.info(f'{n_objects} coordinates were annotated...')
    selected_objects = np.empty(np.shape(stitched_ds))
    selected_objects[coords[:, 0].astype('int'),
                     coords[:, 1].astype('int')] = 1
    unselected_objects = np.empty(np.shape(stitched_ds))

    return selected_objects, unselected_objects


def plot_results(stitched, selected_maxima, unselected_maxima, out_file=None, ny=5, nx=4):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(stitched, cmap='gray')
    y_true, x_true = np.where(selected_maxima)
    y_false, x_false = np.where(unselected_maxima)
    sns.scatterplot(x=x_true, y=y_true, color='#35d130', s=90)
    sns.scatterplot(x=x_false, y=y_false, color='#e63232', s=90)
    ax.set_yticks(np.arange(ny) * stitched.shape[0] / ny)
    ax.set_xticks(np.arange(nx) * stitched.shape[1] / nx)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    plt.grid()
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()


if __name__ == '__main__':
    main()
