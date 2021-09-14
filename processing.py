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
    from skimage.morphology import extrema
    from scipy.ndimage import zoom
    import matplotlib.pyplot as plt
    import seaborn as sns
    import csv
    import argparse
    from utils import available_wells, load_well, stitch_arrays, unstitch_arrays
    import random
except Exception as import_exception:
    logging.error(f'{import_exception}')


def main():
    logging.info('starting main...')
    try:
        parser = argparse.ArgumentParser(
            description="SearchFirst imaging of wells based on matching a template image. ")
        parser.add_argument(dest='folder', type=str, help='Full path to folder of first pass.')
        parser.add_argument('-n', '--n_objects_per_site', type=int, default=3)
        parser.add_argument('-ot', '--object_threshold', type=float, default=0.5,
                            help='Threshold [0.0 - 1.0] for rejecting objects (default 0.5).')
        parser.add_argument('-d', '--downsampling', type=float, default=0.25,
                            help='Downsampling ratio to speed up processing (default 0.25).')
        parser.add_argument('-t', '--template_path', type=str,
                            help='Full path to template image. Default is to search for `template.tif` in the folder.')
        parser.add_argument('-p', '--plot_output', type=bool, default=True,
                            help='Whether to generate plots of the results.')

        args = parser.parse_args()
        run_processing(args.folder, args.object_threshold, args.downsampling, args.n_objects_per_site,
                       args.template_path, args.plot_output)
    except Exception as runtime_exception:
        logging.error(f'{runtime_exception}')


def run_processing(fld, object_threshold, downsampling, n_objects_per_site, template_path=None, plot_output=True):
    fld = Path(fld)

    if not fld.is_dir():
        raise NotADirectoryError(f"Directory {fld.as_posix()} does not exist!")

    logging.info(f'processing folder {fld.as_posix()}')
    if template_path is None:
        template_path = Path(r'C:\Users\CVUser\Documents\Python\searchFirst\templates\template_ZE_10x.tif')
    logging.info(f"loading template from {template_path}...")
    template = imageio.imread(template_path)

    template_ds = zoom(template, downsampling)

    for well in list(available_wells(fld)):
        logging.info(f'processing well {well}...')
        logging.info('loading images...')
        names, imgs = load_well(fld, well)
        logging.info(f'processing {len(imgs)} images...')
        stitched = stitch_arrays(imgs, ny=5, nx=4)

        stitched_ds = zoom(stitched, downsampling)

        match = match_template(stitched_ds, template_ds, pad_input=True, mode='constant', constant_values=100)
        match_thresholded = np.where(match > object_threshold, match, 0)
        if np.sum(match_thresholded) == 0:
            logging.warning(f"no matches found in {well}! Try lowering the `object_threshold` if you expected to find matches in this well.")
            continue
        maxima = extrema.h_maxima(match_thresholded, h=object_threshold)
        n_objects = np.sum(maxima)
        logging.info(f'{n_objects} objects found...')
        score = match[np.where(maxima)]
        nth_largest_score = -np.partition(-score, n_objects_per_site-1)[n_objects_per_site-1]
        weighted_maxima = np.where(maxima, match, 0)
        selected_maxima = np.where(weighted_maxima >= nth_largest_score, weighted_maxima, 0)

        if plot_output:
            unselected_maxima = np.where(np.logical_and(weighted_maxima > 0, weighted_maxima < nth_largest_score),
                                         weighted_maxima, 0)
            plot_results(stitched_ds, selected_maxima, unselected_maxima, out_file=fld / f'plot_{well}.png')

        site_maxima = unstitch_arrays(selected_maxima, ny=5, nx=4)

        for name, site_maximum in zip(names, site_maxima):
            ys, xs = np.where(site_maximum)
            if len(xs) > 0:
                cv_name = name.parent / f'{name.stem}.csv'
                with open(cv_name, 'w', newline='') as f:
                    c = csv.writer(f)
                    for i, (x, y) in enumerate(zip(xs, ys)):
                        c.writerow([i + 1, int(x / downsampling), int(y / downsampling)])


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
