import sys
import napari
import numpy as np
from pathlib import Path
import imageio
import re
from utils import available_wells, load_well, stitch_arrays


def main():
    fld = Path(sys.argv[1])

    imgs = []
    for w in available_wells(fld):
        print(w)
        _, tiles = load_well(fld, well_name=w)
        imgs.append(stitch_arrays(tiles))

    viewer = napari.Viewer()
    viewer.add_image(np.stack(imgs))
    points = viewer.add_points(ndim=3)

    @viewer.bind_key('s')
    def template_crop(viewer, radius=512):
        out = []
        points = viewer.layers['Points'].data
        img = viewer.layers['Image'].data
        for point in points.astype(int):
            z, y, x = point
            crp = img[z, y - radius:y + radius, x - radius:x + radius].copy()
            out.append(crp)
            out.append(np.rot90(crp))
        result = np.stack(out).mean(axis=0).astype(img.dtype)
        rotations = []
        for i in range(4):
            rot = np.rot90(result, k=i)
            rotations.append(rot)
            rotations.append(np.fliplr(rot))
        results_rot = np.stack(rotations).mean(axis=0).astype(img.dtype)
        print("writing result...")
        # imageio.imwrite(fld / 'template.tif', result)
        imageio.imwrite(fld / 'template.tif', results_rot)

    napari.run()


if __name__ == '__main__':
    main()
