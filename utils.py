import re
import imageio
import numpy as np

def main():
    pass

def available_wells(fld, return_sorted=True):
    pattern = re.compile(r'.*_([A-H][0-1]\d)_.*')
    well_names = set(pattern.match(e.name).group(1) for e in fld.rglob("*") if pattern.match(e.name))
    if return_sorted:
        return sorted(well_names)
    return well_names


def load_well(fld, well_name):
    names = []
    imgs = []
    for fn in fld.rglob(f"*_{well_name}_*.tif"):
        names.append(fn)
        imgs.append(imageio.imread(fn))
    return names, imgs


def stitch_arrays(array_list, ny=5, nx=4):
    dy, dx = array_list[0].shape

    out = np.zeros((ny * dy, nx * dx), dtype=array_list[0].dtype)
    for i, arr in enumerate(array_list):
        x = i % nx
        y = i // nx
        out[y * dy:y * dy + dy, x * dx:x * dx + dx] = arr
    return out


def unstitch_arrays(stitched, ny=5, nx=4):
    dy = stitched.shape[0] // ny
    dx = stitched.shape[1] // nx

    out = []
    for i in range(ny * nx):
        x = i % nx
        y = i // nx
        out.append(stitched[y * dy:y * dy + dy, x * dx:x * dx + dx].copy())
    return out

if __name__ == '__main__':
    main()
