import napari
from pydantic import BaseModel
from pathlib import Path
from typing import Union
import numpy as np
import re
import imageio
from utils import available_wells
from itertools import cycle
from collections import deque


def main():
    select_cycles = [0, 1, 2, 3, 4, 5]
    fn = r"Z:\hmax\SearchFirst\20210823-FirstPassMeasurement_20210823_113818\AssayPlate_Greiner_#655090\1009602002_Greiner_#655090.wpp"
    data = PlateData.from_file(fn)
    print('data loaded...')
    plate = Plate(data)
    print('plate generated...')
    scl = np.array([1.3, 1.3])

    viewer = napari.Viewer()
    wells = viewer.add_shapes(plate.wells, shape_type='polygon', edge_color='#dababa', face_color='#00000000',
                              scale=(1000, 1000),
                              edge_width=0.3)
    wells.properties['cycle_deque'] = deque(select_cycles[2:])

    def view_current_cycle(viewer, current_cycle, previous_cycle):
        print('setting visibilities...')
        for layer in viewer.layers:
            if layer.name.endswith(f"{current_cycle}"):
                layer.visible = True
            elif layer.name.endswith(f"{previous_cycle}"):
                layer.visible = False

    @viewer.bind_key('n')
    def toggle_next(viewer):
        print('next')
        wells = viewer.layers['Shapes']
        current_cycle = wells.properties['cycle_deque'][0]
        wells.properties['cycle_deque'].rotate(1)
        previous_cycle = wells.properties['cycle_deque'][0]
        view_current_cycle(viewer, current_cycle, previous_cycle)
        wells.properties['cycle_deque'].rotate(-2)

    @viewer.bind_key('p')
    def toggle_next(viewer):
        print('previous')
        wells = viewer.layers['Shapes']
        current_cycle = wells.properties['cycle_deque'][0]
        wells.properties['cycle_deque'].rotate(-1)
        previous_cycle = wells.properties['cycle_deque'][0]
        view_current_cycle(viewer, current_cycle, previous_cycle)
        wells.properties['cycle_deque'].rotate(2)

    fld = Path(r"Z:\hmax\Zebrafish\20210907_96wellFibronectinPolyL-UV#3")
    for well in available_wells(fld):
        color = cycle(['red', 'green', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue'])
        count = 0
        visible = True
        for i, fn_cycle in enumerate(list(fld.rglob(f"{well}.png"))):
            if i in select_cycles:
                if count >= 3:
                    visible = False
                print(fn_cycle.name, i)
                img = imageio.imread(fn_cycle)
                offset = np.array(img.shape) // 2 * scl
                t = np.array(plate.get_grid_point_str(well)) * 1000
                viewer.add_image(img, name=f"{well}_{i}", translate=t - offset, blending='additive', scale=scl,
                                 colormap=next(color), contrast_limits=(100, np.quantile(img, 0.999)), visible=visible)
                count += 1

    napari.run()


PATTERN = re.compile(r'(?<!^)(?=[A-Z])')


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class PlateData(BaseModel):
    version: float
    name: str
    product_id: int
    width: float
    depth: float
    height: float
    columns: int
    rows: int
    column_pitch: float
    row_pitch: float
    left_margin: float
    top_margin: float
    bottom_height: float
    bottom_thickness: float
    well_top_width: float
    well_top_height: float
    well_bottom_width: float
    well_bottom_height: float
    well_volume: float
    well_shape: str
    bottom_material: str

    @classmethod
    def from_file(cls, fn):
        kwargs = PlateData.parse_wpp_file(fn)
        return cls(**kwargs)

    @staticmethod
    def parse_wpp_file(fn: Union[str, Path]):
        kwargs = {}
        with open(fn) as f:
            for line in f.readlines():
                if line.startswith('  bts'):
                    key, val = re.split(r'[\s=]', line[6:])[:2]
                    kwargs[camel_to_snake(key)] = val.strip('"')
        return kwargs


class Plate:
    WELL_TO_NUMBER = {l: i for l, i in zip('ABCDEFGHIJKLMNOP', range(16))}

    def __init__(self, plate_data: PlateData):
        self.plate_data = plate_data

    def get_circle(self, translate=(0, 0), n=128):
        d = np.linspace(0, 2 * np.pi)
        radius = self.plate_data.well_bottom_width / 2
        return np.stack([np.cos(d) * radius + translate[0], np.sin(d) * radius + translate[1]], axis=1)

    def get_grid_point_str(self, well_str):
        row = self.WELL_TO_NUMBER[well_str[0]]
        col = int(well_str[1:]) - 1
        return self.get_grid_point(row, col)

    def get_grid_point(self, row, col):
        row_coord = row * self.plate_data.row_pitch + self.plate_data.left_margin
        col_coord = col * self.plate_data.column_pitch + self.plate_data.top_margin
        return row_coord, col_coord

    @property
    def grid_points(self):
        points = []
        for r in range(self.plate_data.rows):
            for c in range(self.plate_data.columns):
                points.append(self.get_grid_point(r, c))
        return points

    @property
    def wells(self):
        wells = []
        for point in self.grid_points:
            wells.append(self.get_circle(translate=point))
        return wells


if __name__ == '__main__':
    main()
