import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

XML_NAMESPACES = {'bts': 'http://www.yokogawa.co.jp/BTS/BTSSchema/1.0'}
PREFIX = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}'
ROW_MAP = {k: v for k, v in zip('ABCDEFGHIJKLMNOP', range(1, 16 + 1))}

for ns, uri in XML_NAMESPACES.items():
    ET.register_namespace(ns, uri)


def main():
    pass


def get_xml_mes_template_from_file(secondpass_xml_file: Union[str, Path]) -> ET.ElementTree:
    tree = ET.parse(secondpass_xml_file)
    root = tree.getroot()
    for timeline in root.find('bts:Timelapse', XML_NAMESPACES).findall('bts:Timeline', XML_NAMESPACES):
        root.find('bts:Timelapse', XML_NAMESPACES).remove(timeline)
    return tree


def get_xml_timeline_template(name: str) -> ET.Element:
    timeline_string = f"""
    <bts:Timeline bts:Name="{name}" bts:InitialTime="0" bts:Period="569" bts:Interval="0" bts:ExpectedTime="569" bts:Color="#ff44ee66" bts:OverrideExpectedTime="false" xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0">
        <bts:WellSequence bts:IsSelected="true">
        </bts:WellSequence>
        <bts:PointSequence bts:Method="FixedPosition">
            <bts:FixedPosition bts:IsProportional="false">
            </bts:FixedPosition>
        </bts:PointSequence>
    </bts:Timeline>
    """
    return ET.fromstring(timeline_string)


def estimate_time_for_timeline(timeline: ET.Element, channellist: ET.Element,
                               security_multiplier: float = 1.0) -> int:
    xy_move_time = 5000.0  # time to move from one position to the next [ms]
    z_move_time = 110.0  # time to move one z-step [ms]

    known_actions = [PREFIX + 'ActionAcquire3D']

    n_wells = len(timeline.find('.//bts:WellSequence', XML_NAMESPACES))
    n_positions = len(timeline.find('.//bts:FixedPosition', XML_NAMESPACES))

    time_per_position = 0
    for action in timeline.find('.//bts:ActionList', XML_NAMESPACES):
        if action.tag not in known_actions:
            raise NotImplementedError(
                f'Cannot estimate the time for {action.tag} , only {known_actions} can be estimated.')
        top_distance = float(action.get(PREFIX + 'TopDistance'))
        bottom_distance = float(action.get(PREFIX + 'BottomDistance'))
        slice_interval = float(action.get(PREFIX + 'SliceLength'))
        n_slices = int((top_distance - bottom_distance) / slice_interval)
        exposure_time = max([_lookup_exposure_in_channellist(channellist, ch.text) for ch in action])
        time_per_position += (exposure_time + z_move_time) * n_slices

    total_time_ms = (time_per_position + xy_move_time) * n_positions * n_wells
    return int(math.ceil(total_time_ms * security_multiplier / 1000))


def _lookup_exposure_in_channellist(channellist: ET.Element, channel_name: str) -> float:
    channel = channellist.find(f".//bts:Channel[@{PREFIX}Ch='{channel_name}']", XML_NAMESPACES)
    if channel is None:
        raise ValueError(f"'{channel_name}' is not an available channel.")
    return float(channel.get(PREFIX + 'ExposureTime'))


def get_pixel_scale(measurement_detail_xml_file: Union[str, Path]) -> Tuple[float, float]:
    tag = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}MeasurementChannel'
    attribute_x = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}HorizontalPixelDimension'
    attribute_y = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}VerticalPixelDimension'

    root = ET.parse(measurement_detail_xml_file).getroot()
    y = float(root.find(tag).attrib[attribute_y])
    x = float(root.find(tag).attrib[attribute_x])
    return y, x


def get_xml_action_list_from_file(secondpass_xml_file: Union[str, Path]) -> ET.Element:
    actionlist_tag = 'bts:ActionList'

    root = ET.parse(secondpass_xml_file).getroot()
    action_list = root.find('.//' + actionlist_tag, XML_NAMESPACES)
    return action_list


def get_xml_point(x: float, y: float) -> ET.Element:
    point_tag = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}Point'

    point_attrib_x = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}X'
    point_attrib_y = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}Y'

    return ET.Element(point_tag, attrib={point_attrib_x: str(x), point_attrib_y: str(y)})


def well_name_to_row_column(well_name: str) -> Tuple[int, int]:
    row = ROW_MAP[well_name[0]]
    column = int(well_name[1:])
    return row, column


def get_xml_targetwell(well_name: str) -> ET.Element:
    row, column = well_name_to_row_column(well_name)

    targetwell_tag = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}TargetWell'

    targetwell_attrib_column = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}Column'
    targetwell_attrib_row = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}Row'

    well_element = ET.Element(targetwell_tag,
                              attrib={targetwell_attrib_column: str(column), targetwell_attrib_row: str(row)})
    well_element.text = 'true'

    return well_element


def available_wells(fld: Path, return_sorted: bool = True) -> List:
    pattern = re.compile(r'.*_([A-H][0-1]\d)_.*')
    well_names = set(pattern.match(e.name).group(1) for e in fld.rglob("*") if pattern.match(e.name))
    if return_sorted:
        return sorted(well_names)
    return list(well_names)


def stitched_wells(fld: Path, return_sorted: bool = True) -> List:
    pattern = re.compile(r'.*([A-H][0-1]\d).png')
    well_names = set(pattern.match(e.name).group(1) for e in fld.rglob("*") if pattern.match(e.name))
    if return_sorted:
        return sorted(well_names)
    return list(well_names)


def load_well(fld: Path, well_name: str, channel: str = 'C02'):
    names = []
    imgs = []
    for fn in sorted(fld.rglob(f"*_{well_name}_*{channel}.tif")):
        names.append(fn)
        imgs.append(imageio.imread(fn))
    return names, imgs


def stitch_arrays(array_list: List[np.ndarray], ny: int = 5, nx: int = 4):
    dy, dx = array_list[0].shape

    out = np.zeros((ny * dy, nx * dx), dtype=array_list[0].dtype)
    for i, arr in enumerate(array_list):
        x = i % nx
        y = i // nx
        out[y * dy:y * dy + dy, x * dx:x * dx + dx] = arr
    return out


def unstitch_arrays(stitched: np.ndarray, ny: int = 5, nx: int = 4):
    dy = stitched.shape[0] // ny
    dx = stitched.shape[1] // nx

    out = []
    for i in range(ny * nx):
        x = i % nx
        y = i // nx
        out.append(stitched[y * dy:y * dy + dy, x * dx:x * dx + dx].copy())
    return out


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
