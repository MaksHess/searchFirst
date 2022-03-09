import re
import imageio
import numpy as np
import xml.etree.ElementTree as ET

XML_NAMESPACES = {'bts': 'http://www.yokogawa.co.jp/BTS/BTSSchema/1.0'}
ROW_MAP = {k: v for k, v in zip('ABCDEFGHIJKLMNOP', range(1, 16 + 1))}

for ns, uri in XML_NAMESPACES.items():
    ET.register_namespace(ns, uri)


def main():
    pass


def get_xml_mes_template_from_file(secondpass_xml_file):
    tree = ET.parse(secondpass_xml_file)
    root = tree.getroot()
    for timeline in root.find('bts:Timelapse', XML_NAMESPACES).findall('bts:Timeline', XML_NAMESPACES):
        root.find('bts:Timelapse', XML_NAMESPACES).remove(timeline)
    return tree


def get_xml_timeline_template(name):
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


def get_pixel_scale(measurement_detail_xml_file):
    tag = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}MeasurementChannel'
    attribute_x = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}HorizontalPixelDimension'
    attribute_y = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}VerticalPixelDimension'

    root = ET.parse(measurement_detail_xml_file).getroot()
    y = float(root.find(tag).attrib[attribute_y])
    x = float(root.find(tag).attrib[attribute_x])
    return y, x


def get_xml_action_list_from_file(secondpass_xml_file):
    actionlist_tag = 'bts:ActionList'

    root = ET.parse(secondpass_xml_file).getroot()
    action_list = root.find('.//' + actionlist_tag, XML_NAMESPACES)
    return action_list


def get_xml_point(x, y):
    point_tag = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}Point'

    point_attrib_x = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}X'
    point_attrib_y = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}Y'

    return ET.Element(point_tag, attrib={point_attrib_x: str(x), point_attrib_y: str(y)})


def well_name_to_row_column(well_name):
    row = ROW_MAP[well_name[0]]
    column = int(well_name[1:])
    return row, column


def get_xml_targetwell(well_name):
    row, column = well_name_to_row_column(well_name)

    targetwell_tag = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}TargetWell'

    targetwell_attrib_column = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}Column'
    targetwell_attrib_row = '{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}Row'

    well_element = ET.Element(targetwell_tag,
                              attrib={targetwell_attrib_column: str(column), targetwell_attrib_row: str(row)})
    well_element.text = 'true'

    return well_element


def replace_namespace_tag(fn, namespace_map=None):
    if namespace_map is None:
        namespace_map = {'ns0': 'bts'}

    with open(fn, 'r') as f:
        data = f.read()

    for ns_in, ns_out in namespace_map.items():
        data = data.replace(f'{ns_in}:', f'{ns_out}:').replace(f'xmlns:{ns_in}', f'xmlns:{ns_out}')

    with open(fn, 'w') as f:
        f.write(data)


def available_wells(fld, return_sorted=True):
    pattern = re.compile(r'.*_([A-H][0-1]\d)_.*')
    well_names = set(pattern.match(e.name).group(1) for e in fld.rglob("*") if pattern.match(e.name))
    if return_sorted:
        return sorted(well_names)
    return well_names


def stitched_wells(fld, return_sorted=True):
    pattern = re.compile(r'.*([A-H][0-1]\d).png')
    well_names = set(pattern.match(e.name).group(1) for e in fld.rglob("*") if pattern.match(e.name))
    if return_sorted:
        return sorted(well_names)
    return well_names


def load_well(fld, well_name, channel='C02'):
    names = []
    imgs = []
    for fn in sorted(fld.rglob(f"*_{well_name}_*{channel}.tif")):
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
