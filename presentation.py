FIND_OBJECTS_STRATEGY: {
    'template': find_objects_by_template_matching,
    'multi-template': find_objects_by_multiple_template_matching,
    'threshold': find_objects_by_threshold,
    'manual': find_objects_by_manual_annotation,
    'semiautomatic-threshold': find_objects_by_semiautomatic_annotation,
}
...

def main():
    ...
    objects, non_objects = FIND_OBJECTS_STRATEGY[args.method]
    ...



# Functions need to extract their parameters from the `args` dict.
objects, non_objects = find_objects_by_template_matching(stitched_ds, args)




