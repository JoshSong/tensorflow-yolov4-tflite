"""Reads jsons in syn output dir to get all annotations in single text file.
Each line will have image path and bboxes separated by space.
Each bbox is xmin, ymin, xmax, ymax, class
"""

import os
import sys
import json

this_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = os.path.dirname(this_dir)

type_classes = ['ref', 'fig', 'ref90', 'fig90']
classes = []
with open(os.path.join(up_dir, 'data', 'classes', 'anno.names')) as fp:
    for line in fp.readlines():
        if line.strip():
            classes.append(line.strip())

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_path = sys.argv[2]
    flist = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    output_lines = []
    for f in flist:
        print('{}/{}'.format(len(output_lines), len(flist)))
        items = []
        name = f.split('_info.json')[0]
        img_path = os.path.join(input_dir, name + '.png')
        items.append(img_path)

        with open(os.path.join(input_dir, f)) as fp:
            info = json.load(fp)

        for x in info['drawn texts']:
            if x['type'] in type_classes:
                class_inds = []
                class_inds.append(classes.index(x['type']))

                if x['type'].startswith('ref'):
                    text = x['text']
                else:
                    text = x['text'].split()[-1]
                class_inds.append(classes.index('length{}'.format(len(text))))

                for i in range(len(text)):
                    char = text[i]
                    if char in '0123456789':
                        class_inds.append(classes.index('{}digit{}'.format(char, i)))
                    elif char in classes:
                        class_inds.append(classes.index(char))
                    else:
                        class_inds.append(classes.index(char.upper()))
                coords = [str(c) for c in x['top left'] + x['bot right']]
                items.append(','.join(coords) + ',' + ','.join([str(ind) for ind in class_inds]))

        output_lines.append(' '.join(items))

    with open(output_path, 'w') as fp:
        for line in output_lines:
            fp.write(line + '\n')