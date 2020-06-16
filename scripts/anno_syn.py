"""Reads jsons in syn output dir to get all annotations in single text file.
Each line will have image path and bboxes separated by space.
Each bbox is xmin, ymin, xmax, ymax, class
"""

import os
import sys
import json

classes = ['ref', 'fig', 'ref90', 'fig90']

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
            if x['type'] in classes:
                class_ind = classes.index(x['type'])
                coords = [str(c) for c in x['top left'] + x['bot right']]
                items.append(','.join(coords) + ',' + str(class_ind))

        output_lines.append(' '.join(items))

    with open(output_path, 'w') as fp:
        for line in output_lines:
            fp.write(line + '\n')