import argparse
import numpy
import cv2

from lib.convert import convert_one_image

from pathlib import Path
from lib.align_images import iter_face_alignments, find_faces

input_dir = Path( "original" )
assert input_dir.is_dir()

output_dir = input_dir / "output"

input_files = list( input_dir.glob( "*.jpg" ) )
assert len( input_files ) > 0, "Can't find input files"

i = 0;
for fn in input_files:
    image = cv2.imread( str(fn) )
    if image is None:
        print( "Can't read image file: {}".format(fn) )
        continue

    faces = find_faces(image)

    if faces is None: continue
    if len(faces) == 0: continue
    # if args.only_one_face and len(faces) != 1: continue

    for face, mat in iter_face_alignments(image, faces): # return a list of modified faces ?
        if face is None: continue

        mat = numpy.array(mat).reshape(2,3)
        new_image = convert_one_image( image, mat )

        output_file = output_dir / Path(fn).name
        cv2.imwrite( str(output_file), new_image )

        i += 1
        cv2.imwrite( str(output_dir / "face{}.jpg".format(str(i))), face )
        print(output_file)
