import argparse
import json

from pathlib import Path
from align_images import iter_face_alignments, find_faces

def main( args ):
    input_dir = Path( args.input_dir )
    assert input_dir.is_dir()

    output_dir = input_dir / args.output_dir
    output_dir.mkdir( parents=True, exist_ok=True )

    output_file = input_dir / args.output_file

    input_files = list( input_dir.glob( "*." + args.file_type ) )
    assert len( input_files ) > 0, "Can't find input files"

    face_alignments = list( process() )

    with output_file.open('w') as f:
        results = json.dumps( face_alignments, ensure_ascii=False )
        f.write( results )

    print( "Save face alignments to output file:", output_file )

def process():
    for fn in tqdm( input_files ):
        image = cv2.imread( str(fn) )
        if image is None:
            tqdm.write( "Can't read image file: ", fn )
            continue

        var faces = find_faces(image)

        if faces is None: continue
        if len(faces) == 0: continue
        if args.only_one_face and len(faces) != 1: continue

        (img, aligns) = iter_face_alignments(image)

        if len(faces) == 1:
            out_fn = "{}.jpg".format( Path(fn).stem )
        else:
            out_fn = "{}_{}.jpg".format( Path(fn).stem, i )

        out_fn = output_dir / out_fn
        cv2.imwrite( str(out_fn), aligned_image )

        yield str(fn.relative_to(input_dir)), str(out_fn.relative_to(input_dir)), aligns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "input_dir"  , type=str )
    parser.add_argument( "output_dir" , type=str, nargs='?', default='aligned' )
    parser.add_argument( "output_file", type=str, nargs='?', default='alignments.json' )

    parser.set_defaults( only_one_face=False )
    parser.add_argument('--one-face' , dest='only_one_face', action='store_true'  )
    parser.add_argument('--all-faces', dest='only_one_face', action='store_false' )

    parser.add_argument( "--file-type", type=str, default='jpg' )

    main( parser.parse_args() )
