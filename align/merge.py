import argparse
import json
from pathlib import Path

def main( args ):
    input_dir = Path( args.input_dir )
    assert input_dir.is_dir()

    alignments = input_dir / args.alignments
    with alignments.open() as f:
        alignments = json.load(f)

    output_dir = input_dir / args.output_dir
    #output_dir.mkdir( parents=True, exist_ok=True )

    if args.direction == 'AtoB': autoencoder = autoencoder_B
    if args.direction == 'BtoA': autoencoder = autoencoder_A

    for image_file, face_file, mat in alignments:
        image = cv2.imread( str( input_dir / image_file ) )
        face  = cv2.imread( str( input_dir / face_file  ) )

        if image is None: continue
        if face  is None: continue

        new_image = convert(image, face, mat)

        output_file = output_dir / Path(image_file).name
        cv2.imwrite( str(output_file), new_image )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "input_dir", type=str )
    parser.add_argument( "alignments", type=str, nargs='?', default='alignments.json' )
    parser.add_argument( "output_dir", type=str, nargs='?', default='merged' )
    parser.add_argument( "--direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
    main( parser.parse_args() )

