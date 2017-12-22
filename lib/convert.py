import cv2
import numpy

from lib.model import autoencoder_A
from lib.model import autoencoder_B
from lib.model import encoder, decoder_A, decoder_B

encoder  .load_weights( "models/encoder.h5"   )
decoder_A.load_weights( "models/decoder_A.h5" )
decoder_B.load_weights( "models/decoder_B.h5" )

autoencoder = autoencoder_B

def convert_one_image( image, mat ):
    size = 64
    face = cv2.warpAffine( image, mat * size, (size,size) )
    face = numpy.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )
    new_image = numpy.copy( image )
    image_size = image.shape[1], image.shape[0]
    cv2.warpAffine( new_face, mat * size, image_size, new_image, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )
    return new_image
