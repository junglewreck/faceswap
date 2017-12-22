import cv2
import numpy

from ...lib.model import autoencoder_A
from ...lib.model import autoencoder_B
from ...lib.model import encoder, decoder_A, decoder_B

#TODO adapt path ?
encoder  .load_weights( "models/encoder.h5"   )
decoder_A.load_weights( "models/decoder_A.h5" )
decoder_B.load_weights( "models/decoder_B.h5" )

autoencoder = autoencoder_B

def convert_one_image( image, mat ):
    size = 64
    image_size = image.shape[1], image.shape[0]

    face = cv2.warpAffine( image, mat * size, (size,size) )
    face = numpy.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )

    face_mask = numpy.zeros(new_face.shape,dtype=image.dtype)
    image_mask = numpy.zeros(image.shape,dtype=image.dtype)
    cv2.circle(face_mask, (size//2,size//2), int(size*0.6), (255,255,255),-1 )

    cv2.warpAffine( face_mask, mat * size, image_size, image_mask, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )

    image_mask = cv2.blur(image_mask,(5,5))

    hl = numpy.argwhere( image_mask.mean(axis=-1)==255).mean(axis=0)
    masky,maskx = hl

    base_image = numpy.copy( image )
    new_image = numpy.copy( image )
    
    cv2.warpAffine( new_face, mat * size, image_size, new_image, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )
    return cv2.seamlessClone(new_image,base_image, image_mask, (int(maskx), int(masky)) , cv2.NORMAL_CLONE   ) 




# def convert_one_image( image, mat ):
#     size = 64
#     face = cv2.warpAffine( image, mat * size, (size,size) )
#     face = numpy.expand_dims( face, 0 )
#     new_face = autoencoder.predict( face / 255.0 )[0]
#     new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )
#     new_image = numpy.copy( image )
#     image_size = image.shape[1], image.shape[0]
#     cv2.warpAffine( new_face, mat * size, image_size, new_image, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )
#     return new_image
