import os
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa


def dataAugment(SampleImgPath, DestPath, UpsampleSingle):

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        iaa.Sometimes(
            0.1,
            iaa.GaussianBlur(sigma=(0, 0.1))
        ), # Small gaussian blur with random sigma between 0 and 0.5
        iaa.LinearContrast((0.75, 1.5)), # Strengthen or weaken the contrast in each image.
        
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        
        iaa.Multiply((0.8, 1.2), per_channel=0.2), # Make some images brighter and some darker, 20% of all cases
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-10, 10),
            shear=(-8, 8)
        ) # Scale/zoom them, translate/move them, rotate them and shear them.
    ], random_order=True) # apply augmenters in random order
    
    Base = os.path.basename(SampleImgPath)
    Filename, FileExt = os.path.splitext(Base)

    for i in range(UpsampleSingle):
        Img = np.asarray(Image.open(SampleImgPath))
        AugImage = seq(image=Img)
        AugImage = Image.fromarray(AugImage)
        AugImage.save(DestPath + '/' + Filename + 'upsample_{}'.format(i + 1) + FileExt)