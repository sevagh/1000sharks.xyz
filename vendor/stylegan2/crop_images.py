import sys
import argparse
import os
from PIL import Image, ImageOps

def main():
    parser = argparse.ArgumentParser(
        prog="crop_images.py",
        description="1000sharks StyleGAN2 image cropping tool"
    )

    parser.add_argument("outpath", help="Path to dir to save normalized cropped images")
    parser.add_argument("inpaths", help="Path to dirs containing image files", nargs="+")
    parser.add_argument("--dim", type=int, default=512, help="dim = width = height (default: 512)")
    args = parser.parse_args()

    seq = 0

    for p in args.inpaths:
        for image in os.listdir(p):
            img = Image.open(os.path.join(p, image))

            thumbnail = ImageOps.fit(
                img,
                (args.dim, args.dim),
                Image.ANTIALIAS
            )
            thumbnail.save(os.path.join(args.outpath, '{0}.png'.format(seq)))
            seq += 1


    return 0


if __name__ == '__main__':
    sys.exit(main())
