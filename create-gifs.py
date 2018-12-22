import os
import argparse
import imageio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()

    images = []
    filenames = [fn for fn in os.listdir(args.dir) if fn.endswith(".jpg")]
    for filename in sorted(os.listdir(args.dir), key=lambda x: int(x.strip(".jpg"))):
        img = imageio.imread(os.path.join(args.dir, filename))
        images.append(img)
    imageio.mimsave(args.output, images, duration=0.2)
