'''
find duplicates to remove

usage:
  kopya -i <dir> [options]

options:
 -h --help           show this screen.
 -i --images=<path>  path to directory with images
 -p --plot           whether to plot duplicates.
'''
import logging

import docopt
from imagededup.methods import CNN
from imagededup.utils import plot_duplicates

logger = logging.getLogger('kopya')
logging.basicConfig(level=logging.DEBUG)


def plot(path=None):
    logger.debug(f'plotting duplicates in {path=}')
    cnn = CNN()
    embeddings = cnn.encode_images(image_dir=path)
    duplicates = cnn.find_duplicates(encoding_map=embeddings)
    for original, dups in duplicates.items():
        if len(dups) < 3:
            continue
        plot_duplicates(image_dir=path, duplicate_map=duplicates, filename=original)


def find_duplicates(path=None):
    if not path:
        raise ValueError('cannot remove duplicates without a directory of images')
    logger.debug(f'looking for duplicates in {path=}')
    cnn = CNN()
    embeddings = cnn.encode_images(image_dir=path)
    duplicates = cnn.find_duplicates_to_remove(image_dir=path, encoding_map=embeddings)
    logger.info(f'should remove {duplicates=}')
    return duplicates


def main():
    args = docopt.docopt(__doc__)
    logger.debug(args)
    path = args['--images']
    find_duplicates(path=path)
    if args.get('--plot'):
        plot(path=path)


if __name__ == '__main__':
    main()
