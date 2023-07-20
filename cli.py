'''find duplicates to remove

usage:
  kopya -i <dir> [options]

options:
 -h --help           show this screen.
 -i --images=<path>  path to directory with images
 -o --output=<path>  path to directory to store images without
                     duplicates. no output means the program will output the list of
                     files but not actually copy them, so the user will need to copy them
                     manually
 -p --plot           whether to plot duplicates.

'''
import logging
from pathlib import Path
import shutil

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


def copy_originals(*, src: str, dest: str, duplicates: list[str]):
    logger.debug(f'copying originals from {src=} to {dest=} ignoring {duplicates=}')
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    duplicates = set(duplicates)
    copied = 0
    for path in Path(src).iterdir():
        if path.is_file() and path.name not in duplicates:
            logger.debug(f'copying {path} to {dest=}')
            shutil.copy(path, dest)
            copied += 1
    logger.info(f'copied {copied} files from {src=} to {dest=}')
    return copied


def main():
    args = docopt.docopt(__doc__)
    logger.debug(args)
    path = args['--images']
    duplicates = find_duplicates(path=path)
    if output := args.get('--output'):
        copy_originals(src=path, dest=output, duplicates=duplicates)
    else:
        logger.info('no output provided, so you will ll need to manually remove duplicates.')
    if args.get('--plot'):
        plot(path=path)


if __name__ == '__main__':
    main()
