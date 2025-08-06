import os
import pathlib
import random

from PIL import Image
import pillow_avif



IMAGE_SAVE_PATH = pathlib.Path('data')
RAW_IMAGE_SAVE_PATH = pathlib.Path('raw-data')


def get_unique_id():
    '''
    Gets unique id for data collection
    '''
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

def change_format_to_jpg():
    '''
    Changes format of image from * to .jpg
    '''
    for raw_image in os.listdir(RAW_IMAGE_SAVE_PATH):
        # Looping every image
        raw_image_path = RAW_IMAGE_SAVE_PATH / raw_image
        try:
            # Open image
            image = Image.open(raw_image_path)

            # Save new image
            record_id = str(next(get_unique_id())) # Identifier for record
            new_name = record_id + '.jpg'
            new_path = IMAGE_SAVE_PATH / new_name
            image.save(new_path)
            print(f'Saved reformatted image {new_path}')

            # Delete old image
            os.remove(raw_image_path)
            print(f'Deleted old image: {raw_image_path}')

        except Exception as ex:
            print(ex)
            continue

    return None


if __name__ == '__main__':
    change_format_to_jpg()