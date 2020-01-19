from glob import glob
import tensorflow as tf


def check_validity(image_list, mask_list):
    '''Check Validity of Cityscapes Dataset
    Params:
        image_list -> List of Image files
        maks_list  -> List of Mask files
    '''
    for i in range(len(image_list)):
        assert image_list[i].split('/')[-1].split('_leftImg8bit')[0] == mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]
    for i in range(len(val_image_list)):
        assert val_image_list[i].split('/')[-1].split('_leftImg8bit')[0] == val_mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]
    print('All Right!')


def get_image(image_path, mask=False, flip=0):
    '''Read Image/Mask from file
    Params:
        image_path  -> Image Path
        mask        -> Flag denoting image/mask
        flip        -> Integer flag for horizontal flip
    Code Courtsey: https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0/blob/master/train.py#L35
    '''
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.random_brightness(img, max_delta=50.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.case(
            [(
                tf.greater(flip, 0),
                lambda: tf.image.flip_left_right(img)
            )],
            default=lambda: img
        )
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
                      img_height, img_width]), dtype=tf.uint8)
        img = tf.case(
            [(
                tf.greater(flip, 0),
                lambda: tf.image.flip_left_right(img)
            )],
            default=lambda: img
        )
    return img


def load_data(image_path, mask_path):
    '''Map Function for Dataset
    Params:
        image_path  -> Image Path
        mask_path   -> Mask Path
    '''
    flip = tf.random.uniform(
        shape=[1, ], minval=0,
        maxval=2, dtype=tf.int32
    )[0]
    image = get_image(image_path, flip=flip)
    mask = get_image(mask_path, True, flip)
    return image, mask