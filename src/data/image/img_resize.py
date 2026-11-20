import numpy as np

from skimage.transform import resize


def image_resize(img, height=None, width=None):
    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale*img.shape[1])

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale*img.shape[0])

    img = resize(image=img, output_shape=(height, width)).astype(np.float32)

    return img


def centered_img(word_img, tsize, border_value=None, centering=(.5, .5)):

    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w // 2, word_img.shape[1] - (diff_w - diff_w // 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)
    word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw), 'constant', constant_values=border_value)

    return word_img


def random_position_img(word_img, tsize, border_value=None):
    height, width = tsize
    h, w = word_img.shape[:2]

    if border_value is None:
        border_value = np.median(word_img)

    # Crop if image is larger than target
    if h > height:
        max_y = h - height
        y = np.random.randint(0, max_y + 1)
        word_img = word_img[y:y + height, :]
        h = height

    if w > width:
        max_x = w - width
        x = np.random.randint(0, max_x + 1)
        word_img = word_img[:, x:x + width]
        w = width

    # Available padding
    diff_h = height - h
    diff_w = width - w

    # Randomly choose image position
    pad_top = np.random.randint(0, diff_h + 1)
    pad_left = np.random.randint(0, diff_w + 1)

    pad_bottom = diff_h - pad_top
    pad_right = diff_w - pad_left

    word_img = np.pad(
        word_img,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=border_value
    )

    return word_img

def top_left_img(word_img, tsize, border_value=None):
    height, width = tsize
    h, w = word_img.shape[:2]

    if border_value is None:
        border_value = np.median(word_img)

    # Crop if larger than target
    word_img = word_img[:height, :width]

    h, w = word_img.shape[:2]

    # All padding goes to bottom and right
    pad_bottom = height - h
    pad_right = width - w

    word_img = np.pad(
        word_img,
        ((0, pad_bottom), (0, pad_right)),
        mode='constant',
        constant_values=border_value
    )

    return word_img