import numpy as np
from PIL import Image

RESOLUTION = 224
LUM = np.array([0.299, 0.587, 0.114], dtype=np.float32)

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((RESOLUTION, RESOLUTION), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0

def to_grayscale(img):
    return img @ LUM

def compute_fft(gray):
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    phase = np.angle(fft_shift)
    mag_log = np.log(magnitude + 1e-8)
    return mag_log, phase

# Test on a real COCOFake image
path = 'D:/sweet/binary_deepfake_detection/datasets/coco2014/train2014/COCO_train2014_000000000009.jpg'
img = load_image(path)
print('img shape:', img.shape, 'dtype:', img.dtype)
gray = to_grayscale(img)
print('gray shape:', gray.shape)
mag, phase = compute_fft(gray)
print('mag shape:', mag.shape, 'phase shape:', phase.shape)
print('SUCCESS')