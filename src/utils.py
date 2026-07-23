import cv2
import numpy as np
import random


# ==============================
# 1. Illumination Shading
# ==============================
def add_illumination_shading(image, strength=0.5):
    h, w = image.shape[:2]

    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)

    radius = np.sqrt(xv**2 + yv**2)

    mask = 1 - strength * radius
    mask = np.clip(mask, 0.4, 1.0)
    mask = np.dstack([mask] * 3)

    result = image.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)

# ==============================
# 2. Contrast Reduction
# ==============================
def reduce_contrast(image, alpha=0.8):
    beta = random.uniform(-5, 5)
    result = image.astype(np.float32) * alpha + beta
    return np.clip(result, 0, 255).astype(np.uint8)


# ==============================
# 3. Scattering Haze
# ==============================
def add_scattering_haze(image, strength=1.5):
    h, w = image.shape[:2]

    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)

    radius = np.sqrt(xv**2 + yv**2)

    t = np.exp(-strength * radius)
    t = np.clip(t, 0.4, 1.0)
    t = np.dstack([t] * 3)

    A = np.array([240, 240, 240], dtype=np.float32)

    result = image.astype(np.float32) * t + A * (1 - t)
    return np.clip(result, 0, 255).astype(np.uint8)


# ==============================
# 4. Sensor Noise
# ==============================
def add_sensor_noise(image, sigma=5):

    noise = np.random.normal(0, sigma, image.shape)

    result = image.astype(np.float32) + noise

    return np.clip(result, 0, 255).astype(np.uint8)


# ==============================
# 5. Color Cast
# ==============================
def add_color_cast(image, strength=0.05):
    r_gain = 1 + random.uniform(-strength, strength)
    g_gain = 1 + random.uniform(-strength, strength)
    b_gain = 1 + random.uniform(-strength, strength)

    result = image.astype(np.float32)

    result[:, :, 0] *= b_gain
    result[:, :, 1] *= g_gain
    result[:, :, 2] *= r_gain

    return np.clip(result, 0, 255).astype(np.uint8)


# ==============================
# degrade_image
# ==============================
def degrade_image(image, severity=2):
    degraded = image.astype(np.float32)

    s = severity / 3.0

    if random.random() < 0.9:
        degraded = add_illumination_shading(degraded, strength=0.3 + 0.3 * s)

    if random.random() < 0.8:
        degraded = reduce_contrast(degraded, alpha=0.9 - 0.3 * s)

    if random.random() < 0.5:
        degraded = add_scattering_haze(degraded, strength=1.0 + s)

    if random.random() < 0.7:
        degraded = add_sensor_noise(degraded, sigma=2 + 6 * s)

    if random.random() < 0.4:
        degraded = add_color_cast(degraded, strength=0.05 * s)

    return np.clip(degraded, 0, 255).astype(np.uint8)