import numpy as np
import pytest


@pytest.fixture
def flat_art():
    """A plum card on a white page with a honey disc and a cream dot on it."""
    h = w = 256
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[24:232, 24:232] = (0x27, 0x14, 0x20)
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (xx - 128) ** 2 + (yy - 128) ** 2 < 70 ** 2
    img[disc] = (0xE6, 0xA7, 0x25)
    dot = (xx - 108) ** 2 + (yy - 108) ** 2 < 18 ** 2
    img[dot] = (0xF5, 0xD1, 0x86)
    return img
