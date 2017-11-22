import ctypes
import os
import matplotlib.pyplot as plt
import sysv_ipc
from struct import unpack

from PIL import Image
import numpy as np
import cv2

def get_screen_shm(shared_memory, w, h):
    fmt = w * h * 3 * 'B'
    memory_value = shared_memory.read()
    data = unpack(fmt, memory_value)
    img_data = np.array(data, dtype=np.uint8).reshape((h, w, 3))
    return img_data


if __name__ == '__main__':
    # im = get_screen(0, 0, 640, 480, ":45")
    # cv2.imwrite('/data/2.png', im)
    shared_memory = sysv_ipc.SharedMemory(3252)
    im = get_screen_shm(shared_memory, 205, 230)
    plt.imshow(im), plt.show()
    # plt.imshow(im), plt.show()
    cv2.imwrite('/data/1.png', im)
