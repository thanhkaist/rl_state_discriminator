import imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

WIDTH = 128
HEIGHT = 128
def test_frame_zero():
    v = imageio.read('demo_video.mp4')
    # get frame 0
    d = v.get_data(0)
    plt.imshow(d); plt.show()

# get episode information
def get_eps_info():
    d = np.load('demonstrations.npz')
    ep_info = d['ep_lens']
    print('Total trajectory {}'.format(ep_info.shape[0]))
    return ep_info

def read_video(filepath):
    v = imageio.read(filepath)
    return v

def get_frame(video, frame = 0,size = (128,128)):
    img = Image.fromarray(video.get_data(frame))
    re_im = img.resize(size[0], size[1])
    im_np = np.array(re_im)