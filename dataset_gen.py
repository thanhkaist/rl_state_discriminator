import imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import pdb

WIDTH = 128
HEIGHT = 128
CHANNEL = 3
DIFF = 5

def unnormalize_img(img):
    img = img*(255.0/2)
    img = img + (255.0/2)
    img = np.uint8(img)
    return img

def test_frame_zero():
    v = imageio.read('demo_video.mp4')
    # get frame 0
    d = v.get_data(0)
    plt.imshow(d);
    plt.show()

def normalize_img(img, crop = False):
    if crop:
        img = img[20:-20,20:-20,:]
    obs = Image.fromarray(img,'RGB')
    obs = obs.resize((WIDTH,HEIGHT),Image.ANTIALIAS)
    obs = np.array(obs,dtype=np.float32)
    obs = obs - 255.0 / 2
    obs = obs / (255.0 / 2)
    return obs

# get episode information
def get_eps_info(file_path):
    d = np.load(file_path)
    ep_info = d['ep_lens']
    print('Total trajectory {}'.format(ep_info.shape[0]))
    return ep_info, ep_info.shape[0]


def read_video(filepath):
    v = imageio.read(filepath)
    return v


def get_frame(video, frame=0, size=(128, 128),normalize = True):
    img = Image.fromarray(video.get_data(frame))
    re_im = img.resize((size[0], size[1]))
    im_np = np.array(re_im, dtype=np.float32)
    if normalize:
        im_np = im_np - 255.0 / 2
        im_np = im_np / (255.0 / 2 )
    return im_np


def get_trajectory_frame_index(eps_info, total_eps):
    # 0 --- ep_len1 , ep_len1+4 .. ep_len2+ (ep_len1+4), ep_len2+ (ep_len1+4) + 4 ...
    begin = 0
    eps_trace_index = np.zeros((total_eps, 2), dtype=int)
    for i in range(total_eps):
        end = begin + eps_info[i]
        eps_trace_index[i] = [int(begin), int(end)]
        begin = end + 4
    return eps_trace_index


def load_data_from_npz(filepath):
    dataset = np.load(filepath)
    data = dataset['data']
    labels = dataset['labels']
    return data, labels


def visualize_data(data,label,num =4, offset = 0, zoom = False):
    rows  = num
    cols  = 2
    for i in range(rows*cols):
        j = offset + i // 2
        plt.subplot(rows,cols,i+1,xlabel = str(label[j]))
        im = Image.fromarray(unnormalize_img(data[j][i % 2]),'RGB')
        if zoom:
            im = im.resize((WIDTH*2,HEIGHT*2))
        plt.imshow(im)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate data set from video and npz file. Currently, this script "
                                                 "support state-state dataset and state-target dataset")
    parser.add_argument('dataset', type=str, default='state', choices=['state', 'target'],
                        help="Dataset 'state' [state-state] or 'target' [state-target]")
    parser.add_argument('num', type=int, default=1000, help='Number of examples')
    parser.add_argument('--video', '-v', type=str, default='demo_video.mp4', help='video path')
    parser.add_argument('--npz', '-n', type=str, default='demostrations.npz', help='npz path')
    parser.add_argument('--out_file', '-o', type=str, default='outfile.npz', help='output dataset path')
    parser.add_argument('--test','-t',action='store_true',help='Test generated data')
    parser.add_argument('--offset',type = int, default=0, help = 'Offset of test data')
    args = parser.parse_args()

    video_file = args.video
    demo_file = args.npz
    out_file = args.out_file
    num_pair = args.num
    test = args.test
    offset = args.offset

    if test:
        data, label = load_data_from_npz(out_file)
        visualize_data(data,label,4,offset)
        exit(0)

    def state_gen_data():
        v = read_video(video_file)
        ep_info, total_eps = get_eps_info(demo_file)
        eps_trace_index = get_trajectory_frame_index(ep_info, total_eps)
        datas = np.zeros((num_pair, 2, WIDTH, HEIGHT, CHANNEL),dtype=np.float32)  # (:,0) : state (:,1): state/target
        labels = np.zeros((num_pair, 1),dtype = int)

        total = 0
        for i in range(total_eps):
            assert eps_trace_index[i][1] - DIFF >= eps_trace_index[i][0], "Episode is too short compair with DIFF"
            for idx in np.arange(eps_trace_index[i][0], eps_trace_index[i][1] - DIFF + 1):
                state1_id_t = idx
                state2_id_t = idx + 1
                state2_id_w = np.random.choice(np.arange(idx + DIFF, eps_trace_index[i][1] + 1))

                datas[total, 0] = get_frame(v, state1_id_t, (WIDTH, HEIGHT))
                datas[total, 1] = get_frame(v, state2_id_t, (WIDTH, HEIGHT))
                labels[total] = 1
                if False:
                    # plot data
                    plt.subplot(2, 2, 1, xlabel=str(1))
                    plt.imshow(datas[total][0].astype(int))
                    plt.subplot(2, 2, 2, xlabel=str(1))
                    plt.imshow(datas[total][1].astype(int))

                total += 1
                if total >= num_pair:
                    break

                datas[total, 0] = get_frame(v, state1_id_t, (WIDTH, HEIGHT))
                datas[total, 1] = get_frame(v, state2_id_w, (WIDTH, HEIGHT))
                labels[total] = 0
                # plot data
                if False:
                    plt.subplot(2, 2, 3, xlabel=str(0))
                    plt.imshow(datas[total][0].astype(int))
                    plt.subplot(2, 2, 4, xlabel=str(0))
                    plt.imshow(datas[total][1].astype(int))
                    plt.show()
                    pdb.set_trace()

                total += 1
                if total >= num_pair:
                    break

                if total % 100 == 0:
                    print('Total: {}'.format(total))

            if total >= num_pair:
                break

        return datas, labels


    def target_gen_data():
        datas, labels = None, None
        return datas, labels


    if args.dataset == 'state':
        data, labels = state_gen_data()
    elif args.dataset == 'target':
        data, labels = target_gen_data()
    else:
        raise Exception('The dataset is not support: {}'.format(args.dataset))

    print("Saving dataset to: {}.npz".format(out_file))
    np.savez_compressed(out_file, data=data, labels=labels)
    print("Done")
