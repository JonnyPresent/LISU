import argparse
import os


def get_arguments(argv):

    parser = argparse.ArgumentParser(description="framework")

    parser.add_argument('--data_path', type=str, default=r'/mnt/disk2/data/stu010/lj/datasets/LISU/LISU_LLRGBD_real',
                        help='path to your dataset')
    parser.add_argument('--pretrained_lisu_path', type=str, default=r'ckpt/save/LISU_LLRGBD_real_best.pth.tar', help='')
    parser.add_argument('--pretrained_snr_path', type=str, default=r'ckpt/save/indoor_G.pth', help='')
    parser.add_argument('--pretrained_resnet_path', type=str, default=r'ckpt/snr/epoch50.pth', help='')

    parser.add_argument('--save_model_path', type=str, default=r'ckpt/snr', help='path to save trained model')
    parser.add_argument('--visualization_path', type=str, default=r'cat_image/snr')

    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each batch')  # fifo 中需=4r

    parser.add_argument('--num_classes', type=int, default=14, help='num of object classes (with void)')
    parser.add_argument("--lambda-fsm", type=float, default=0.0000001)
    parser.add_argument("--lambda-con", type=float, default=0.0001)



    # parser.add_argument("--modeltrain", type=str)

    return parser.parse_args(argv)