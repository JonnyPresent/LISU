import argparse
import os


def get_arguments(argv):
    modelname = 'resnet-snr-fifo-syn'
    parser = argparse.ArgumentParser(description="framework")

    parser.add_argument('--data_path', type=str, default=r'/mnt/disk2/data/stu010/lj/datasets/LISU/LISU_LLRGBD_syn/',
                        help='path to your dataset')
    # parser.add_argument('--pretrained_lisu_path', type=str, default=r'ckpt/save/LISU_LLRGBD_real_best.pth.tar', help='')
    # parser.add_argument('--pretrained_fifo_path', type=str, default=r'ckpt/save/dec-res-fifo_maxscore0.4562_epoch400_266.pth', help='')
    # parser.add_argument('--pretrained_resnet_path', type=str, default=r'ckpt/save/resnet-snr_maxscore0.6332_epoch400_96.pth', help='')
    parser.add_argument('--pretrained_path', type=str, default=r'ckpt/resnet-snr-fifo-syn/resnet-snr-fifo-syn_ms0.5056_epoch50_26.pth', help='')

    parser.add_argument('--modelname', type=str, default=f'{modelname}', help='path to save trained model')
    parser.add_argument('--save_model_path', type=str, default=rf'ckpt/{modelname}', help='path to save trained model')
    parser.add_argument('--visualization_path', type=str, default=rf'cat_image/{modelname}')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=27, help='Start counting epochs from this number')
    parser.add_argument('--batch_size', type=int, default=6, help='Number of images in each batch')  # fifo 中需=4r

    parser.add_argument('--num_classes', type=int, default=14, help='num of object classes (with void)')
    parser.add_argument("--lambda-fsm", type=float, default=0.0000001)
    parser.add_argument("--lambda-con", type=float, default=0.0001)



    # parser.add_argument("--modeltrain", type=str)

    return parser.parse_args(argv)