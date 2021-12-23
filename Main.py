"""Run the project from the main function"""
import argparse
import os

import torch

from Train_test import Solver
from data_loader import get_loader_INF,get_loader
from torch.backends import cudnn
import random
"""main function"""
def main(config):
    global train_loader, valid_loader, test_loader
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net',"DualNorm_Unet","NestedUNet","CE_Net_"]:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    """ Create directories if not exist"""
    # class=3
    config.model_path_3 = os.path.join(config.model_path_3, config.model_type)
    config.result_path_3 = os.path.join(config.result_path_3, config.model_type)
    if not os.path.exists(config.model_path_3):
        os.makedirs(config.model_path_3)
    if not os.path.exists(config.result_path_3):
        os.makedirs(config.result_path_3)

    if not os.path.exists(config.train_path_3):
        os.makedirs(config.train_path_3)
    if not os.path.exists(config.valid_path_3):
        os.makedirs(config.valid_path_3)
    if not os.path.exists(config.test_path_3):
        os.makedirs(config.test_path_3)
    # class=1
    config.result_path = os.path.join(config.result_path, config.model_type)
    config.model_path = os.path.join(config.model_path, config.model_type)
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)





    print(config)
    """import the data"""
    if config.classes == 1:
        train_loader = get_loader(classes=config.classes,
                                  image_path=config.train_path,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  augmentation_prob=config.augmentation_prob)
        valid_loader = get_loader_INF(classes=config.classes,
                                  image_path=config.valid_path,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='valid',
                                  augmentation_prob=0.)
        test_loader = get_loader_INF(classes=config.classes,
                                 image_path=config.test_path,
                                 image_size=config.image_size,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 mode='test',
                                 augmentation_prob=0.)
    if config.classes > 1:
        train_loader = get_loader(classes=config.classes,
                                  image_path=config.train_path_3,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  augmentation_prob=config.augmentation_prob)
        valid_loader = get_loader_INF(classes=config.classes,
                                  image_path=config.valid_path_3,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='valid',
                                  augmentation_prob=0.)
        test_loader = get_loader_INF(classes=config.classes,
                                 image_path=config.test_path_3,
                                 image_size=config.image_size,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 mode='test',
                                 augmentation_prob=0.)


    solver = Solver(config, train_loader, valid_loader, test_loader)




   """Train and sample the images"""
    if config.mode == 'train':
        solver.train()
    """Test and sample the images"""
    elif config.mode == 'test':
        solver.test()

"""config"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=288)
    parser.add_argument('--t', type=int, default=2, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    #classes number
    parser.add_argument('--classes', type=int, default=3)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=66)
    parser.add_argument('--num_workers', type=int, default=0,help="number of workers in dataloader.in windows,set num_workers=0")
    parser.add_argument('--lr', type=float, default=0.0002) #default lr=0.0002
    #parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    #parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='test',help='train/test') #测试时随机洗牌会导致结果有所差别
    parser.add_argument('--model_type', type=str, default='SD-UNet', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/SD-UNet/FCN8/CE_Net_/NestedUNet')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset2/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset2/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset2/test/')
    parser.add_argument('--result_path', type=str, default='./result/')
    #MISC-3classes
    parser.add_argument('--model_path_3', type=str, default='./model-3')
    parser.add_argument('--train_path_3', type=str, default='./dataset_3/train/')
    parser.add_argument('--valid_path_3', type=str, default='./dataset_3/valid/')
    parser.add_argument('--test_path_3', type=str, default='./dataset_3/test/')
    parser.add_argument('--result_path_3', type=str, default='./result_3/')

    #parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()

    main(config)

