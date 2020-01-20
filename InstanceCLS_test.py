import os
import argparse
import models
from utils.NetPredict import NetPredict
import cv2

squeezenetv2_path = '/home/deep/kk/201912-InstanceCLS/log/squeezenetv2_sch/checkpoint_ep120_57.264957427978516.pth.tar'
shufflenetv2_path = '/home/deep/kk/201912-InstanceCLS/log/shufflenetv2_sch/checkpoint_ep120_57.43589782714844.pth.tar'
mobilenetv2_path = '/home/deep/kk/201912-InstanceCLS/log/mobilenetv2_sch/checkpoint_ep120_62.56410598754883.pth.tar'
densenet121_path = '/home/deep/kk/201912-InstanceCLS/log/densenet121_sch/checkpoint_ep120_65.64102935791016.pth.tar'
resnet50_path = '/home/deep/kk/201912-InstanceCLS/log/resnet50_sch/checkpoint_ep120_65.81196594238281.pth.tar'
vggnet16_path = '/home/deep/kk/201912-InstanceCLS/log/vggnet16_sch/checkpoint_ep120_61.02564239501953.pth.tar'

parser = argparse.ArgumentParser(description='Aircraft_Carrier test')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('--model', type=str, default=resnet50_path, help="model file path")
parser.add_argument('--data', type=str, default='/home/deep/kk/data/FineGrained/Aircraft_Carrier', help="the root path of test images")
args = parser.parse_args()

model_arch = args.arch
model_path = args.model
images_list = args.data

cls_num = 20
Pred_Model = NetPredict(model_arch, cls_num, model_path, use_gpu=True)

test_dataset = []
images_test = os.path.join(args.data, 'test_label.txt')
with open(images_test, 'r', encoding='UTF-8') as f:
    lines_train_test = f.readlines()
    for line in lines_train_test:
        strs = line.split(' ')
        image_path = strs[0]
        label = strs[1].strip()
        image_info = [image_path, int(label)]
        test_dataset.append(image_info)


truth_label = []
pred_label = []

i=0
j=0
for img_path, label in test_dataset:
    (path, img_name) = os.path.split(img_path)
    rank5_val = Pred_Model.predict(img_path)

    # print(rank5_val)
    truth_label.append(label)
    pred_label.append(rank5_val[0])

    # if 'resnet50' not in model_path: # the best model for error
    #     continue
    #
    # if label == rank5_val[0]:
    #     if label == i:
    #         im = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    #         im = cv2.resize(im, (600, 400))
    #         cv2.imwrite("examples_truth/True_{}_{}_{}_{}.jpg".format(model_arch,
    #                                 i, label, rank5_val[0]), im)
    #         i = i + 1
    # else:
    #     # if label == j:
    #         im = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    #         im = cv2.resize(im, (600, 400))
    #         cv2.imwrite("examples_error/Error_{}_{}_{}_{}.jpg".format(model_arch,
    #                                 j, label, rank5_val[0]), im)
    #         j = j + 1
    # if not (label == rank5_val[0]):
    #     print('{0} truth: {1} top1: {2}'.format(
    #         img_name, label, rank5_val[0]))

with open("results/%s_truth_label.txt"%(model_arch), 'w') as f:
    for tl in truth_label:
        f.write('{0}\n'.format(tl))

with open("results/%s_pred_label.txt"%(model_arch), 'w') as f:
    for pl in pred_label:
        f.write('{0}\n'.format(pl))
