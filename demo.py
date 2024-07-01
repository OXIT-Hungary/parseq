import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
import os
import re
import shutil
import time
import numpy as np
from matplotlib import cm
import cv2

def get_image_list(path):
    img_list = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        if ext == '.jpg':
            img_list.append(file)
    return img_list

def get_position_list(path):
    pos_list = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        if ext == '.txt':
            pos_list.append(file)
    return pos_list

def main_init():

    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()

    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

    return parseq, img_transform

def main_eval(cfg, parseq, img_transform, img, bboxes, craft_results):

    outputs = []

    for idx, bbox in enumerate(bboxes):

        cropped_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        current_craft_result = craft_results[idx]
        if len(current_craft_result) !=0:

            parameters = cfg.submodules.parseq.parameters

            possible_numbers = parameters.possible_numbers

            max_idx = np.argmax(current_craft_result[0], axis=0)
            min_idx = np.argmin(current_craft_result[0], axis=0)
            max_x, max_y = current_craft_result[0][max_idx]
            min_x, min_y = current_craft_result[0][min_idx]

            maxX = max_x[0]
            minX = min_x[0]
            maxY = max_y[1]
            minY = min_y[1]

            _img = Image.fromarray(cropped_img.astype('uint8'), 'RGB')

            crop_img = _img.crop((minX, minY, maxX, maxY))

            """ cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.rectangle(cropped_img, (int(minX),int(maxY)), (int(maxX),int(minY)), (0, 0, 255) , 2)
            cv2.imshow("test", cropped_img) """

            # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
            _img = img_transform(crop_img).unsqueeze(0)

            logits = parseq(_img)
            logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

            # Greedy decoding
            pred = logits.softmax(-1)
            label, confidence = parseq.tokenizer.decode(pred)

            """ if label[0].isdigit():
                if possible_numbers.count(int(label[0])):
                    print('Decoded label = {}'.format(label[0]))
                else:
                    print('Nope')
            else:
                print('Nope') """

            outputs.append(label[0])

    return outputs


def main(cfg):

    parameters = cfg.submodules.parseq.parameters
    dataloader = cfg.submodules.parseq.dataloader
    datawriter = cfg.submodules.parseq.datawriter

    if not os.path.isdir(datawriter.final_results_path):
        os.makedirs(datawriter.final_results_path, exist_ok=True)

    current_time = time.localtime()
    file_name = datawriter.final_results_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", current_time) + '.txt'
    with open(file_name, "a") as f:

        # Load model and image transforms
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

        data_path = dataloader.data_path

        img_list = get_image_list(data_path)
        pos_list = get_position_list(data_path)

        possible_numbers = parameters.possible_numbers
        img_list_size = len(img_list)
        img_list_with_string = 0
        correct = 0
        incorrect = 0

        
        for i in range(img_list_size):

            img_name = img_list[i].split('.')[0]
            #pos_name = pos_list[i].split('.')[0]

            path = data_path + str(img_name) + '.txt'
            if os.path.getsize(path) > 0:
                img_list_with_string += 1
                img = Image.open(data_path + '/' + img_list[i]).convert('RGB')
                pos = open(path).read()
                pos = pos.replace('\n', ',')
                pos = pos.split(',')

                pos = [ elem for elem in pos if elem != '']
                pos = [int(i) for i in pos]

                number_of_positions = int(len(pos)/8)

                for p in range(number_of_positions):

                    _img = img

                    p = p*8
                    maxX = max([pos[0+p],pos[2+p],pos[4+p],pos[6+p]])
                    minX = min([pos[0+p],pos[2+p],pos[4+p],pos[6+p]])
                    maxY = max([pos[1+p],pos[3+p],pos[5+p],pos[7+p]])
                    minY = min([pos[1+p],pos[3+p],pos[5+p],pos[7+p]])

                    crop_img = _img.crop((minX, minY, maxX, maxY))
                    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
                    _img = img_transform(crop_img).unsqueeze(0)

                    logits = parseq(_img)
                    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

                    # Greedy decoding
                    pred = logits.softmax(-1)
                    label, confidence = parseq.tokenizer.decode(pred)

                    print(img_list[i] + ' - Decoded label = {}'.format(label[0]))
                    print(img_list[i] + ' - Decoded label = {}'.format(label[0]), file=f)


                    if label[0].isdigit():
                        if possible_numbers.count(int(label[0])):
                            correct += 1
                        else:
                            incorrect += 1
                    else:
                        incorrect += 1

        print('Number of pictures: '+ str(img_list_size), file=f)
        print('Number of pictures with characters: '+ str(img_list_with_string), file=f)
        print('Correct label number: ' + str(correct) + " - " + str(int(correct*100/(correct+incorrect))) + "%" , file=f)
        print('Incorrect label number: ' + str(incorrect) + " - " + str(int(incorrect*100/(correct+incorrect))) + "%" , file=f)

if __name__ == '__main__':
    main()
