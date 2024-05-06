import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
import os
import re
import shutil
import time

def get_image_list(path):
    img_list = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        if ext == '.jpg' and re.search('res_[0-9]+$', name):
            img_list.append(file)
    return img_list

def get_position_list(path):
    pos_list = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        if ext == '.txt' and re.search('res_[0-9]+$', name):
            pos_list.append(file)
    return pos_list

def main(cfg):

    if os.path.isdir(cfg.final_results_path):
        shutil.rmtree(cfg.final_results_path)
        os.makedirs(cfg.final_results_path, exist_ok=True)
    else:
        os.makedirs(cfg.final_results_path, exist_ok=True)

    current_time = time.localtime()
    file_name = cfg.final_results_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", current_time) + '.txt'
    with open(file_name, "a") as f:

        # Load model and image transforms
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

        data_path = cfg.data_path

        img_list = get_image_list(data_path)
        pos_list = get_position_list(data_path)

        possible_numbers = cfg.possible_numbers
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

                maxX = max([pos[0],pos[2],pos[4],pos[6]])
                minX = min([pos[0],pos[2],pos[4],pos[6]])
                maxY = max([pos[1],pos[3],pos[5],pos[7]])
                minY = min([pos[1],pos[3],pos[5],pos[7]])

                crop_img = img.crop((minX, minY, maxX, maxY))
                # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
                img = img_transform(crop_img).unsqueeze(0)

                logits = parseq(img)
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
