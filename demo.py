import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
import os

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img_path = 'data/imgs/ninja_18'
pos_path = 'data/positions/ninja_18'

possible_numbers = [5,8,10,11,14]
right = 0
wrong = 0

img_list = os.listdir(img_path)
pos_list = os.listdir(pos_path)
for i in range(len(img_list)):

    img_name = img_list[i].split('.')[0]
    pos_name = pos_list[i].split('.')[0]

    path = pos_path + '/res_' + str(img_name) + '.txt'
    if os.path.getsize(path) > 0:
        img = Image.open(img_path + '/' + img_list[i]).convert('RGB')
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


        if label[0].isdigit():
            if possible_numbers.count(int(label[0])):
                right += 1
        else:
            wrong += 1


print('Correct labels number: ' + str(right) + " - " + str(int(right*100/(right+wrong))) + "%" )
print('Incorrect labels number: ' + str(wrong) + " - " + str(int(wrong*100/(right+wrong))) + "%" )