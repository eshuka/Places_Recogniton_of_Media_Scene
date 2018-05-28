import os
import numpy as np
from detect import detect

input_drama = raw_input('drama name : ')
result_txt = input_drama + '.txt'

result_file = open(result_txt, 'w')
model_file = 'models/resnet-50.pth.tar'
label_txt = 'txt/categories.txt'
image_path = os.path.join('drama_image', input_drama)

result, scene_num, image_list = detect(model_file, label_txt, image_path, result_file)

result_file.write('scene number : ' + scene_num + '\n')
result_file.write('image count : ' + str(len(image_list)) + '\n')
result_file.write(str(result) + '\n\n')