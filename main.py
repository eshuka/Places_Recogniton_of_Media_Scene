import torch, os, cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
from accuracy import get_top_k_result, accuracy
import numpy as np

input_drama = raw_input('drama name : ')
result_txt = input_drama + '.txt'

result_file = open(result_txt, 'w')
model_file = 'models/resnet-50.pth.tar'
file_name = 'txt/categories.txt'

model = torch.load(model_file)
model.eval()

centre_crop = trn.Compose([
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][0:])
classes = tuple(classes)

scene_result_count = []

for x in classes:
    scene_result_count.append([x, 0])

image_path = os.path.join('drama_image', input_drama)
result_drama_folder = os.path.join('result_scene', image_path.split('/')[1])
scene_list = np.sort(os.listdir(image_path))

if not os.path.isdir(result_drama_folder):
    os.mkdir(result_drama_folder)

for scene_num in scene_list:
    scene_path = os.path.join(image_path, scene_num)
    image_list = np.sort(os.listdir(scene_path))

    result_scene_folder = os.path.join(result_drama_folder, scene_num)

    if not os.path.isdir(result_scene_folder):
        os.mkdir(result_scene_folder)

    scene_result_count = [0 for i in range(221)]
    label_result = []

    for image in image_list:
        img_name = os.path.join(scene_path, image)

        img = Image.open(img_name)
        cv2_image = cv2.imread(img_name)
        img = img.convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)

        input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        for i in range(0, 5):
            if i == 0:
                scene_result_count[idx[i]] += 1
            cv2.putText(cv2_image, str(classes[idx[i]]) + ' ' + str(probs[i]), (10, 28 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        result_image_name = os.path.join(result_scene_folder, img_name.split('/')[3])
        cv2.imwrite(result_image_name, cv2_image)

    for c, x in enumerate(classes):
        label_result.append([x, scene_result_count[c]])

    result = get_top_k_result(label_result, 8)

    result_file.write('scene number : ' + scene_num + '\n')
    result_file.write('image count : ' + str(len(image_list)) + '\n')
    result_file.write(str(result) + '\n\n')

# accuracy(top_list)




