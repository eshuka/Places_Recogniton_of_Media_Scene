import torch, os, cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import numpy as np
from darknet import load, darknet

def get_top_k_result(label_result, k):
    result = (sorted(label_result, key=lambda l: l[1], reverse=True))
    return result[0 : k]


def accuracy(top5_lists):
    fileGT = 'txt/correct.txt'

    with open(fileGT, 'r') as f:
        lines = f.readlines()
    truthVector = []
    for line in lines:
        items = line.split()
        print items
        truthVector.append(int(items[0]))

    predictionVector = []
    predictionVector_top5 = []
    for line in top5_lists:
        print "line =", line
        predictionVector.append(int(line[0]))
        if len(line) == 5:
            predictionVector_top5 = top5_lists

    n_classes = 221
    confusionMat = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predictionVector, truthVector):
        confusionMat[pred][exp] += 1
    t = sum(sum(l) for l in confusionMat)

    accuracy = sum(confusionMat[i][i] for i in range(len(confusionMat))) * 1.0 / t

    top5error = 'NA'
    if len(predictionVector_top5) == len(truthVector):
        top5error = 0
        for i, curPredict in enumerate(predictionVector_top5):
            curTruth = truthVector[i]
            curHit = [1 for label in curPredict if label == curTruth]
            if len(curHit) == 0:
                top5error = top5error + 1
        top5error = top5error * 1.0 / len(truthVector)

    print ("accuracy:" + str(accuracy))
    print ("top 5 error rate:" + str(top5error))


def detect(model_file, label_txt, image_path, result_file):
    model = torch.load(model_file)
    model.eval()

    centre_crop = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = list()
    with open(label_txt) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][0:])
    classes = tuple(classes)

    scene_result_count = []

    for x in classes:
        scene_result_count.append([x, 0])

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

        result = get_top_k_result(label_result, 5)

        result_file.write('scene number : ' + scene_num + '\n')
        result_file.write('image count : ' + str(len(image_list)) + '\n')
        result_file.write(str(result) + '\n\n')

    #accuracy(top_list)

    #return result, scene_num, len(image_list)


def remove_object_detect(model_file, label_txt, image_path, result_file):
    model = torch.load(model_file)
    net, meta = load.load_darknet()

    model.eval()


    centre_crop = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = list()
    with open(label_txt) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][0:])
    classes = tuple(classes)

    scene_result_count = []

    for x in classes:
        scene_result_count.append([x, 0])

    result_drama_folder = os.path.join('object_remove_scene', image_path.split('/')[1])
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
            img = img.convert('RGB')

            cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            width, height, channel = np.shape(cv2_image)
            image_size = width * height

            img = img.resize((224, 224), Image.ANTIALIAS)

            box_size = darknet.object_detect(net, meta, img_name, image)

            if box_size / image_size <= 0.2:
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

        result = get_top_k_result(label_result, 5)

        result_file.write('scene number : ' + scene_num + '\n')
        result_file.write('image count : ' + str(len(image_list)) + '\n')
        result_file.write('image_name : ' + str(image_list) + '\n')
        result_file.write(str(result) + '\n\n')

    #accuracy(top_list)

    #return result, scene_num, len(image_list)



