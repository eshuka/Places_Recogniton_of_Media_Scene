def get_top_k_result(similar_list, k):
    result = (sorted(similar_list, key=lambda l: l[1], reverse=True))
    return result[0:k + 1]


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
