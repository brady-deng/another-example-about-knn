from numpy import *
import matplotlib.pyplot as plt


def load_data(filename):
    f = open(filename)
    lines = f.readlines()
    num_l = len(lines)
    data = zeros((num_l, 3))
    label = []
    index = 0
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        data[index, :] = temp[0:3]
        if temp[-1] == 'didntLike':
            label.append(0)
        elif temp[-1] == 'smallDoses':
            label.append(1)
        else:
            label.append(2)
        index += 1
    return data, label


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autonorm(dataset):
    max = dataset.max(0)
    min = dataset.min(0)
    ranges = max - min
    normdataset = (dataset - min) / (max - min)
    return normdataset


def classify0(inx, dataset, label, k):
    temp = inx
    temp2 = temp - dataset
    temp3 = temp2 ** 2
    temp4 = temp3.sum(1)
    temp5 = temp4 ** 0.5
    index = temp5.argsort()
    classcount = {}
    for i in range(k):
        votelabel = label[index[i]]
        classcount[votelabel] = classcount.get(votelabel, 0) + 1
    index_count = sorted(classcount)
    return index_count[0]


def datingtest():
    R = 0.20
    data, label = load_data('datingTestSet.txt')
    normmat = autonorm(data)
    si = normmat.shape[0]
    num_test = int(si * R)
    errorcount = 0
    for i in range(num_test):
        res = classify0(normmat[i, :], normmat[num_test:si, :], label[num_test:si], 5)
        print("the classifier came back with: %d, the real answer is: %d" % (res, label[i]))
        if (res != label[i]): errorcount += 1
    print(" the total error rate is %f" % (errorcount / float(num_test) * 100))


if __name__ == "__main__":
    datingtest()
