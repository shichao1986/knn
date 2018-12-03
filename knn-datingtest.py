# -*- coding: utf-8 -*-

import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 线性归一化函数, 归一化函数要求对数据进行归一化后数据要尽量保持原来的分散特征，若归一化导致数据值分散区间严重变化，则应
# 考虑归一化函数的使用
# Y = （X - MIN）/（MAX - MIN）
def autoNorm_line(dataset):
    # min(0)返回每列的最小值
    col_min = dataset.min(0)
    col_max = dataset.max(0)
    row = dataset.shape[0]
    ranges = col_max - col_min
    min_matrix = np.tile(col_min, (row, 1))
    range_matrix = np.tile(ranges, (row, 1))

    normdata_line = (dataset - min_matrix)/range_matrix

    return normdata_line

def file2matrix(filename):
    f = open(filename)
    data = np.zeros((0,3))
    vector = []
    index = 0
    # for line in iter(lambda :f.readline(1000), b''):
    while True:
        line = f.readline(1000)
        if not line:
            break
        line = line.strip()
        values = line.split('\t')
        # print('{}:{}'.format(index+1, line))
        # index += 1
        # line = line.strip()
        # if len(values) != 4:
        #     break
        new_data = np.zeros((1,3))
        new_data[0][:] = values[0:3]
        data = np.r_[data, new_data]
        vector.append(int(values[3]))

    return data, vector

def classfy(td_item, dataset, lableset, k):
    row = dataset.shape[0]
    td_dataset = np.tile(td_item, (row, 1))
    diffsqure_dataset = (td_dataset - dataset) ** 2
    distance_dataset = diffsqure_dataset.sum(axis=1) ** 0.5
    index_dataset = np.argsort(distance_dataset)
    classfy_dict = {}
    for i in range(k):
        val = lableset[index_dataset[i]]
        classfy_dict[val] = classfy_dict.get(val, 0) + 1

    sorted_classfy_dict = sorted(classfy_dict, key=lambda k:classfy_dict[k], reverse=True)
    return sorted_classfy_dict[0]

def dating_test(test_rate, normdata, datingLabels, k):
    row = normdata.shape[0]
    test_nums = int(round(row * test_rate, 0))

    error = 0
    for i in range(test_nums):
        td = normdata[i][:]
        ret = classfy(td, normdata[test_nums:][:], datingLabels[test_nums:], k)
        if ret != datingLabels[i]:
            error += 1.0

    print('error rate is {}%%'.format(round(error / test_nums * 100, 2)))

def knn_dating_test(test_rate, filepath, k):
    datingDataMat, datingLabels = file2matrix(filepath)

    # fig = plt.figure()
    # ax = fig.add_subplot(221)
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], np.array(datingLabels), np.array(datingLabels))
    # ax = fig.add_subplot(222)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], np.array(datingLabels), np.array(datingLabels))
    # ax = fig.add_subplot(223)
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 2], np.array(datingLabels), np.array(datingLabels))
    # plt.show()

    normdata = autoNorm_line(datingDataMat)
    test_rate = test_rate
    dating_test(test_rate, normdata, datingLabels, k)
    return

def input_dating_test(filepath, k):
    result_list = ['not like', 'like a little', 'like very much']
    fly_miles = float(input('input fly miles per year:'))
    video_game_percentage = float(input('input time percentages of playing video games per day:'))
    pounds_of_icecream = float(input('input the mount of ice-creame eat per week:'))
    datingDataMat, datingLabels = file2matrix(filepath)
    normdata = autoNorm_line(datingDataMat)
    data_min = datingDataMat.min(0)
    data_max = datingDataMat.max(0)
    ranges = data_max - data_min

    td = np.shape([fly_miles, video_game_percentage, pounds_of_icecream])
    td = (td - data_min)/ranges

    ret = classfy(td, normdata, datingLabels, k)

    print('the person you may {}'.format(result_list[ret-1]))

    return

def main(argv=None):
    if not argv:
        argv = sys.argv

    # import pdb;pdb.set_trace()
    knn_dating_test(0.05, 'datingTestSet2.txt', 3)
    input_dating_test('datingTestSet2.txt', 3)

    return 0

if __name__ == '__main__':
    sys.exit(main())
