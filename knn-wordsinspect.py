# -*- coding: utf-8 -*-

import sys
import os
import time
from os import listdir
import numpy as np
from multiprocessing import Pool, cpu_count, Process, Manager

def my_wrap(func):
    def inner(*args, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        print('{} cost {} seconds'.format(func.__name__, time.perf_counter() - start))
        return ret
    return inner

class MyExcept(Exception):
    def __int__(self, msg):
        super().__init__(self)
        self.msg = msg

    def __str__(self):
        return self.msg

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


def image2vector(filepath):
    f = open(filepath)
    data = np.zeros((1, 1024))
    index = 0
    for line in f.readlines():
        if not line:
            break
        line = line.strip()
        if len(line) != 32:
            raise MyExcept('数字文件错误:长度不为32')
        index += 1
        if index > 32:
            raise MyExcept('数字文件错误:超过32行')
        # l_array = np.array([[i for i in line]])
        # print('l_array.shape={}, data.shape={}'.format(l_array.shape, data.shape))
        data[0][(index-1)*32:index*32] = [float(i) for i in line]

    return data

@my_wrap
def get_words_dataset(dir):
    words_labels = []
    filelist = listdir(dir)
    training_dataset = np.zeros((0, 1024))
    for file in filelist:
        words_labels.append(int(file.split('.')[0].split('_')[0]))
        words_dataset = image2vector(os.path.join(dir, file))
        training_dataset = np.r_[training_dataset, words_dataset]

    return training_dataset, words_labels

@my_wrap
def main(argv=None):
    if not argv:
        argv = sys.argv

    try:
        training_dataset, training_labels = get_words_dataset('trainingDigits')
        test_dataset, test_labels = get_words_dataset('testDigits')
        # print('training_dataset.shape={}'.format(training_dataset.shape))
        # print('test_dataset.shape={}'.format(test_dataset.shape))
        words_count = test_dataset.shape[0]
        error = 0
        for idx in range(words_count):
            ret = classfy(test_dataset[idx][:], training_dataset, training_labels, 3)
            if ret != test_labels[idx]:
                error += 1

        print('error rate is {}%'.format(round(error / words_count * 100, 2)))
        print('error rate is {}% ({} {})'.format(round(error / words_count * 100, 2), error, words_count))

    except MyExcept as e:
        print(e)
        return 2

    return 0


if __name__ == '__main__':
    sys.exit(main())