# -*- coding: utf-8 -*-

import sys
import os
import time
from os import listdir
import numpy as np
from multiprocessing import Pool, cpu_count, Process, Manager
import asyncio
import aiofiles
from aiofile import AIOFile, LineReader, Writer

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

def classfy_process(td_item, dataset, lableset, k, expect, d, file):
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
    if sorted_classfy_dict[0] != expect:
        d['error'] += 1
        # print('error file={},ret={}, expect={}'.format(file, sorted_classfy_dict[0], expect))
    return

# 此异步io函数与同步io函数相比新能优化并不大，原因
# 是由于文件内容较少，文件数量过多，该异步任务执行
# 的总次数较多，可以考虑将多个文件合并为一个大文件
# 用以减少异步io的次数来提升整体的性能，当文件变大
# 后可以同样考虑使用进程池来进行性能优化（多进程+协程）
async def image2vector_async(filepath, d, label):
    async with aiofiles.open(filepath) as f:
        contents = await f.read()
        lines = contents.split('\n')
        data = np.zeros((1, 1024))
        index = 0
        for line in lines:
            if not line:
                break
            line = line.strip()
            # if filepath == 'trainingDigits\\0_0.txt':
            #     print('line={}'.format(line))
            if len(line) != 32:
                raise MyExcept('数字文件错误:长度不为32,content={}, line={}, file={}'.format(line, index, filepath))
            index += 1
            if index > 32:
                raise MyExcept('数字文件错误:超过32行,content={}'.format(line))
            # l_array = np.array([[i for i in line]])
            # print('l_array.shape={}, data.shape={}'.format(l_array.shape, data.shape))
            data[0][(index-1)*32:index*32] = [float(i) for i in line]

        d[filepath] = {}
        d[filepath]['dataset'] = data
        d[filepath]['label'] = label

@my_wrap
def get_words_dataset_async(dir, **kwargs):
    d = {}
    filelist = listdir(dir)
    tasks = []
    for file in filelist:
        label = int(file.split('.')[0].split('_')[0])
        #image2vector_async 函数为异步读取文件的函数
        func = image2vector_async(os.path.join(dir, file), d, label)
        tasks.append(func)

    if 'loop' not in kwargs:
        loop = asyncio.get_event_loop()
    else:
        loop = kwargs['loop']

    # 将任务加入事件循环处理
    loop.run_until_complete(asyncio.wait(tasks))

    training_dataset = np.zeros((0, 1024))
    words_labels = []
    words_file = []

    # 将字典中的数据生成用于科学计算的矩阵
    for k in d:
        words_dataset = d[k]['dataset']
        words_labels.append(d[k]['label'])
        words_file.append(k)
        training_dataset = np.r_[training_dataset, words_dataset]

    return training_dataset, words_labels, words_file

def task(start, end, test_dataset, training_dataset, training_labels, k, test_labels, d, files):
    for idx in range(start, end):
        classfy_process(test_dataset[idx][:], training_dataset, training_labels, k, test_labels[idx], d, files[idx])

@my_wrap
def main_process(argv=None):
    if not argv:
        argv = sys.argv

    try:
        loop = asyncio.get_event_loop()
        training_dataset, training_labels, training_files = get_words_dataset_async('trainingDigits', loop=loop)
        test_dataset, test_labels, test_files = get_words_dataset_async('testDigits', loop=loop)
        loop.close()
        pool = Pool(processes=cpu_count()*4)
        words_count = test_dataset.shape[0]
        d = Manager().dict()
        d['error'] = 0
        chunk_size = 110
        blocks = words_count//chunk_size + 1 # 最后一块可能少于100
        for idx in range(blocks):
            start = idx * chunk_size
            end = min((idx + 1) * chunk_size, words_count)
            pool.apply_async(task, args=(start, end, test_dataset, training_dataset, training_labels, 3, test_labels, d, test_files))
            # pool.apply_async(classfy_process, args=(test_dataset[idx][:], training_dataset, training_labels, 3, test_labels[idx],d))

        #关闭进程池，不再向池中加入新的进程
        pool.close()
        #等待子进程退出
        pool.join()

        print('error rate is {}% ({}/{})'.format(round(d['error'] / words_count * 100, 2),d['error'], words_count))

    except MyExcept as e:
        print(e)
        return 2

    return 0

if __name__ == '__main__':
    sys.exit(main_process())