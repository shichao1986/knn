# -*- coding: utf-8 -*-

import sys
import os
import time
import multiprocessing
from multiprocessing import Pool, cpu_count, Process, Manager

def sub_process_1():
    for i in range(5):
        print('{}:{}'.format(sub_process_1.__name__, i))
        time.sleep(1)

    return

if __name__ == '__main__':
    p = multiprocessing.Process(target=sub_process_1)
    p.start()

    while True:
        alive = p.is_alive()
        print('p is alive {}'.format(alive))
        time.sleep(1)
        if not alive:
            print('p was dead, over')
            p.join()
            break

    print('over')
    sys.exit(0)