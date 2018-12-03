# -*- coding: utf-8 -*-

import sys
import numpy as np

class MyExcept(Exception):
    def __int__(self, msg):
        super().__init__(self)
        self.msg = msg

    def __str__(self):
        return self.msg

def image2vector(filepath):
    f = open(filepath)
    data = np.zeros((0, 32))
    while True:
        line = f.readline()
        if not line:
            break;
        line = line.strip()
        if len(line) != 32:
            raise 

def main(argv=None):
    if not argv:
        argv = sys.argv


if __name__ == '__main__':
    sys.exit(main())