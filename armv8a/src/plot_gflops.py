import matplotlib.pyplot as plt
import numpy as np


def solve(filename):
    with open(filename) as f:
        sizes = [40]
        times = [0.0]
        title = f.readline()
        f.readline()
        while True:
            line = f.readline()         
            if line:
                slices = line.split(" ")
                print(slices)
                if len(slices) <= 2:
                    break
                size = int(slices[0])
                time = float(slices[1])
                sizes.append(size)
                times.append(time)
    return title, sizes, times

if __name__ == '__main__':
    plt.xlabel('size')
    plt.ylabel('gflops')
    t1, x1, y1 = solve('output_old.m')
    plt.plot(x1, y1, label=t1)
    t2, x2, y2 = solve('output_new.m')
    plt.plot(x2, y2, label=t2)
    plt.legend()
    plt.show()