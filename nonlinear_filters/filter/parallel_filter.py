from nonlinear_filters.preprocessing.padding import add_padding
from nonlinear_filters.preprocessing.image import MyImage
from concurrent.futures import ThreadPoolExecutor
from nonlinear_filters.config import RGB
from multiprocessing import Process, Manager
import numpy as np
import time

class ParallelMedianFilter:

    def filter(self, image:MyImage, kernalsize):
        padding = kernalsize//2
        
        if image.type != RGB:
            raise Exception("Only rgb images are supported")
        
        channels = list(image.return_channel())
        manager = Manager()
        resulting_channels = manager.list([])
        n = len(channels)
        nn, m = channels[0].shape

        def toManagerList(manager, arr):
            return manager.list([manager.list(row) for row in arr])
        
        for i in range(n):
            resulting_channels.append(toManagerList(manager, np.zeros_like(channels[i])))
            channels[i] = add_padding(channels[i], padding)
        
        p0 = Process(target=self.conv, args=(nn, m, channels[0], kernalsize, resulting_channels[0]))
        p1 = Process(target=self.conv, args=(nn, m, channels[1], kernalsize, resulting_channels[1]))
        p2 = Process(target=self.conv, args=(nn, m, channels[2], kernalsize, resulting_channels[2]))

        p0.start()
        p1.start()
        p2.start()

        p0.join()
        p1.join()
        p2.join()

        
        def toList(arr):
            return np.array([np.array(row) for row in arr])
        
        r, g, b = resulting_channels

        return MyImage(data=np.dstack((toList(r),toList(g), toList(b))))
    

    def conv(self, n, m, channel , kernalsize, result):
        for i in range(n):
            for j in range(m):
                self.median(i,j,kernalsize, channel, result)
    
    def median(self, i, j, kernalsize, channel, result):
        result[i][j] = np.median(channel[i:i+kernalsize, j:j+kernalsize])