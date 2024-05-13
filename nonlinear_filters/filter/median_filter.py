from nonlinear_filters.preprocessing.padding import add_padding
from nonlinear_filters.preprocessing.image import MyImage
from nonlinear_filters.config import RGB
import numpy as np

class MedianFilter:

    def filter(self, image:MyImage, kernalsize):
        padding = kernalsize//2
        
        if image.type == RGB:
            channels = list(image.return_channel())
        else:
            channels = [image.data.copy()]
        
        resulting_channels = []
        n = len(channels)
        for i in range(n):
            resulting_channels.append(np.zeros_like(channels[i]))
            channels[i] = add_padding(channels[i], padding)
        
        for i in range(n):
            resulting_channels[i] = self.conv(resulting_channels[i], channels[i], kernalsize)
        
        if n == 1:
            return MyImage(data=resulting_channels[0], type="gray")
        r, g, b = resulting_channels
        return MyImage(data=np.dstack((r,g,b)))


    def conv(self, result, channel, kernalsize):
        n, m = result.shape
        for i in range(n):
            for j in range(m):
                result[i,j] = np.median(channel[i:i+kernalsize, j:j+kernalsize])
        return result