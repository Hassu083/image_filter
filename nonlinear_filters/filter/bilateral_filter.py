from nonlinear_filters.preprocessing.padding import add_padding
from nonlinear_filters.preprocessing.image import MyImage
from nonlinear_filters.config import RGB
from math import exp
import numpy as np


class BilateralFilter:

    def euc(i,j,k):
        return ((i-k)**2 + (j-k)**2)**0.5
    
    def create_kernal(self, sigma, kernalsize):
        kernal = np.zeros((kernalsize,kernalsize),np.float32)
        for i in range(kernalsize):
            for j in range(kernalsize):
                kernal[i, j] = (1/(2*np.pi*sigma*sigma))*exp(-(BilateralFilter.euc(i,j,kernalsize//2)/(2*sigma*sigma)))
        return kernal
    
    def create_spatial_kernal(self, sigma, p, q):
        k = np.abs(p-q)
        kernal = (1/(2*np.pi*sigma*sigma))*np.exp(-((k*k)/(2*sigma*sigma)))
        return kernal
    
    def Ws(self, guassian_kernal, spatial_kernal):
        return np.sum(guassian_kernal*spatial_kernal)

    def filter(self,  image:MyImage, kernalsize, sigma_s, sigma_b):
        padding = kernalsize//2
        kernal = self.create_kernal(sigma_s, kernalsize)

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
            resulting_channels[i] = self.conv(resulting_channels[i], channels[i], kernalsize, kernal, sigma_b)
        
        if n == 1:
            return MyImage(data=resulting_channels[0], type="gray")
        r, g, b = resulting_channels
        return MyImage(data=np.dstack((r,g,b)))


    def conv(self, result, channel, kernalsize, kernal, sigma_b):
        n, m = result.shape
        for i in range(n):
            for j in range(m):
                Ip = channel[i, j]
                Iq = channel[i:i+kernalsize, j:j+kernalsize]
                spatial_kernal = self.create_spatial_kernal(sigma_b, Ip, Iq)
                W = self.Ws(kernal, spatial_kernal)
                result[i,j] = int(min(255.0,max(np.sum(Iq*kernal*spatial_kernal)/W,0.0)))
        return result

