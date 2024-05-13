from PIL import Image
import numpy as np
from matplotlib import pyplot as plt



class MyImage:

    def __init__(self, path:str = None, data = None, type:str = "RGB"):
        self.type = type
        if path:
            self.data = self.read_image(path)
        elif data is not None:
            self.data = data/255.0 if data.max()>1 else data
        else:
            raise Exception("Neither path or data is specified")
    
    def read_image(self, path):
        image = Image.open(path)
        data = np.asarray(image)
        if self.type == "gray":
            return np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])/255.0
        data = data/255.0
        return data
    
    def return_channel(self):
        if self.type == "gray":
            raise Exception("Only one allowed channel")
        return self.data[:,:,0].copy(),self.data[:,:,1].copy(),self.data[:,:,2].copy()

    def show_image(self, title = ""):
        plt.imshow(self.data)
        plt.title(title)
        plt.show()


def showImages(images:list[MyImage], title = ""):
    n = len(images)
    f, axarr = plt.subplots(1,n, figsize=(12, 12))
    axarr = axarr.flatten()
    for img, ax in zip(images, axarr):
        ax.imshow(img.data)
    plt.title(title)
    plt.show()

if __name__=="__main__":
    im = MyImage(path="../images/cameraman.jpg", type="gray")
    im.show_image()