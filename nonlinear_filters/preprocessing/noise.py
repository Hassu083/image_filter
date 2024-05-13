from nonlinear_filters.preprocessing.image import MyImage
import numpy as np

class Noise:

    def add_guassian_noise(self, image:MyImage, std:int):
        noise = np.random.normal(0, std, image.data.shape)
        new_image = image.data.copy() + noise
        new_image = np.clip(new_image, 0, 1)
        return MyImage(data=new_image, type= image.type)
    
    
    def add_salt_pepper_noise(self, image:MyImage, noise_percentage):
        img_size = image.data.size
        noise_size = int(noise_percentage*img_size)
        random_indices = np.random.choice(img_size, noise_size)
        img_noised = image.data.copy()
        noise = np.random.choice([image.data.min(), image.data.max()], noise_size)
        img_noised.flat[random_indices] = noise
        return MyImage(data=img_noised, type= image.type)