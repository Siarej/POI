import numpy as np
import csv
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

def main():
    filepath = 'D:\\Study\\8EiT\\POI\\Cw3\\textures\\wall'
    filename_prefix = 'wall'
    
    
    BINS = np.array(range(0, 257, 16))
    ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    """
    'wall01',0,1
    'wall01',2,3
    'wall01',4,6
    """
    
    with open("features.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(FEATURES)
        for i in range(6):
            for j in range(6):
                filename = f"{filename_prefix}{i}{j}"
                image = io.imread(f"{filepath}\\{filename}.jpg")
                grayscaled_image = color.rgb2gray(image)
                ubyte_image = img_as_ubyte(grayscaled_image)
                indices = np.digitize(ubyte_image, BINS)
                max_value = indices.max() + 1
                matrix = greycomatrix(indices, [1], ANGLES, levels=max_value)
                writer.writerow([greycoprops(matrix, feature).flatten() for feature in FEATURES])

if __name__ == "__main__":
    main()