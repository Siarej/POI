# importing Image class from PIL package
from PIL import Image
import os




path = "D:\Study\8EiT\POI\Cw3\textures"

filename = "table"
filetype = ".jpg"

lines = 6
columns = 6


x = path+"\\"+filename
isExist = os.path.exists(x)

if not isExist:
  
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory is created!")

im = Image.open(path+"\\"+filename+filetype)

x1 = 0
y1 = 0

x2 = im.size[0] / lines
y2 = im.size[1] / columns


for l in range(lines):
    for c in range(columns):
        temp = im.crop((x1, y1, x2, y2))
        temp.save(path+"\\"+filename+"\\"+filename+str(l)+str(c)+filetype)
        buf = x2 - x1
        x1 = x2
        x2 = buf + x1
    buf = y2 - y1
    y1 = y2
    y2 = buf + y1
    x1 = 0
    x2 = im.size[0] / lines

writer.writerow([filename, contrast_feature(matrix_coocurrence), 
                 dissimilarity_feature(matrix_coocurrence), 
                 homogeneity_feature(matrix_coocurrence), 
                 energy_feature(matrix_coocurrence), 
                 correlation_feature(matrix_coocurrence), 
                 asm_feature(matrix_coocurrence)])


