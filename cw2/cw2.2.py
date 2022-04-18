import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from csv import writer
import pandas as pd

def xyz_reader():
    with open('LidarDataX.xyz', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for ax, ay, az in reader:
            yield(float(ax), float(ay), float(az))

S = []
x = []
y = []
z = []

for p in xyz_reader():
    # print(p)
    S.append(p)   
    x.append(p[0])
    y.append(p[1])
    z.append(p[2])
      
model_size = 0    
iter_num = 2000;
# =============================================================================
#                                   Krok 1
# =============================================================================

for j in range(iter_num) :
    
    A = S[random.randint(0, len(S) - 1)]
    B = S[random.randint(0, len(S) - 1)]
    C = S[random.randint(0, len(S) - 1)]
    
    Va = []
    for i in range(len(A)):
        Va.append(A[i] - C[i])
    
    Vb = []
    for i in range(len(B)):
        Vb.append(B[i] - C[i])
    
    Ua = np.divide(Va, np.linalg.norm(Va))
    Ub = np.divide(Vb, np.linalg.norm(Vb))
    Uc = np.multiply(Ua, Ub)
    
    #D = -(Uc[0]*C[0] + Uc[1]*C[1] + Uc[2]*C[2])
    D = -np.sum(np.multiply(Uc, C))
    
    
    # =============================================================================
    #                                   Krok 2
    # =============================================================================
    
    
    thresh = 10;
    distance_all_points = (Uc*S + D)/np.linalg.norm(Uc)
    inliers = np.where(np.abs(distance_all_points) <= thresh )
    if (len(inliers[0]) > model_size) and (len(inliers[0]) < len(S)) :
        model_size = len(inliers[0])
        saved_inliers = inliers

inliers = saved_inliers[0]
x_in = []
y_in = []
z_in = []

for g in range(len(inliers)):
    x_in.append(x[inliers[g]])
    y_in.append(y[inliers[g]])
    z_in.append(z[inliers[g]])

plt.figure()
ax = plt.axes(projection='3d')
#ax.scatter3D(x, y, z)
ax.scatter3D(x_in, y_in, z_in, c="r")
plt.title('Points scattering in 3D', fontsize=14)
plt.tight_layout()
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
# =============================================================================
#                                Dalej już się bawiłem
# =============================================================================

# x_in = pd.DataFrame(x_in)
# y_in = pd.DataFrame(y_in)
# z_in = pd.DataFrame(z_in)

# x_in = x_in.T
# y_in = y_in.T
# z_in = z_in.T

# x_ini = []
# y_ini = []
# z_ini = []

# for g in range(len(inliers)):
#     x_ini.append(x_in[g][0])
#     y_ini.append(y_in[g][0])
#     z_ini.append(z_in[g][0])
    
# points = zip(x_ini, y_ini, z_ini)
# cloud_points = points
# with open('LidarDataX_in.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
#     csvwriter = writer(csvfile)
#     # csvwriter.writerow('x', 'y', 'z')
#     for p in cloud_points:
#         csvwriter.writerow(p)