import os
from sklearn import decomposition
from sklearn import metrics
from numpy import *
import matplotlib.pyplot as plt
import cv2
import numpy as np


# covert image to vactor
def img2vector(filename):
    img = cv2.imread(filename,0) #read as gray
    rows,cols = img.shape  #(265,200)
    imgvector = zeros((1,rows*cols))
    imgvector = reshape(img,(1,rows*cols))  #(1,53000)
    return imgvector, rows, cols

def img2vector_cut1(filename):
    img = cv2.imread(filename,0)
    cropped = img[33:232,25:175]   #keep center 3/4
    rows,cols = cropped.shape 
    imgvector = zeros((1,rows*cols))
    imgvector = reshape(cropped,(1,rows*cols))
    return imgvector, rows, cols

def img2vector_cut_mask(filename):
    img = cv2.imread(filename,0)
    cropped = img[132:264,:]   #keep lower 1/2
    rows,cols = cropped.shape 
    imgvector = zeros((1,rows*cols))
    imgvector = reshape(cropped,(1,rows*cols))
    return imgvector, rows, cols

def img2vector_cut_eyes(filename):
    img = cv2.imread(filename,0)
    cropped = img[100:150,25:175]   #keep eyes
    rows,cols = cropped.shape 
    imgvector = zeros((1,rows*cols))
    imgvector = reshape(cropped,(1,rows*cols))
    return imgvector, rows, cols

file = r"C:\Users\zsyan\Box Sync\SZhang\Fall 2021\PSY 394S ML\python_class\pca\processed\AS_A_M.jpg"
v, rows, cols = img2vector(file)
# v, rows, cols = img2vector_cut1(file)
# v, rows, cols = img2vector_cut_mask(file)
# v, rows, cols = img2vector_cut_eyes(file)
# plt.imshow(v.reshape(rows,cols),cmap=plt.cm.bone)
# plt.show()
# exit()

# build numpy.ndarray
path = r"C:\Users\zsyan\Box Sync\SZhang\Fall 2021\PSY 394S ML\python_class\pca\processed"
files = os.listdir(path)

train_face = np.zeros((15*8,rows*cols))  # 15 people, 8 photos/person
train_label_emo = np.zeros((15*8,1))
train_label_mask = np.zeros((15*8,1))
train_label_people = np.zeros((15*8,1))
for i in range(15):
    people_num = i+1
    # print("i=",i)
    for j in range(8):
        k = 8*i+j
        file = files[k]
        # print("j=",j)
        # print(file)
        filename = path + "\\" + file
        img, rows, cols = img2vector(filename)
        # img, rows, cols = img2vector_cut1(filename)
        # img, rows, cols = img2vector_cut_mask(filename)
        # img, rows, cols = img2vector_cut_eyes(filename)
        train_face[k,:] = img  #(120, 53000)
        train_label_people[k,0] = people_num
        emo = file[-7]
        mask = file[-5]
        if emo == "A":
            train_label_emo[k,0] = 0
        if emo == "H":
            train_label_emo[k,0] = 1
        if emo == "N":
            train_label_emo[k,0] = 2
        if emo == "S":
            train_label_emo[k,0] = 3
        if mask == "M": 
            train_label_mask[k,0] = 1
        if mask == "N":
            train_label_mask[k,0] = 0

# print(train_face.shape)  # (120,53000)

# grand mean center
# train_face = train_face - train_face.mean(axis=0)
# person mean center
train_face = train_face - train_face.mean(axis=0)
train_face -= train_face.mean(axis=1).reshape(120,-1)

n_component = 8
pca = decomposition.PCA(n_components=n_component, svd_solver="randomized", whiten=True).fit(train_face)
# average face
# plt.imshow(pca.mean_.reshape(265,200),cmap=plt.cm.bone)
# plt.show()

# eigenvectors
# print(pca.components_.shape)  #(8, 53000)  (8,120)
eigenfaces = pca.components_.reshape((n_component, rows, cols))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(eigenfaces[i,:,:],cmap = plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
plt.show()
# exit()

# eigenvalues
eigenvalues = pca.explained_variance_
# print(eigenvalues)

# transform
train_face_pca = pca.transform(train_face)  #(120, 8) numpy array

# test
data = np.concatenate((train_face_pca,train_label_emo,train_label_mask,train_label_people),axis=1)  #(120,11)

data_emo_a = data[data[:,8]==0,:]  #(30,11)
data_emo_h = data[data[:,8]==1,:]
data_emo_n = data[data[:,8]==2,:]
data_emo_s = data[data[:,8]==3,:]

data_mask_m = data[data[:,9]==1,:]  #(60, 11)
data_mask_n = data[data[:,9]==0,:]

data_people_amr = data[data[:,10]==1,:]  #(8, 11)
data_people_ap = data[data[:,10]==2,:]
data_people_as = data[data[:,10]==3,:]
data_people_dl = data[data[:,10]==4,:]
data_people_eb = data[data[:,10]==5,:]
data_people_id = data[data[:,10]==6,:]
data_people_mm = data[data[:,10]==7,:]
data_people_mvm = data[data[:,10]==8,:]
data_people_od = data[data[:,10]==9,:]
data_people_ro = data[data[:,10]==10,:]
data_people_sv = data[data[:,10]==11,:]
data_people_sz = data[data[:,10]==12,:]
data_people_uk = data[data[:,10]==13,:]
data_people_ww = data[data[:,10]==14,:]
data_people_zm = data[data[:,10]==15,:]


def silhouette(column_num):
    score12 = metrics.silhouette_score(data[:,[0,1]],data[:,column_num])
    score13 = metrics.silhouette_score(data[:,[0,2]],data[:,column_num])
    score14 = metrics.silhouette_score(data[:,[0,3]],data[:,column_num])
    score23 = metrics.silhouette_score(data[:,[1,2]],data[:,column_num])
    score24 = metrics.silhouette_score(data[:,[1,3]],data[:,column_num])
    score34 = metrics.silhouette_score(data[:,[2,3]],data[:,column_num])
    return [score12, score13, score14, score23, score24, score34]


# Subtask 1: Emotion
column_num = 8
print(silhouette(column_num))

la = plt.scatter(data_emo_a[:,0],data_emo_a[:,3],c='g')
lh = plt.scatter(data_emo_h[:,0],data_emo_h[:,3],c='b')
ln = plt.scatter(data_emo_n[:,0],data_emo_n[:,3],c='c')
ls = plt.scatter(data_emo_s[:,0],data_emo_s[:,3],c='k')
plt.legend((la,lh,ln,ls),("Angry","Happy","Normal","Sad"))
plt.xlabel('Eigenvector1')
plt.ylabel('Eigenvector4')
plt.show()


# Subtask 2: Mask
column_num = 9
print(silhouette(column_num))

lm = plt.scatter(data_mask_m[:,0],data_mask_m[:,1],c='g')
ln = plt.scatter(data_mask_n[:,0],data_mask_n[:,1],c='c')
plt.legend((lm,ln),('Mask','No Mask'))
plt.xlabel('Eigenvector1')
plt.ylabel('Eigenvector2')
plt.show()


# Subtask 3: People
column_num = 10
print(silhouette(column_num))

plt.scatter(data[:,0],data[:,3],c=data[:,10],cmap='coolwarm')
plt.xlabel('Eigenvector1')
plt.ylabel('Eigenvector4')
plt.show()