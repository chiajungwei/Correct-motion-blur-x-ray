import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import pandas as pd

def getMatchNum(matches,ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            matchesMask[i]=[1,0]
            matchNum+=1
    return (matchNum,matchesMask)

#path='D:/CodeProjects/AnacondaSamples/HandleImages/'
path=r'C:\Users\USER\Desktop\model\picture_check\Explainability/' #sim
queryPath=path+'compare/' #图库路径 #no
samplePath=path+r'sim_test/NORMAL2-IM-1102-0001-0002.png' #样本图片
comparisonImageList=[] #记录比较结果
output_path = r'C:\Users\USER\Desktop\model\picture_check\Explainability\output'

#创建SIFT特征提取器
sift = cv2.xfeatures2d.SIFT_create() 
#创建FLANN匹配对象
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)

sampleImage=cv2.imread(samplePath,0)
kp1, des1 = sift.detectAndCompute(sampleImage, None) #提取样本图片的特征
img = []
ssim_list = []
for parent,dirnames,filenames in os.walk(queryPath):
    for p in filenames:
        p=queryPath+p
        print(p)
        queryImage=cv2.imread(p,0)
        kp2, des2 = sift.detectAndCompute(queryImage, None) #提取比对图片的特征
        matches=flann.knnMatch(des1,des2,k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
        (matchNum,matchesMask)=getMatchNum(matches,0.9) #通过比率条件，计算出匹配程度
        matchRatio=matchNum*100/len(matches)
        drawParams=dict(matchColor=(0,255,0),
                singlePointColor=(255,0,0),
                matchesMask=matchesMask,
                flags=0)
        img.append(p)
        ssim_list.append(matchRatio)
        comparisonImage=cv2.drawMatchesKnn(sampleImage,kp1,queryImage,kp2,matches,None,**drawParams)
        comparisonImageList.append((comparisonImage,matchRatio)) #记录下结果

    df = pd.DataFrame({'file': img, 'ratio': ssim_list})
    df.to_csv(os.path.join(output_path, 'result.csv'), index = False)

comparisonImageList.sort(key=lambda x:x[1],reverse=True) #按照匹配度排序



count=len(comparisonImageList)
print(count)
column=1
row=math.ceil(count/column)
#绘图显示
figure,ax=plt.subplots(row,column)
for index,(image,ratio) in enumerate(comparisonImageList):
    print(index)
    ax[int(index/column)].set_title('Similiarity %.2f%%' % ratio)
    ax[int(index/column)].imshow(image)# [index%column]

    # ax.set_title('Similiarity %.2f%%' % ratio)
    # ax.imshow(image)
    # ax[int(index/column)].imshow(image)
plt.show()