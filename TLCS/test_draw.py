# from PIL import Image, ImageDraw
# img = Image.open('blank.png')
# draw_img = ImageDraw.Draw(img)
 
# data = ['4','5','87','1','44','83','93','2','54','84','100','64'] 
# x = 0
 
# for i in data:
#     x = x + 30 
#     y = 200 - int(i) 
#     draw_img.line((x,200,x,y), width=10, fill=(255,0,0,255))
         
# img.show()


 
"""
Author: roguesir
Date: 2017/8/30
GitHub: https://roguesir.github.com
Blog: http://blog.csdn.net/roguesir
"""
 
import numpy as np
import matplotlib.pyplot as plt  
similarity_list = [1.4166666666666667, 1.0833333333333333, 0.6666666666666666, 0.9166666666666666, 1.0833333333333333, 1.4166666666666667, 1.5833333333333333, 1.4166666666666667, 1.75]
x1=list(range(0,len(similarity_list)))
y1=similarity_list

plt.plot(x1,y1,'ro-')
plt.title('The average prediction error for each lane')
plt.xlabel('row')
plt.ylabel('column')
plt.legend()
plt.show()