#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import urllib.request as ur
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from time import sleep
url='http://192.168.1.4:8080/shot.jpg'
kk=[[0,0,1,0]]#change this for each gesture assign it 
k0=[]
ii=[]
for i in range(0,50):
    sleep(1)#takes pic every one sec
    #to take images from phone camera and process it.
    imgResp = ur.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    i = cv2.imdecode(imgNp,-1)
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    g=cv2.resize(g,(100,100))
    n=g.reshape(1,10000)
    ii.append(n)
    
    k0.append(kk)    #creating an array labelled according to the image data.
#converting it to numpy array and storing.
image_data=np.array(ii)
label_data=np.array(k0)
np.save(r'C:/Users/Anusha/Desktop/mini project/mini project 2/up',image_data)#change this also for each gesture 
np.save(r'C:/Users/Anusha/Desktop/mini project/mini project 2/up_0',label_data)


# In[17]:


import numpy as np

# importing image and label data as numpy array
op=np.load(r'D:/sherlock/autoencoder/Untitled Folder/anu/open.npy')
opi=np.load(r'D:/sherlock/autoencoder/Untitled Folder/anu/open_0.npy')
close=np.load(r'D:/sherlock/autoencoder/Untitled Folder/anu/close.npy')
closei=np.load(r'D:/sherlock/autoencoder/Untitled Folder/anu/close_0.npy')
up=np.load(r'D:/sherlock/autoencoder/Untitled Folder/anu/up.npy')
upi=np.load(r'D:/sherlock/autoencoder/Untitled Folder/anu/up_0.npy')
do=np.load(r'D:/sherlock/autoencoder/Untitled Folder/anu/do.npy')
doi=np.load(r'D:/sherlock/autoencoder/Untitled Folder/anu/do_0.npy')


op=op.reshape(50,100,100,1)
opi=opi.reshape(50,4)
close=close.reshape(50,100,100,1)
closei=closei.reshape(50,4)
up=up.reshape(50,100,100,1)
upi=upi.reshape(50,4)
do=do.reshape(50,100,100,1)
doi=doi.reshape(50,4)
inp=np.concatenate((op,close,up,do))
out=np.concatenate((opi,closei,upi,doi))
#testing 
import matplotlib.pyplot as plt
k=op[5]
k=k.reshape(100,100)
plt.imshow(k)
plt.show()
k=close[5]
k=k.reshape(100,100)
plt.imshow(k)
plt.show()
k=up[5]
k=k.reshape(100,100)
plt.imshow(k)
plt.show()
k=do[5]
k=k.reshape(100,100)
plt.imshow(k)
plt.show()


# In[4]:


from random import shuffle 
y=[]
for i in range(0,200):
    y.append(i)
shuffle(y)
#print(y)

ind=[]
for i in range(0,200):
    k=[y[i], inp[i], out[i]]
    ind.append(k)
ind.sort()
out1=[]
inp1=[]
for i in range(0,200):
    inp1.append(ind[i][1])
    out1.append(ind[i][2])
inp1=np.array(inp1).astype('float32')/255
out1=np.array(out1)


# In[18]:



from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
net = Sequential()
net.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(100,100,1)))
net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
net.add(Conv2D(64, (5, 5), activation='relu'))
net.add(MaxPooling2D(pool_size=(2, 2)))
net.add(Flatten())
net.add(Dense(1000, activation='relu'))
net.add(Dense(100, activation='relu'))
net.add(Dense(20, activation='relu'))
net.add(Dense(4, activation='softmax'))
net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
net.fit(inp1,out1,epochs=5)


# In[19]:


for i in range(0,10):

    k = op[i]
    k = k.reshape(1,100,100,1)
    e=net.predict(k)
    ee = list(map(int,e[0]))
    a="".join(str(i) for i in ee)
    print(a)
    
    
    
    k = close[i]
    k = k.reshape(1,100,100,1)
    e=net.predict(k)
    ee = list(map(int,e[0]))
    a="".join(str(i) for i in ee)
    print(a)
    
    
    
    k = up[i]
    k = k.reshape(1,100,100,1)
    e=net.predict(k)
    ee = list(map(int,e[0]))
    a="".join(str(i) for i in ee)
    print(a)
    
    
    
    k = do[i]
    k = k.reshape(1,100,100,1)
    e=net.predict(k)
    ee = list(map(int,e[0]))
    a="".join(str(i) for i in ee)
    print(a)


# In[30]:



import urllib.request as ur
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pyfirmata
import time
from time import sleep

port = 'COM4'
board = pyfirmata.Arduino(port)




url='http://192.168.43.1:8080/shot.jpg'

for i in range(0,1000):
    #to take images from phone camera and process it.
   
    imgResp = ur.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    i = cv2.imdecode(imgNp,-1)
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g,(100,100))
    k = g.reshape(1,100,100,1)
    e = net.predict(k)
    print(e)
    
    ee = list(map(int,e[0]))
    a="".join(str(i) for i in ee)
    print(a)
    if(a=="1000"):
        print("open")
        board.digital[13].write(1)
    elif(a=="0100"):
        print("close")
        board.digital[13].write(0)
    elif(a=="0010"):
        print("up")
        board.digital[4].write(1)
    elif(a=="0001"):
        print("down")
        board.digital[4].write(0)
        
    k=k.reshape(100,100)
    plt.imshow(k)
    plt.show()
    
    
    
    
board.exit()


# In[ ]:




