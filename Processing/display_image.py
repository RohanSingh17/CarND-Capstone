import cv2
import time
import matplotlib.pyplot as plt

image_path = '/home/student/CarND-Capstone/imgs/Train_Imgs'
for i in range(135):
	img = cv2.imread(image_path+'/Image_'+str(i+1)+'.jpeg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print(i)
	plt.imshow(img)
	plt.show()
	#time.sleep(0.01)
