import os
import shutil
default = ''

label=['Apple','Baby','Bicycle','Bird','Bus','Camel','Car','Chimpanzee','Clock','Crocodile','Deer','Dog','Elephant','Fish','Flower','Frog','Guitar','Horse','Lamp','Lock','Man','Motorcycle','Mushroom','Orange','Pear','Plane','Rocket','Ship','Table','Television','Tiger','Tractor','Train','Truck','Woman']
print(len(label))
for i in range(35):
	print(i)
	for j in ['train','validation','test']:
		src = default+'ImageNet_Subset/'+label[i]+'/'+j
		src_files = os.listdir(src)
		directory = default+'data/'+j+'/'+label[i]+'/'
		os.makedirs(directory)
		for file_name in src_files:
		    full_file_name = os.path.join(src, file_name)
		    dest = default+'data/'+j+'/'+label[i]+'/'+file_name
		    if (os.path.isfile(full_file_name)):
		        shutil.copy(full_file_name, dest)