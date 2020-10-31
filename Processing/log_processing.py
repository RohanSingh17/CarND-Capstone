file1 = open('/home/student/.ros/log/a13f3d8c-1369-11eb-a6ad-08002707b8e3/rosout.log','r')
file2 = open('/home/student/CarND-Capstone/imgs/Train_Imgs/Train_data.txt','w')
count=1

while True:
	line = file1.readline()

	if not line:
		break

	# if line.split(' ')[2][1:7]=='tl_det':
	sep_line =  line.split(' ')
	if len(sep_line)>2 and sep_line[2][-5:-1]=='data':
		if sep_line[-2]!='image':
			print(sep_line[-2],sep_line[-1].rstrip())
			file2.write(''.join(sep_line[-2:]))

file1.close()
file2.close()
