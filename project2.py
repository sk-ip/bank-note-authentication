# program to read the input data.
import pickle
import os
import random

data = []
labels = []

f = open('data set whether the bank note is real or not.txt')  # name of the txt file where data is stored.
read_f = f.readline().rstrip("\n")

while read_f:
	get_values = read_f.split(",")
	get_values = [float(i) for i in get_values]
	data.append(get_values[0:-1])
	
	label = int(get_values[-1])
	if label == 0:
		labels.append([1, 0])
	else:
		labels.append([0, 1])
	
	read_f = f.readline().rstrip("\n")
    
f.close()

test_ratio = 0.2

# program to split the data into training and testing data.

data_0 = data[0:762]
data_1 = data[762:]

label_0 = labels[0:762]
label_1 = labels[762:]

test_0 = int(test_ratio * len(data_0))
test_1 = int(test_ratio * len(data_1))

train_0 = len(data_0) - test_0
train_1 = len(data_1) - test_1

train_0_indexes = random.sample(range(0, len(data_0)), train_0)
train_1_indexes = random.sample(range(0, len(data_1)), train_1)

test_0_indexes = [i for i in range(len(data_0)) if not i in train_0_indexes]
test_1_indexes = [i for i in range(len(data_1)) if not i in train_1_indexes]

train_data_0 = [data_0[i] for i in train_0_indexes]
train_label_0 = [label_0[i] for i in train_0_indexes]
train_data_1 = [data_1[i] for i in train_1_indexes]
train_label_1 = [label_1[i] for i in train_1_indexes]

test_data_0 = [data_0[i] for i in test_0_indexes]
test_label_0 = [label_0[i] for i in test_0_indexes]
test_data_1 = [data_1[i] for i in test_1_indexes]
test_label_1 = [label_1[i] for i in test_1_indexes]

train_data = []
train_label = []
test_data = []
test_label = []

def appending(list1, list2):
	temp = []
	for i in list1:
		temp.append(i)
	for i in list2:
		temp.append(i)
	return temp

train_data = appending(train_data_0, train_data_1)
train_label = appending(train_label_0, train_label_1)
test_data = appending(test_data_0, test_data_1)
test_label = appending(test_label_0, test_label_1)


