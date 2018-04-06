# importing dependencies.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# reading the data from the csv file.
raw_data = []
raw_labels = []

file = open('dataset.csv')
read_file = file.readline().rstrip("\n")

while read_file:
	values = read_file.split(",")
	values = [float(i) for i in values]
	raw_data.append(values[0:-1])
	
	label = int(values[-1])
	if label == 0:
		raw_labels.append([float(0)])
	else:
		raw_labels.append([float(1)])
	
	read_file = file.readline().rstrip("\n")
    
file.close()

# splitting the data into training and testing data.
# training and testing data.

train_data = raw_data[0:500]
test_data = raw_data[501:]
train_label = raw_labels[0:500]
test_label = raw_labels[501:]

# defining variables and placeholders.
w = tf.Variable(tf.random_uniform([4, 1], -1.0,1.0), tf.float32)
b = tf.Variable(tf.constant(0.5), tf.float32)

x = tf.placeholder(name = "data", dtype = tf.float32, shape = [None, 4])
y = tf.placeholder(name = "label", dtype = tf.float32, shape = [None, 1])

# creating the computation graph.

# logistic model.
z = tf.matmul(x, w) + b
pred = tf.sigmoid(-z) 

# cost function.
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = pred))

#optimizer.
optimizer = tf.train.GradientDescentOptimizer(1e-3)

# training the model or minimizing the cost.
train = optimizer.minimize(cost)

# running/training the model under a session.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_trace = []
    for step in range(10):
        for _ in range(1000):
            _ ,c = sess.run([train, cost], feed_dict = {x:train_data, y:train_label})
            loss_trace.append(c)
        print('step',step,'weights',sess.run(w),'cost',c)
        
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('accuracy:',accuracy.eval({x: test_data, y: test_label}))
    
              

# to visualize the loss with the no of epochs.
plt.plot(loss_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
