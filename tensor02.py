import tensorflow as tf

'''
H(x)=Wx+b
'''

'''
X, Y data for training set
x_train = [1,2,3]
y_train = [1,2,3]
'''

'''
Variable means trainable variable, that tensorflow can change the value
through its learning process
'''

#random_normal([1]) - 1 dimension value
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis definition
hypothesis =x_train*W + b

#cost function definition
#reduce_mean : average function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

'''
Minimization process
'''
#GradientDescentOptimizer :
#learning_rate : 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
#must initialize the variable with global_variables_initializer()
sess.run(tf.global_variables_initializer())

#Fitting the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

'''
Without defining x_train, y_train,

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
...
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train]),
            feed_dict={X:[1,2,3], Y:[1,2,3]})
    
    if step % 20 = 0:
        print(step, cost_val, W_val, b_val)
'''