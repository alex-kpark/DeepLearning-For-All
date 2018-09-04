import tensorflow as tf

'''Concept of tensorflow and session'''
#step01 : build the graph
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(3.0, tf.float32)
node3 = tf.add(node1, node2)
#node3 = node1 + node2 is also available

print("node1:", node1, "node2:", node2)
print("node3:", node3)

#step02 : establish the session
sess = tf.Session()

#step03 : run the session, and return the result
print('sess.run(node1, node2):', sess.run([node1, node2]))
print('sess.run(node3):', sess.run(node3))


'''Placeholder : let the node to be filled at certain timing'''
#build the graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

#feed_dict: put value for placeholder (fill the empty status)
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))