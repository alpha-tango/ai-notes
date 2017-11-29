from __future__ import print_function
import tensorflow as tf

'''
In the tutorial:
The central unit of data in TensorFlow is the tensor. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's rank is its number of dimensions.
From wikipedia:
In mathematics, tensors are geometric objects that describe linear relations between geometric vectors, scalars, and other tensors. Elementary examples of such relations include the dot product, the cross product, and linear maps. Geometric vectors, often used in physics and engineering applications, and scalars themselves are also tensors.
'''

'''
A computational graph is a series of TensorFlow operations arranged into a graph of nodes. Let's build a simple computational graph. Each node takes zero or more tensors as inputs and produces a tensor as an output.
'''

'''
One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it stores internally.
'''

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

'''
A session encapsulates the control and state of the TensorFlow runtime.
You can run a computational graph within a session, at which point everything
within the computational graph gets evaluated:
'''

sess = tf.Session()
print(sess.run([node1, node2]))

'''
Another type of node is an operation. You can get fancier by combining
constant nodes with operation nodes.'''

node3 = tf.add(node1, node2)
print("node3: ", node3) # expect this to print the object and shape gobbledygook
print("sess.run(node3):", sess.run(node3)) # expect this to actually evaluate and result in 7

'''
Here they explain that TensorBoards are basically just visualizations
of the Flow / ComputationalGraph / DAG. They look exactly like Airflow or 
KNIME.
'''
