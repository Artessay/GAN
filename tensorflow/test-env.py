import tensorflow as tf
a = tf.constant(1.0, tf.float32)
b = tf.constant(2.0, tf.float32)
sess = tf.Session()
print(a, b)
print(sess.run(a + b))