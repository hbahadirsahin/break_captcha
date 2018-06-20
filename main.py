import numpy as np
import tensorflow as tf

from input_generator import generate_data, pos_to_label

CONV_FILTER_SIZE = [3, 3]
STRIDE_SIZE = [2, 2]
MAX_POOL_FILTER_SIZE = [2, 2]


def get_data_for_train(read_data):
    if read_data:
        print("Reading data")
        train_data = np.load("train.data.npy")
        train_label = np.load("train.label.npy")
        vali_data = np.load("vali.data.npy")
        vali_label = np.load("vali.label.npy")
        test_data = np.load("test.data.npy")
        test_label = np.load("test.label.npy")
    else:
        print("Generating data")
        train_data, train_label = generate_data(n=12000, max_digs=5)
        train_mean = train_data.mean(axis=0)
        train_std  = train_data.std(axis=0)
        train_data = (train_data - train_mean) / (train_std + 0.00001)
        np.save("train.data", train_data)
        np.save("train.label", train_label)
        vali_data, vali_label = generate_data(n=1500,max_digs=5)
        vali_mean = vali_data.mean(axis=0)
        vali_std  = vali_data.std(axis=0)
        vali_data = (vali_data - vali_mean) / (vali_std + 0.00001)
        np.save("vali.data", vali_data)
        np.save("vali.label", vali_label)
        test_data, test_label = generate_data(n=1500,max_digs=5)
        test_mean = test_data.mean(axis=0)
        test_std  = test_data.std(axis=0)
        test_data = (test_data - test_mean) / (test_std + 0.00001)
        np.save("test.data", test_data)
        np.save("test.label", test_label)
    return train_data, train_label, vali_data, vali_label, test_data, test_label


def get_data_for_eval(test_size, max_digit, read_data):
    if read_data:
        print("Reading data")
        test_data = np.load("test.data.npy")
        test_label = np.load("test.label.npy")
    else:
        print("Generating data")
        test_data, test_label = generate_data(n=test_size,max_digs=max_digit)
        test_mean = test_data.mean(axis=0)
        test_std  = test_data.std(axis=0)
        test_data = (test_data - test_mean) / (test_std + 0.00001)
        np.save("test2.data", test_data)
        np.save("test2.label", test_label)
    return test_data, test_label


def weight_variable(shape, wd=None, stddev=0.1 ):
    initial = tf.random_normal(shape, stddev=stddev)
    var = tf.Variable(initial)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)
    return var



def bias_variable(shape, value=0.1):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name="weights")


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')


def max_pool(x, filter_size, stride):
    return tf.nn.max_pool(x, filter_size, strides=stride, padding='SAME')


def norm(x, depth_radius, bias, alpha, beta):
    return tf.nn.lrn(x, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)


def training(model_path, read_data=False):

    train_data, train_label, vali_data, vali_label, test_data, test_label = get_data_for_train(read_data=read_data)

    max_epoch = 50
    batch_size = 50
    train_size = train_data.shape[0]

    x = tf.placeholder(tf.float32, [None, 6144])
    y = tf.placeholder(tf.float32, [None, 50])
    keep_prob = tf.placeholder(tf.float32)

    input = tf.reshape(x, [-1, 48, 128, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 48])
        b_conv1 = bias_variable([48])
        h_conv1 = tf.nn.relu(conv2d(input, W_conv1, stride=[1, 2, 2, 1]) + b_conv1)
        h_pool1 = max_pool(h_conv1, filter_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])
        h_norm1 = norm(h_pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 48, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, stride=[1, 2, 2, 1]) + b_conv2)
        h_pool2 = max_pool(h_conv2, filter_size=[1, 2, 2, 1], stride=[1, 2, 1, 1])
        h_norm2 = norm(h_pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_norm2, W_conv3, stride=[1, 2, 2, 1]) + b_conv3)
        h_norm3 = norm(h_conv3, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)
        h_pool3 = max_pool(h_norm3, filter_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])


    # Flatten before fully connected layer
    with tf.name_scope('flatten'):
        h_pool3_flat = tf.contrib.layers.flatten(h_pool3)


    with tf.name_scope('fully_connected1'):
        W_fc1 = weight_variable([h_pool3_flat.get_shape()[1].value, 2048], stddev=0.04)
        b_fc1 = bias_variable([2048])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # Output layer
    with tf.name_scope('output'):
        W_fc3 = weight_variable([2048, 5*10], stddev=1/2048.0)
        b_fc3 = bias_variable([5*10], 0.0)
        output = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

    # Calculate loss of model
    print("Loss")
    with tf.name_scope("LOSS"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(output, [-1, 50]),
                                                                      labels=tf.reshape(y, [-1, 50])))

    # Calculate accuracy of the model
    print("Accuracy")
    with tf.name_scope('Accuracy'):
        prediction = tf.reshape(output, [-1, 5, 10])
        ground_truth = tf.reshape(y, [-1, 5, 10])
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 2), tf.argmax(ground_truth, 2)), tf.float32))

    # Optimize the loss
    with tf.name_scope('Optimizer'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(1e-3, global_step*batch_size, train_size, 0.95, staircase=True)
        grad = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_batches = int(train_size / batch_size)
        for epoch in range(max_epoch):
            total_train_loss = 0,
            total_train_acc = 0;
            for batch in range(num_batches):
                batch_offset = (batch * batch_size) % train_size
                batch_data = train_data[batch_offset:(batch_offset+batch_size)]
                batch_labels = train_label[batch_offset:(batch_offset+batch_size)]

                train_loss, _, train_pred, train_gt = sess.run([loss, grad, prediction, ground_truth],
                                                               feed_dict={x: batch_data,
                                                                          y: batch_labels,
                                                                          keep_prob: 0.7})

                total_train_loss += train_loss
                total_train_acc += calculate_label_accuracy(train_pred, train_gt, 5)

            vali_loss, vali_pred, vali_gt = sess.run([loss, prediction, ground_truth], feed_dict={x: vali_data,
                                                                                                  y: vali_label,
                                                                                                  keep_prob: 1.0})
            vali_accuracy = calculate_label_accuracy(vali_pred, vali_gt, 5)
            print("Epoch:", (epoch + 1), "- Training loss:", total_train_loss/num_batches, "- Training accuracy:",
                  total_train_acc/num_batches,"- Validation Loss", vali_loss,"- Validation Accuracy:", vali_accuracy)

        test_pred, test_gt = sess.run([prediction, ground_truth], feed_dict={x: test_data, y: test_label, keep_prob: 1.0})
        test_accuracy = calculate_label_accuracy(test_pred, test_gt, 5)
        print("Test accuracy:", test_accuracy)
        saver.save(sess, model_path)


def evaluate(model_path, test_size, max_digit, read_data=False):
    test_data, test_label = get_data_for_eval(test_size, max_digit, read_data=read_data)

    x = tf.placeholder(tf.float32, [None, test_data.shape[1]])
    y = tf.placeholder(tf.float32, [None, test_label.shape[1]])
    keep_prob = tf.placeholder(tf.float32)

    input = tf.reshape(x, [-1, 48, 128, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 48])
        b_conv1 = bias_variable([48])
        h_conv1 = tf.nn.relu(conv2d(input, W_conv1, stride=[1, 2, 2, 1]) + b_conv1)
        h_pool1 = max_pool(h_conv1, filter_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])
        h_norm1 = norm(h_pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 48, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, stride=[1, 2, 2, 1]) + b_conv2)
        h_pool2 = max_pool(h_conv2, filter_size=[1, 2, 2, 1], stride=[1, 2, 1, 1])
        h_norm2 = norm(h_pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_norm2, W_conv3, stride=[1, 2, 2, 1]) + b_conv3)
        h_norm3 = norm(h_conv3, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)
        h_pool3 = max_pool(h_norm3, filter_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])


    # Flatten before fully connected layer
    with tf.name_scope('flatten'):
        h_pool3_flat = tf.contrib.layers.flatten(h_pool3)


    with tf.name_scope('fully_connected1'):
        W_fc1 = weight_variable([h_pool3_flat.get_shape()[1].value, 2048], stddev=0.04)
        b_fc1 = bias_variable([2048])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # Output layer
    with tf.name_scope('output'):
        W_fc3 = weight_variable([2048, 5*10], stddev=1/2048.0)
        b_fc3 = bias_variable([5*10], 0.0)
        output = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

    # Calculate loss of model
    print("Loss")
    with tf.name_scope("LOSS"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(output, [-1, 50]),
                                                                      labels=tf.reshape(y, [-1, 50])))

    # Calculate accuracy of the model
    print("Accuracy")
    with tf.name_scope('Accuracy'):
        prediction = tf.reshape(output, [-1, 5, 10])
        ground_truth = tf.reshape(y, [-1, 5, 10])
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction[:,:max_digit], 2), tf.argmax(ground_truth[:,:max_digit], 2)), tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        pred, gt, test_accuracy = sess.run([prediction, ground_truth, accuracy], feed_dict={x: test_data, y: test_label, keep_prob: 1.0})
        test_accuracy = calculate_label_accuracy(pred, gt, max_digit)
        print("Test accuracy:", test_accuracy, "for", max_digit,"digit-labeled test set")


def col_to_number(input):
    res = np.zeros(input.shape[0])
    dim = input.shape[1]
    for i in range(input.shape[0]):
        if dim == 2:
            res[i] = int(input[i][0] * 10 + input[i][1])
        elif dim == 3:
            res[i] = int(input[i][0] * 100 + input[i][1]* 10+ input[i][2])
        elif dim == 4:
            res[i] = int(input[i][0] * 1000 + input[i][1]* 100+ input[i][2]*10 + input[i][3])
        elif dim == 5:
            res[i] = int(input[i][0] * 10000 + input[i][1]* 1000+ input[i][2]*100 + input[i][3]*10+input[i][4])
    return res


def calculate_label_accuracy(pred, gt, max_digit):
    preds = col_to_number(np.argmax(pred[:, :max_digit], 2))
    gts = col_to_number(np.argmax(gt[:, :max_digit], 2))
    return np.mean(np.equal(preds, gts))


if __name__ == '__main__':
    model_path = "D:/PycharmProjects/CS552Assignment4/five_digit_model"
    #training(model_path, read_data=True)
    evaluate(model_path, test_size=1500, max_digit=5, read_data=False)
