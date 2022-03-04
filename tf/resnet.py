"""

Implementation of the CIFAR-10 res net from He et al, 2015

"""

import utils

import numpy as np
import tensorflow as tf
from typing import Any, Dict, Tuple


def augment(image: np.array) -> np.array:
    # pad with zeros
    aug_image = np.pad(image, ((4, 4), (4, 4), (0, 0)), "reflect")

    # random crop
    i = np.random.randint(0, 9)
    j = np.random.randint(0, 9)
    aug_image = aug_image[i : i + 32, j : j + 32, :]

    # possibly reflect
    if np.random.randint(0, 2) == 0:
        aug_image = np.fliplr(aug_image)

    return aug_image


def make_conv_and_bn(
    input: Any,
    n_filters: int,
    kernel_sz: int,
    name: str,
    training: bool,
    stride: int = 1,
    dtype: Any = tf.float32,
) -> Any:
    """make a conv op and batch norm."""
    conv = tf.compat.v1.layers.Conv2D(
        n_filters,
        kernel_sz,
        padding="same",
        dtype=dtype,
        strides=stride,
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    )(input)
    return tf.layers.batch_normalization(conv, training=training)


def make_subsample(input: Any, name: str) -> Any:
    """
    Return ops used for projection described in paper.
    
    Ex. (32, 32, 16) -> (16, 16, 32):
    (32, 32, 16) -> avg pooling -> (16, 16, 16) -> proj -> (16, 16, 32)
    """
    with tf.variable_scope(name):
        pool = tf.nn.pool(
            input=input,
            window_shape=(2, 2),
            strides=(2, 2),
            padding="SAME",
            pooling_type="AVG",
        )
        channels = input.shape[-1]
        new_channels = channels * 2
        return tf.linalg.matmul(
            pool,
            tf.get_variable(
                f"proj_{new_channels}_{channels}",
                shape=(channels, new_channels),
                initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            ),
        )


def make_block(
    input: any,
    n_filters: int,
    kernel_sz: int,
    name: str,
    training: bool,
    residual: bool = False,
    subsample: bool = False,
    dtype: Any = tf.float32,
) -> Any:
    """
    Create a residual building block.

    residual: whether the network should  be "residual" or "plain".
    subsample: whether to keep the sampe dimensions or subsample

    """
    with tf.variable_scope(name):
        if subsample:
            stride = 2
        else:
            stride = 1
        c1 = make_conv_and_bn(
            input, n_filters, kernel_sz, name="c1", stride=stride, training=training
        )
        relu = tf.nn.relu(c1, name="relu")
        c2 = make_conv_and_bn(relu, n_filters, kernel_sz, name="c2", training=training)
        if residual:
            if subsample:
                addn = make_subsample(input, name) + c2
            else:
                addn = input + c2
            # bn = tf.layers.batch_normalization(addn, training=training)
            return tf.nn.relu(addn, name="relu2")
        return tf.nn.relu(c2, name="relu2")


def add_block_stack(
    start: int,
    end: int,
    input: Any,
    n_filters: int,
    kernel_sz: int,
    residual: bool,
    name: str,
    training: bool,
    subsample: bool = True,
):
    """Add stack of blocks of the same size."""
    with tf.variable_scope(name):
        blocks = []
        # first block subsamples
        blocks.append(
            make_block(
                input,
                n_filters,
                kernel_sz,
                name=f"block_{start}",
                residual=residual,
                subsample=subsample,
                training=training,
            )
        )
        for i in range(start + 1, end):
            blocks.append(
                make_block(
                    blocks[-1],
                    n_filters=n_filters,
                    kernel_sz=kernel_sz,
                    name=f"block_{i}",
                    residual=residual,
                    training=training,
                )
            )
        return blocks[-1]


def create_graph(name: str, n: int, residual: bool = False, ty: Any = tf.float32):
    """Create all nodes necessary for graph."""
    nodes = {}
    with tf.variable_scope(name):
        nodes["training"] = tf.placeholder(tf.bool, shape=(), name="training")

        nodes["ph_input"] = tf.placeholder(
            tf.uint8, shape=(None, 32, 32, 3), name="input"
        )
        # roughly normalize image without iterating through entire training/test set
        nodes["input_to_float"] = tf.cast(nodes["ph_input"], ty) / (255 / 2) - 1

        nodes["ph_labels"] = tf.placeholder(tf.uint8, shape=(None), name="labels")
        nodes["labels_to_int"] = tf.cast(nodes["ph_labels"], tf.int32)

        nodes["ph_lr"] = tf.placeholder(name="lr", dtype=tf.float32, shape=[])

        nodes["block_0"] = make_conv_and_bn(
            input=nodes["input_to_float"],
            n_filters=16,
            kernel_sz=3,
            name="block_0",
            training=nodes["training"],
        )

        # output: 32, 32, 16
        b1 = add_block_stack(
            start=1,
            end=n + 1,
            input=nodes["block_0"],
            n_filters=16,
            kernel_sz=3,
            residual=residual,
            name=name + "_stack_1",
            subsample=False,
            training=nodes["training"],
        )

        # output: 16, 16, 32
        b2 = add_block_stack(
            start=n + 1,
            end=2 * n + 1,
            input=b1,
            n_filters=32,
            kernel_sz=3,
            residual=residual,
            name=name + "_stack_2",
            training=nodes["training"],
        )

        # output: 8, 8, 64
        b3 = add_block_stack(
            start=2 * n + 1,
            end=3 * n + 1,
            input=b2,
            n_filters=64,
            kernel_sz=3,
            residual=residual,
            name=name + "_stack_3",
            training=nodes["training"],
        )

        nodes["reduce_mean"] = tf.nn.pool(
            input=b3,
            window_shape=(8, 8),
            strides=(1, 1),
            padding="VALID",
            pooling_type="AVG",
        )

        nodes["squeeze"] = tf.squeeze(
            input=nodes["reduce_mean"], name="squeeze", axis=[1, 2]
        )

        print("size of reduce_mean:", nodes["reduce_mean"].shape)
        print("size of squeeze:", nodes["squeeze"].shape)

        nodes["logits"] = tf.matmul(
            nodes["squeeze"],
            tf.get_variable(
                "fc",
                shape=(nodes["squeeze"].shape[-1], 10),
                dtype=ty,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            ),
        )
        nodes["loss"] = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=nodes["labels_to_int"], logits=nodes["logits"]
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.compat.v1.train.MomentumOptimizer(
                learning_rate=nodes["ph_lr"], momentum=0.9, use_nesterov=True
            ).minimize(
                nodes["loss"],
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
            )

        nodes["train_op"] = train_op

        return nodes


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    datasets = tf.keras.datasets
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = datasets.cifar10.load_data()
    train_labels = np.array([lab[0] for lab in train_labels])
    test_labels = np.array([lab[0] for lab in test_labels])
    return train_images, test_images, train_labels, test_labels


def train(
    sess: Any,
    nodes: Dict[str, Any],
    train_images: np.ndarray,
    test_images: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    batch_sz: int = 128,
    test_batch_sz: int = 100,
    n_epochs: int = 100,
):
    it = 0
    max_it = int(n_epochs * len(train_labels) / batch_sz)
    print(f"training for {max_it} iterations")

    losses = []
    test_acc = []
    train_acc = []

    for epoch in range(n_epochs):
        print(f"                          epoch {epoch}")
        idx_all = np.random.permutation(len(train_labels))
        for i in range(0, len(train_labels), batch_sz):
            it += 1
            idx = idx_all[i : i + batch_sz]
            _, loss = sess.run(
                [nodes["train_op"], nodes["loss"]],
                feed_dict={
                    nodes["ph_input"]: np.array(
                        [augment(img) for img in train_images[idx]]
                    ),
                    nodes["ph_labels"]: train_labels[idx],
                    nodes["ph_lr"]: utils.lr_schedule(it, max_it=max_it),
                    nodes["training"]: True,
                },
            )
            losses.append(np.mean(loss))
            if it % 100 == 0:
                print(f"iteration {it}, loss {np.mean(loss)}")

        # after each epoch, get test and train accuracies
        n_correct = 0
        for i in range(0, len(test_labels), test_batch_sz):
            test_img = test_images[i : i + test_batch_sz]
            test_lab = test_labels[i : i + test_batch_sz]
            logits = sess.run(
                nodes["logits"],
                feed_dict={nodes["ph_input"]: test_img, nodes["training"]: False},
            )
            pred = np.argmax(logits, axis=1)
            n_correct += np.sum(pred == test_lab)
        test_acc.append(n_correct / len(test_labels))
        print(f"TEST accuracy at end of epoch {epoch}: {n_correct / len(test_labels)}")

        n_correct = 0
        idx_all = np.random.permutation(len(train_labels))
        train_img = train_images[idx_all[:test_batch_sz]]
        train_lab = train_labels[idx_all[:test_batch_sz]]
        logits = sess.run(
            nodes["logits"],
            feed_dict={nodes["ph_input"]: train_img, nodes["training"]: False,},
        )
        pred = np.argmax(logits, axis=1)
        n_correct += np.sum(pred == train_lab)
        train_acc.append(n_correct / test_batch_sz)

        print(f"TRAIN accuracy at end of epoch {epoch}: {n_correct / test_batch_sz}")
    return losses, test_acc, train_acc
