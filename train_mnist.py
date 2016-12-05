#!/usr/bin/env python
from __future__ import print_function
import argparse
import functools
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from sklearn.datasets import fetch_mldata
from PIL import Image

def extract_images(raw_data, out_dir = "./image", sample = 1000):
    for i in range(0, sample):
        this_data = raw_data[i]
        if this_data.shape != (28*28,):
            print("Error raw_data seems not to be MNIST images.")
            return
        
        img = Image.frombytes(mode="L", size=(28,28), data=this_data.tobytes())
        img.save("%s/%d.png" % (out_dir, i))

def get_bit(byte, loc):
    return (byte & (1 << loc)) >> loc

# bit: 1 or 0
def set_bit(byte, loc, bit):
    if bit == 1:
        return (byte | (1 << loc))
    elif bit == 0:
        return (byte & ~(1 << loc))
    else:
        print("Error: bit argment should be 1 or 0 (%d given)" % bit)
        raise

def flip_one_bit(byte, loc):
    target_bit = get_bit(byte, loc)
    flipped_bit = (~target_bit) & 1
    ret = set_bit(byte, loc, flipped_bit)
    return ret

def get_indices(r, shape):
    ret = ()

    for i in range(0, len(shape)):
        mul = functools.reduce(lambda x,y: x*y, shape[i+1:], 1)
        ret += (int(r/mul), )
        r = int(r % mul)

    return ret

def inject_error(np_array, error_rate):
    buff = bytearray(np_array.data.tobytes())

    total_bits = len(buff) * 8
    n_error_bits = int(total_bits * error_rate)

    for _ in range(0, n_error_bits):
        r = int(np.random.randint(0, total_bits))
        (pos, bit) = get_indices(r, (len(buff), 8))
        byte = buff[pos]
        buff[pos] = flip_one_bit(byte, bit)

    return np.ndarray(shape=np_array.shape, dtype=np_array.dtype, buffer=buff)

def load_data():
    print("fetch MNIST dataset")
    mnist = fetch_mldata('MNIST original', data_home="./data")
    #mnist = fetch_mldata('MNIST', data_home="./data")
    mnist.target = mnist.target.astype(np.int32)

    return mnist.data, mnist.target

def make_tuple_dataset(x, y):
    tuples = []
    for i in range(0, len(x)):
        tuples.append((x[i], y[i]))
    return tuples

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--ber', '-r', type=float, default=0,
                        help='Bit Error Rate')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # load data using sklearn.dataset
    N = 60000
    img, label = load_data()
    img_train, img_test = np.split(img, [N])
    label_train, label_test = np.split(label, [N])

    print("inject bit errors. BER: %f" % args.ber)
    img_train = inject_error(img_train, args.ber)

    #extract_images(img_train)

    # convert data into float32 (as chainer requires so) after injecting error,
    # otherwise NaNs and INFs are generated and the learning does not work
    img_train = img_train.astype(np.float32) / 255
    img_test  = img_test.astype(np.float32) / 255

    train = make_tuple_dataset(img_train, label_train)
    test = make_tuple_dataset(img_test, label_test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name="%f.log" % args.ber))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
