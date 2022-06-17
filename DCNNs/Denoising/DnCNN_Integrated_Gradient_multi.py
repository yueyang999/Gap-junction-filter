#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import keras
import os
from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers import Conv2D
from keras import optimizers
from keras import callbacks
import random
from IntegratedGradients import *
from keras.layers import Dense
from keras.layers.core import Activation
import matplotlib.pyplot as plt
import pdb
import scipy.io
import random



def load_data(args):
    with open('../Data/'+args.dataset_test+'.pkl', 'rb') as f:
        noise_test, origin_test = pickle.load(f)
        x_test = noise_test.astype('float32')
        y_test = origin_test.astype('float32')
        return x_test, y_test


def PSNR(y_true, y_pred):
    img = K.batch_flatten(y_pred)
    ref = K.batch_flatten(y_true)
    img = K.clip(img, 0., 1.)
    ref = K.clip(ref, 0., 1.)
    mse = K.mean(K.square(img-ref), axis=1)
    psnr = -2.*K.log(mse)/np.log(2.)
    return psnr


def PSNR_np(img, ref):
    img = np.reshape(img, (np.shape(img)[0], -1))
    ref = np.reshape(ref, (np.shape(ref)[0], -1))
    img = np.clip(img, 0., 1.)
    ref = np.clip(ref, 0., 1.)
    mse = np.mean((img-ref)**2, axis=1)
    psnr = -2.*np.log2(mse)
    return psnr


def SSIM_np(img, ref):
    from skimage.measure import compare_ssim
    img = np.clip(img, 0., 1.)
    ref = np.clip(ref, 0., 1.)
    ssim = []
    for img_i, ref_i in zip(img, ref):
        ssim.append(compare_ssim(img_i, ref_i, multichannel=True, gaussian_weights=True))
    return np.stack(ssim)


def build_model(args):
    # define model
    y = Input(shape=(321, 481, 1))

    v = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(y)

    for i in range(args.depth-2):
        v = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(v)
        v = BatchNormalization()(v)
        v = Activation('relu')(v)

    v = Conv2D(1, 3, padding='same', kernel_initializer='he_normal')(v)

    #x = keras.layers.subtract([y, v])
    
    x = keras.layers.Reshape((154401,))(v)
    model = Model(inputs=y, outputs=x)
    
    return model


def train(single_model, model, data, args):
    """
    Training
    :param model: the model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    
    class ParallelModelCheckpoint(callbacks.ModelCheckpoint):
        def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                    save_best_only=False, save_weights_only=False,
                    mode='auto', period=1):
            self.single_model = model
            super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

        def set_model(self, model):
            super(ParallelModelCheckpoint,self).set_model(self.single_model)

    checkpoint = ParallelModelCheckpoint(single_model, args.save_dir + '/weights-{epoch:03d}.h5', monitor='val_PSNR',
                                         verbose=1, save_best_only=True, save_weights_only=True, mode='max')
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr/2. if epoch > 30 else args.lr)

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss='mean_squared_error',
                  metrics=[PSNR])

    # training
    def train_generator(x, y, args):
        datagen_args = dict(horizontal_flip=args.aug,
                            vertical_flip=args.aug)
        x_datagen = ImageDataGenerator(**datagen_args)
        y_datagen = ImageDataGenerator(**datagen_args)
        seed = 1234
        x_generator = x_datagen.flow(x, batch_size=args.batch_size, seed=seed)
        y_generator = y_datagen.flow(y, batch_size=args.batch_size, seed=seed)
        while 1:
            x_batch = x_generator.next()
            y_batch = y_generator.next()
            if args.aug:
                x_batch_aug = np.zeros_like(x_batch)
                y_batch_aug = np.zeros_like(y_batch)
                for i in range(len(x_batch)):
                    x_i = x_batch[i]
                    y_i = y_batch[i]
                    cnt = np.random.randint(4)
                    x_i = np.rot90(x_i, k=cnt)
                    y_i = np.rot90(y_i, k=cnt)
                    x_batch_aug[i] = x_i
                    y_batch_aug[i] = y_i
                yield (x_batch_aug, y_batch_aug)
            else:
                yield (x_batch, y_batch)
    
    model.fit_generator(train_generator(x_train, y_train, args),
                        steps_per_epoch=len(y_train) // args.batch_size,
                        epochs=args.epochs,
                        validation_data=[x_test, y_test],
                        callbacks=[log, tb, checkpoint, lr_decay])


def test(model, data, args):
    """
    Training
    :param model: the model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: None
    """
    # unpacking the data
    x_test, y_test = data

    # psnr
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    
    import pickle
    f = open(args.save_dir + '/result.pkl', 'wb')
    pickle.dump(y_pred, f)
    f.close()

    mean_psnr = np.mean(PSNR_np(y_pred, y_test))
    print('Average PSNR: %.2f' % mean_psnr)
    mean_ssim = np.mean(SSIM_np(y_pred, y_test))
    print('Average SSIM: %.4f' % mean_ssim)


if __name__ == "__main__":
    import os
    import argparse
    import tensorflow as tf
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model
    from keras.utils import multi_gpu_model
    import pickle
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="DnCNN Network")
    parser.add_argument('--dataset_train', default='train_Gaussian_noise_50_128')
    parser.add_argument('--dataset_test', default='test_Gaussian_noise')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--aug', action='store_true',
                        help="Apply data augmentation")
    # parser.add_argument('--input_size', default=50, type=int)
    parser.add_argument('--depth', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    # parser.add_argument('--lr_decay', default=0.871, type=float,
    #                     help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-w', '--weights', default="./trained_weights.h5",
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--gpus', default=4, type=int)
    parser.add_argument('--region', default="out_double.npy")
    parser.add_argument('--region_name', default="65_texture_block")
    
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_test, y_test) = load_data(args)

    # define model
    if args.gpus > 1:
        with tf.device('/cpu:0'):
            model = build_model(args)
    else:
        model = build_model(args)
    model.summary()
    
    # define muti-gpu model
    if args.gpus > 1:
        multi_model = multi_gpu_model(model, gpus=args.gpus)
    
    # train
    if args.weights is not None:  # init the model weights with provided one
        try:
            model.load_weights(args.weights)
        except ValueError as e:
            multi_model.load_weights(args.weights)
    else:
        if args.gpus > 1:
            train(single_model=model, model=multi_model, data=((x_train, y_train), (x_test, y_test)), args=args)
        else:
            train(single_model=model, model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
        # save weights
        model.save_weights(args.save_dir + '/trained_weights.h5')
        print('Trained weights saved to \'%s/trained_weights.h5\'' % args.save_dir)

    # test
    if args.gpus > 1:
        multi_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss='mean_squared_error')
        os.mkdir('./'+args.region_name)
        import matplotlib.pyplot as plt
        

        outchannels=[]
        out_double=[]
        out_double=np.load('../'+'out_double/'+args.region)
        outchannels=np.dot(out_double,[481,1])
       	

        plt.imshow(np.reshape(x_test[237],(321,481)),cmap='gray')
        plt.savefig('./'+args.region_name+'/test.png')

        plt.imshow(np.reshape(y_test[33],(321,481)),cmap='gray')
        plt.savefig('./'+args.region_name+'/test_truth.png')

        result=np.reshape(multi_model.predict(np.expand_dims(x_test[237],axis=0)),(321,481))
        plt.imshow(result,cmap='gray')
        plt.savefig('./'+args.region_name+'/test_result.png')



        ig=integrated_gradients(multi_model,outchannels=outchannels)
        
        xx_test=np.reshape(x_test[237],(321,481))
        #np.save('out_double.npy',out_double)
        
        for i in range(len(outchannels)):
            
            x=out_double[i][0]
            y=out_double[i][1]

            ige=np.reshape(ig.explain(x_test[237],outc=outchannels[i],num_steps=10),(321,481))
            scipy.io.savemat('./'+args.region_name+'/result_noref'+str(i)+'_'+str(x)+'_'+str(y), mdict={'ige_data': ige})

            
            

    else:
        test(model=model, data=(x_test, y_test), args=args)
        ig=integrated_gradients(model)
        ig.explain(x_test[237],outc=[[10,10]])
