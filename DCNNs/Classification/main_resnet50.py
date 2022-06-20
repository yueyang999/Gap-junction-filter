import math
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
from keras import callbacks
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping

batchsize = 64
sgd_momentum = 0.997
lr = 0.025
epochs = 50
train_data_size = 213555
test_data_size = 10225

# learning rate schedule
def step_decay(epoch, learning_rate):
    lrate = learning_rate
    if epoch in [30, 60, 80, 90]:
        lrate = learning_rate * 0.1
    return lrate

def build_model():
    # create the base_model model
    base_model = ResNet50(weights=None, include_top=False,input_shape = (224,224,1))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 16 classes
    predictions = Dense(16, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train(single_model,model, train_data, test_data, save_weights):
    # compile the model (should be done *after* setting layers to non-trainable)
    sgd = SGD(lr=lr, momentum=sgd_momentum, decay=0.0)
    adam =Adam(lr=0.012, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True) 
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)
    #early_stopping =EarlyStopping(monitor='val_acc',patience=10,min_delta=0.01)
    # callbacks
    log = callbacks.CSVLogger('weights/resnet50/'+ args.save_weights + '/log.csv')
    tb = callbacks.TensorBoard(log_dir='weights/resnet50/'+ args.save_weights + '/tensorboard-logs',
                               batch_size=batchsize, histogram_freq=int(args.debug))
    
    class ParallelModelCheckpoint(callbacks.ModelCheckpoint):
        def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                    save_best_only=False, save_weights_only=False,
                    mode='auto', period=1):
            self.single_model = model
            super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

        def set_model(self, model):
            super(ParallelModelCheckpoint,self).set_model(self.single_model)

    checkpoint = ParallelModelCheckpoint(single_model, 'weights/resnet50/'+ args.save_weights + '/weights-{epoch:03d}.h5', monitor='val_acc',
                                         verbose=1, save_best_only=True, mode='max')
    # train the model on the new data for a few epochs
    model.fit_generator(generator=train_data,
                        validation_data=test_data,
                        epochs=epochs,
                        steps_per_epoch=math.ceil(train_data_size/batchsize),
                        validation_steps=math.ceil(test_data_size/batchsize),
                        callbacks=[log, tb, checkpoint,lrate],
                        class_weight=[0.0027,0.0112,0.0203,0.0133,0.0047,0.0099,0.0248,0.0164,0.0083,0.0042,0.0326,0.2694,0.0110,0.0157,0.0307,0.5246],
                        verbose=1)

def load_data(train_data,test_data):
    # loaddata
    # seed=1234
    train_path=train_data
    test_path=test_data
    train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
    train_data = train_datagen.flow_from_directory(train_path,
                                                   batch_size=batchsize,
                                                   #  seed=seed,
                                                   class_mode='categorical',
                                                   target_size=(224, 224),
                                                   color_mode='grayscale')
    test_data = test_datagen.flow_from_directory(test_path,
                                                 batch_size=batchsize,
                                                 #    seed=seed,
                                                 class_mode='categorical',
                                                 target_size=(224, 224),
                                                 color_mode='grayscale')
    return train_data, test_data


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="ResNet50 Network")
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--train_data',default=None)
    parser.add_argument('--test_data',default=None)
    parser.add_argument('--save_weights',default=None)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    args = parser.parse_args()
    #print(args)
    with tf.device('/cpu:0'):
        train_data, test_data = load_data(args.train_data,args.test_data)
    # train_data, test_data = load_data()
    if args.gpus > 1:
        with tf.device('/cpu:0'):
            model = build_model()
    else:
        model = build_model()
    #model.summary()
    # define muti-gpu model
    if args.gpus > 1:
        multi_model = multi_gpu_model(model, gpus=args.gpus)

    if args.weights is not None:  # init the model weights with provided one
        try:
            model.load_weights(args.weights)
        except ValueError as e:
            multi_model.load_weights(args.weights)
    else:
        if args.gpus > 1:
            train(model,multi_model, train_data, test_data,args.save_weights)
        else:
            train(model,model, train_data, test_data,args.save_weights)
    
    # save weights
    model.save_weights((args.save_weights+'_trained_weights.h5'))
    

