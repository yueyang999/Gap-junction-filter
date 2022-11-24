import math
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
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
from keras.callbacks import ModelCheckpoint

batchsize = 64
sgd_momentum = 0.997
lr = 0.025
epochs = 100
train_data_size = 213555
test_data_size = 10225
print (tf.__version__)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
def build_model():
    # create the base_model model
    base_model = VGG19(weights=None, include_top=False)

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
    sgd = SGD(lr=lr, momentum=sgd_momentum, decay=1e-6)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    # # learning schedule callback
    # lrate = LearningRateScheduler(step_decay)
    #early_stopping =EarlyStopping(monitor='val_acc',patience=10,min_delta=0.01)
    # callbacks
    log = callbacks.CSVLogger('/raid/yy/weights/vgg19/'+ args.save_weights + '/log.csv')
    tb = callbacks.TensorBoard(log_dir='/raid/yy/weights/vgg19/'+ args.save_weights + '/tensorboard-logs',
                               batch_size=batchsize, histogram_freq=int(args.debug))
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1)
    checkpoint = ModelCheckpoint('/raid/yy/weights/vgg19/'+ args.save_weights + '/weights-{epoch:03d}.h5', monitor='val_accuracy',
                                         verbose=1, save_best_only=False, mode='max', period=1)
    # train the model on the new data for a few epochs
    model.fit_generator(generator=train_data,
                        validation_data=test_data,
                        epochs=epochs,
                        steps_per_epoch=math.ceil(train_data_size/batchsize),
                        validation_steps=math.ceil(test_data_size/batchsize),
                        callbacks=[checkpoint, early_stopping, log, tb],
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
                                                   target_size=(224, 224))
    test_data = test_datagen.flow_from_directory(test_path,
                                                 batch_size=batchsize,
                                                 #    seed=seed,
                                                 class_mode='categorical',
                                                 target_size=(224, 224))
    return train_data, test_data


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="VGG19 Network")
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--train_data',default=None)
    parser.add_argument('--test_data',default=None)
    parser.add_argument('--save_weights',default=None)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    args = parser.parse_args()
    print(args.train_data)
    with tf.device('/cpu:0'):
        train_data, test_data = load_data(args.train_data,args.test_data)
    # train_data, test_data = load_data()
    model = build_model()
    #model.summary()
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    else:
        train(model,model, train_data, test_data,args.save_weights)
    
    # save weights
    model.save_weights(('/raid/yy/weights/vgg19/'+ args.save_weights+'_trained_weights.h5'))
    
    # test
    # test_path = 'elephant.png'
    # img = image.load_img(img_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # preds = model.predict(x)
    # # decode the results into a list of tuples (class, description, probability)
    # # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=3)[0])
    # # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
