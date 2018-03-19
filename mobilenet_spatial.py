import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten
from keras import optimizers

train_path = '/mnt/smalldata/train'
test_path = '/mnt/smalldata/test'

batch_size = 16
train = False
classes = 3

if train:
    train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_path,
        target_size=(224, 224),
        classes=['BaseballPitch', 'Basketball', 'BasketballDunk'],
        batch_size=batch_size
    )
else:
    test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_path,
        target_size=(224, 224),
        classes=['BaseballPitch', 'Basketball', 'BasketballDunk'],
        batch_size=batch_size
    )

# MobileNet model
if train:
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        weights='imagenet'
    )
else:
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
    )
# model.summary()

# Non-trainable layers, only train last layer
# for layer in model.layers: layer.trainable = False

# Modify network some last layer

x = Flatten()(model.layers[-4].output)
x = Dense(classes, activation='softmax', name='predictions')(x)

#Then create the corresponding model 
my_model = Model(input=model.input, output=x)
my_model.summary()

my_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy','top_k_categorical_accuracy'])

if train:
    my_model.load_weights('mobilenet_nor_weights_v3.h5')
    print len(train_batches)
    print batch_size
    print len(train_batches)

    my_model.fit_generator(
        train_batches,
        epochs=2,
    )
    
    my_model.save_weights('mobilenet_nor_weights_v3.h5')
else:
    my_model.load_weights('mobilenet_nor_weights_v3.h5')
    # score = my_model.evaluate_generator(test_batches)
    # for i in range(len(my_model.metrics_names)):
    #     print(str(my_model.metrics_names[i]) + ": " + str(score[i]))
    pred = my_model.predict_generator(test_batches)
    print pred
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

# mobilenet_nor_weights_v2 for regular MobileNet with RGB non-stack with change last layer
# mobilenet_stack_weights for regular MobileNet with RGB stack with change last layer, frist layer