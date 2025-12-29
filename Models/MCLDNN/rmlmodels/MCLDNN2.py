import os
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, concatenate, Reshape, Conv2D, LSTM, BatchNormalization, AlphaDropout
from keras.utils import plot_model

def MCLDNN(weights=None, input_shape1=[2,128], input_shape2=[128,1], classes=11, **kwargs):
    dr = 0.5 
    
    input1 = Input(input_shape1+[1], name='input1')
    input2 = Input(input_shape2, name='input2')
    input3 = Input(input_shape2, name='input3')

    # SeparateChannel Combined Convolutional Neural Networks
    x1 = Conv2D(50, (2,8), padding='same', activation="relu", name="conv1_1", kernel_initializer='glorot_uniform')(input1)
    x1 = BatchNormalization()(x1)

    x2 = Conv1D(50, 8, padding='causal', activation="relu", name="conv1_2", kernel_initializer='glorot_uniform')(input2)
    x2 = BatchNormalization()(x2)
    x2_reshape = Reshape([-1, 128, 50])(x2)

    x3 = Conv1D(50, 8, padding='causal', activation="relu", name="conv1_3", kernel_initializer='glorot_uniform')(input3)
    x3 = BatchNormalization()(x3)
    x3_reshape = Reshape([-1, 128, 50], name='reshap2')(x3)

    x = concatenate([x2_reshape, x3_reshape], axis=1)
    x = Conv2D(50, (1,8), padding='same', activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    
    x = concatenate([x1, x])
    x = Conv2D(100, (2,5), padding='valid', activation="relu", name="conv4", kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)

    # LSTM Unit
    x = Reshape(target_shape=((124, 100)), name='reshape')(x)
    # Fixed typo: recurrent_dropout
    x = LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = LSTM(units=128, dropout=0.2)(x)

    # DNN - SELU + AlphaDropout + Lecun_normal is a powerful combo
    x = Dense(128, activation='selu', kernel_initializer='lecun_normal', name='fc1')(x)
    x = AlphaDropout(dr)(x)

    x = Dense(128, activation='selu', kernel_initializer='lecun_normal', name='fc2')(x)
    x = AlphaDropout(dr)(x)

    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=[input1, input2, input3], outputs=x)

    if weights is not None:
        model.load_weights(weights)
    
    return model

if __name__ == '__main__':
    # Changed classes to 11 for RML2016.10a
    model = MCLDNN(None, classes=11)
    
    # Using learning_rate instead of lr
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    
    model.summary()