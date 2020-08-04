from keras.layers import *
from keras.models import *
from keras.optimizers import Adam,RMSprop


#model configuration (parameters)
#======================================================================================= 
config = {}
config['epoch'] = 10
config['batch_size'] = 10
config['lr'] = 0.01 # optimizer: learning rate
config['decay'] = 0.0 # optimizer: decay of learning rate
config['dropout'] = 0.2 # lstm: dropout rate
config['lstm_out'] = 1000 # lstm: output unit 
config['time_ahead'] = 10 # lstm: number of time steps predicted
config['cnn_act'] = 'relu' # cnn: activation function --> tanh
config['loss'] = 'mean_squared_error'

#model
#=======================================================================================

def cnn(images):
    #data_format='channels_first' / data_format='channels_last'
    a = Conv2D(16, (3, 3), padding='valid',data_format='channels_last',input_shape=images.shape[1:])(images)
    a = Activation(config['cnn_act'])(a)
    a = MaxPooling2D(pool_size=(2, 2))(a)
    a = Conv2D(32, (3, 3), padding='valid')(a)
    a = Activation(config['cnn_act'])(a)
    a = MaxPooling2D(pool_size=(2, 2))(a)
    a = Flatten()(a)
    return a


def lstm(seq):
    b = Bidirectional(LSTM(config['lstm_out'], dropout=config['dropout'], return_sequences=False))(seq)
    b = RepeatVector(config['time_ahead'])(b)
    b = Bidirectional(LSTM(config['lstm_out'], return_sequences=True))(b)
    b = TimeDistributed(Dense(166))(b)   
    return b


def buildCNNLSTM():
    
    cnn_input = Input((80,80,20))
    lstm_input = Input((20,166))
    cnn_output = cnn(cnn_input) 
    lstm_output = lstm(lstm_input) 
    concat = Concatenate()([cnn_output, lstm_output])
    output = Dense(166,use_bias=True)(concat)
    model = Model(inputs=[cnn_input, lstm_input], outputs=output)

    return model # model


model = buildCNNLSTM()
adam = Adam(lr=config['lr'], beta_1=0.9, beta_2=0.999, decay=config['decay'])
rmsprop = RMSprop(lr=config['lr'], rho=0.9, decay=config['decay'])
model.compile(optimizer=rmsprop, loss=config['loss'], metrics=['accuracy'])

#read dataset
#======================================================================================= 





#train
#======================================================================================= 
print('Training------------')
model.fit(X_train, y_train, epochs=config['epoch'], batch_size=config['batch_size'])


#test
#======================================================================================= 
print('Testing--------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss:', loss)
print('test accuracy:', accuracy)