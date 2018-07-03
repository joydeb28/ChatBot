from preprocessor import Dataset, pad_vec_sequences, labels, pad_class_sequence
from sklearn import model_selection
import numpy as np

#from keras.preprocessing import sequence
#from keras.models import Model
from keras.layers import Dense, Input, merge, Embedding, Bidirectional,GRU
#from keras.layers import Dropout
from keras.layers.recurrent import LSTM
#from keras import optimizers
from keras.layers import concatenate, Activation
#from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from keras.models  import Sequential
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras.layers.core import Dropout
from keras.regularizers import l2

from keras import backend as K

params = {
        "kernel_sizes_cnn":[1,2,3],
        "embedding_size":384,
        "filters_cnn": 256,
        "confident_threshold": 0.5,
        "optimizer": "Adam",
        "lear_rate": 0.01,
        "lear_rate_decay": 0.1,
        "loss": "binary_crossentropy",
        "last_layer_activation": "softmax",
        "text_size": 50,
        "coef_reg_cnn": 0.001,
        "coef_reg_den": 0.01,
        "dropout_rate": 0.5,
        "dense_size": 100,
        "model_name": "cnn_model"}

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
       
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    def f1Score(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return 2*((p*r)/(p+r+K.epsilon()))


def intialization():
    maxlen = 50 #sentences with length > maxlen will be ignored
    hidden_dim = 32
    nb_classes = len(labels)
    #initialise batch_size
    batch_size = 10
    #initialise num_epoch
    num_epoch = 100
    return maxlen,hidden_dim,nb_classes,batch_size,num_epoch

def make_data_set():
    ds = Dataset()
    print("Datasets loaded.")
    X_all = pad_vec_sequences(ds.X_all_vec_seq)
    Y_all = ds.Y_all
    #print (X_all.shape)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X_all,Y_all,test_size=0.2)
    y_train = pad_class_sequence(y_train, nb_classes)
    y_test = pad_class_sequence(y_test, nb_classes)
    y_test = np.array(y_test)
    x_train = np.asarray(x_train)
    x_train.ravel()
    
    y_train = np.asarray(y_train)
    y_train.ravel()
    return x_train,y_train,x_test,y_test

'''
def create_model(maxlen,hidden_dim,nb_classes):
    
    model=Sequential()
    model.add(Dropout(0.6, input_shape=(50, 384)))
    #model.add(Embedding(len(vocabulary), embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(32,activation='relu'))
    #model.add(Bidirectional((LSTM(200,activation='relu'))))
    model.add(Dropout(0.015))
    model.add(Dense(20))
    # model.add(LSTM(200,activation='relu',return_sequences=True))
    # model.add(Dropout(0.005))
    # model.add(LSTM(100,activation='relu',return_sequences=True))
    # model.add(LSTM(50))
    
    model.add(Dense(nb_classes, activation='softmax'))
    #rmsprop=optimizers.rmsprop(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'f1score', 'precision', 'recall'])
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[f1.precision,f1.recall,f1.f1Score])
    return model
'''
def cnn_model(params):
    inp = Input(shape=(params['text_size'], params['embedding_size']))
    outputs = []
    for i in range(len(params['kernel_sizes_cnn'])):
        output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                          activation=None,
                          kernel_regularizer=l2(params['coef_reg_cnn']),
                          padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(params['dense_size'], activation=None,
                    kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=params['dropout_rate'])(output)
    output = Dense(nb_classes, activation=None,
                   kernel_regularizer=l2(params['coef_reg_den']))(output)
    output = BatchNormalization()(output)
    act_output = Activation("softmax")(output)
    model = Model(inputs=inp, outputs=act_output)
    model.compile(optimizer="Adam",
                      loss="binary_crossentropy",
                      metrics=['accuracy'],
                      #loss_weights=loss_weights,
                      #sample_weight_mode=sample_weight_mode,
                      # weighted_metrics=weighted_metrics,
                      # target_tensors=target_tensors
                      )
    return model

def model_train(model,x_train,y_train,x_test,y_test,batch_size,num_epoch):

    print("Fitting to model")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, validation_data=[x_test, y_test])
    print("Model Training complete.")
    return model

def save_model(model):    
    model.save("backup/intent_models/model_test_benchmark_DSCNN.h5")
    print("Model saved to Model folder.")
		
		
maxlen,hidden_dim,nb_classes,batch_size,num_epoch = intialization()
x_train,y_train,x_test,y_test = make_data_set()
#model = create_model(maxlen,hidden_dim,nb_classes)
model = cnn_model(params)
model = model_train(model,x_train,y_train,x_test,y_test,batch_size,num_epoch)
save_model(model)

