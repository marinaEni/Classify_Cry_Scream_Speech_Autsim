# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 16:59:03 2021

@author: marinamu
"""
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, AveragePooling1D, Activation, BatchNormalization, LeakyReLU, ReLU, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import RandomUniform


class Models:
    def __init__(self, params, input_shape):
        self.params = params
        self.input_shape = input_shape # (1, n_embeddings)
        self.model_arc = params['model_arc']
        self.n_out = params.get('n_out', 3)
        self.out_act = params.get('out_act', 'softmax')
        self.loss = [params.get('loss', 'categoricl_crossentropy')]
        self.metric = [params.get('metric', 'accuracy')]

        self.decay_rate = 0  # decay rate of the initial learning rate. 0-1.
        self.decay_steps = 0  # every x steps drop the lr
        self.model = None

        self.define_mdl()

    # --------------------------------------------------------------------------------------------#
    def define_mdl(self):
        if self.model_arc == 'CNN_mdl':
            self.CNN_mdl()
            
        elif self.model_arc == "CNN_batch_mdl":
            self.CNN_batch_mdl()
            
        elif self.model_arc == 'dense_mdl':
            self.dense_mdl()
            
        elif self.model_arc == 'CNN1d_pool_fc':
            self.CNN1d_pool_fc()
            
        elif self.model_arc == 'dense_embSize_mdl':
            self.dense_embSize_mdl()
            
        elif self.model_arc == 'dense_embSize_batchNorm_mdl':
            self.dense_embSize_batchNorm_mdl()
            
        elif self.model_arc == 'dense_embSize_batchNorm_Leaky':
            self.dense_embSize_batchNorm_Leaky()
        else:
            raise AssertionError(f"Mission aborted: no such model architecture: {self.model_arc}")

    # --------------------------------------------------------------------------------------------#            
    def CNN_mdl(self):
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=3,
                         input_shape=self.input_shape,  # w = kx101xf, b = fx1
                         activation='relu',
                         kernel_initializer=RandomUniform(seed=1)))
        model.add(MaxPool1D(pool_size=3))# default: stride=pool_size
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu',
                         kernel_initializer=RandomUniform(seed=1)))  # , bias_initializer=Zeros()
        model.add(Dense(1024, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dense(128, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        #
        model.add(Flatten())
        model.add(Dense(self.n_out, activation=self.out_act,
                        kernel_initializer=RandomUniform(seed=1)))
        self.model = model
    
    # --------------------------------------------------------------------------------------------#
    def CNN_batch_mdl(self):
        """
        BN After Conv1D: This normalizes the output of each convolutional layer, helping the network
            converge faster and avoid vanishing/exploding gradients.
        BN Before ReLU: Normalization works best when applied before activation functions.
        """
        model = Sequential()
        # Reshape layer to add the channel dimension
        model.add(Reshape((self.input_shape, 1), input_shape=(self.input_shape,)))  # (1024,) -> (1024, 1)
        model.add(Conv1D(filters=256, kernel_size=3,
                         kernel_initializer=RandomUniform(seed=1)))
        model.add(BatchNormalization())  # Add BN after Conv
        model.add(ReLU())  # Explicitly specify activation
        model.add(MaxPool1D(pool_size=3))# default: stride=pool_size
        
        # Second Conv Block
        model.add(Conv1D(filters=256, kernel_size=3, 
                         kernel_initializer=RandomUniform(seed=1)))  # , bias_initializer=Zeros()
        model.add(BatchNormalization())  # Add BN after Conv
        model.add(ReLU())  # Explicitly specify activation
        model.add(MaxPool1D(pool_size=3))
        
        model.add(Flatten())
        
        # Fully Connected Layers
        model.add(Dense(1024, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dense(128, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        # 
        model.add(Dense(self.n_out, activation=self.out_act,
                        kernel_initializer=RandomUniform(seed=1)))
        self.model = model
        
    # --------------------------------------------------------------------------------------------#
    def CNN1d_pool_fc(self):
        """
        Understanding Spoken Language Development of Children with ASD Using Pre-trained Speech Embeddings.
        Interspeech 2023
        """
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=3,
                         input_shape=self.input_shape,  # w = kx101xf, b = fx1
                         activation='relu',
                         kernel_initializer=RandomUniform(seed=1)))
        model.add(Conv1D(filters=256, kernel_size=3,
                         activation='relu',
                         kernel_initializer=RandomUniform(seed=1)))
        model.add(Conv1D(filters=256, kernel_size=3,
                         activation='relu',
                         kernel_initializer=RandomUniform(seed=1)))
        model.add(AveragePooling1D(channels_first='first'))
        model.add(Dense(256, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dense(self.n_out, activation=self.out_act,
                        kernel_initializer=RandomUniform(seed=1)))

        self.model = model

    # ==============================================================================================
    def conv1d_mdl(self):
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=3,
                         input_shape=self.input_shape, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.params["n_out"], activation='softmax'))

        # model.summary()

        self.model = model

    # ==============================================================================================
    def dense_mdl(self):
        model = Sequential()
        model.add(Dense(256, input_shape=self.input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.params["n_out"], activation=self.params["act_out"]))  # softmax: returns probabilities

        # model.summary()
        self.model = model

    # ==============================================================================================
    def dense_embSize_mdl(self):
        n_embeddings = self.input_shape[0]
        model = Sequential()
        model.add(Dense(n_embeddings, input_shape=self.input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n_embeddings/2, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.params["n_out"], activation=self.params["act_out"]))  # softmax: returns probabilities

        # model.summary()
        self.model = model
        
    # ==============================================================================================
    def dense_embSize_batchNorm_mdl(self):
        n_embeddings = self.input_shape[0]
        
        model = Sequential()
        model.add(Dense(n_embeddings, input_shape=self.input_shape, activation=None))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(n_embeddings/2, activation=None))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        # Last leyer:
        model.add(Dense(self.params["n_out"], activation=self.params["act_out"]))  # softmax: returns probabilities

        # model.summary()
        self.model = model
        
    # ==============================================================================================
    def dense_embSize_batchNorm_Leaky(self):
        n_embeddings = self.input_shape[0]
        
        model = Sequential()
        model.add(Dense(n_embeddings, input_shape=self.input_shape, activation=None))
        model.add(BatchNormalization())
        model.add(LeakyReLU()) # default: (negative_slope=0.3)
        model.add(Dropout(0.2))
        
        model.add(Dense(n_embeddings/2, activation=None))
        model.add(BatchNormalization())
        model.add(LeakyReLU()) # default: (negative_slope=0.3
        model.add(Dropout(0.2))
        # Last leyer:
        model.add(Dense(self.params["n_out"], activation=self.params["act_out"]))  # softmax: returns probabilities

        # model.summary()
        self.model = model


