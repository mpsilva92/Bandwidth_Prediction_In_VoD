from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import logging
import numpy as np
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")
log = logging.getLogger(__name__)


class DataPrepare(object):
    def __init__(self, path, window, horizon, FLAG=True):
        try:
            log.debug('Load data...\n')
            file = open(path)
            self.dataLoad = np.loadtxt(file, delimiter=',')
            log.debug('The dataset was loaded successfully!')
            # self.data              = np.zeros(self.dataLoad.shape)
            self.window            = window     # 10 segments in past
            self.horizon           = horizon    # 5 segments in future
            self.target            = None
            # self.dataScale        = np.ones(self.col)
            # self.scaleList         = []
            self.FLAG              = FLAG
            # self.data_x          = self.dataLoad.iloc[:, :]
            # self.data_t          = self.dataLoad[['downthpt']] # target feature
            
            # normalize data
            self.data_normalise()
            
            # for RNN model
            if FLAG == True:
                self.data_prepare(FLAG)

        except IOError as err:
            log.error('Error opening dataset for reading... %s', err)


    # apply the z-score standardization on Numpy Array using the .mean() and .std() methods
    def z_score(self):
        
        # This function transforms the data into 
        # a distribution with a mean of 0 and a standard deviation of 1 

        # create an array with zeros value 
        self.data               = np.zeros(self.dataLoad.shape)
        self.row, self.col      = self.data.shape  # numbers of rows and columns

        # apply the z-score method
        for column in range(self.col):
            self.data[:, column] = (self.dataLoad[:, column] - np.mean(self.dataLoad[:, column])) / np.std(self.dataLoad[:, column])
            
            # self.data[:, column] = np.isfinite((self.dataLoad[:, column] - np.mean(self.dataLoad[:, column])) / np.std(self.dataLoad[:, column])).all()
        self.target = self.data[:, -1]
        
        
    def data_normalise(self):
        # default normalization
        # logging.debug('Mode %d: Z-Score\n', norm_t)
        self.z_score()


    # split data to lookback
    def data_division(self, start, end, lag=True):
        X = []
        T = []
        
        start = start + self.window
        if lag is False:
            end = end - self.horizon
        for index in range(start, end):
            indices = range(index-self.window, index)
            # print('index', index)
            # print('indices', indices)
            X.append(self.data[indices])
            indicey = range(index+1, index+1+self.horizon)
            # print('index horizon', indicey)
            T.append(self.target[indicey])
        return [np.array(X), np.array(T)]


    def data_prepare(self, flag):
        # train_s        = int(train_size * len(self.data))
        # valid_s        = int(train_size * len(self.data)) + int(valid_size * len(self.data))
        # test_s         = int((valid_size * len(self.data)) + valid_s)
        
        # self.set_len   = [train_s, valid_s, test_s]

        # self.X_train = self.data_division(0, train_s, lag=True)
        # self.X_valid = self.data_division(train_s, valid_s, lag=False)
        START, END = 0, self.data.shape[0]
        if flag == True:
            self.X_test  = self.data_division(START, END, lag=False)

        # log.info("Training shape: X:%s Target:%s", str(self.X_train[0].shape), str(self.X_train[1].shape))
        # log.info('Validation shape: X:%s, Target:%s', str(self.X_valid[0].shape), str(self.X_valid[1].shape))
        log.info('Test shape X:%s, Target:%s', str(self.X_test[0].shape), str(self.X_test[1].shape))


# metrics
class Metrics(object):
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    # This function calculates the mae
    def mean_absolute_error(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true), axis=None)


    # This function calculates the mse
    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=None)
        
        
    # This function calculates the correlation
    def corr(y_true, y_pred):
        #
        # This function calculates the correlation between the true and the predicted outputs
        #
        num1 = y_true - K.mean(y_true, axis=0)
        num2 = y_pred - K.mean(y_pred, axis=0)
        
        num  = K.mean(num1 * num2, axis=0)
        den  = K.std(y_true, axis=0) * K.std(y_pred, axis=None)
        
        return K.mean(num / den)


    # This function calculates the RSE
    def rse(y_true, y_pred):
        #
        # The formula is:
        #           K.sqrt(K.sum(K.square(y_true - y_pred)))     
        #    RSE = -----------------------------------------------
        #           K.sqrt(K.sum(K.square(y_true_mean - y_true)))       
        #
        #           K.sqrt(K.sum(K.square(y_true - y_pred))/(N-1))
        #        = ----------------------------------------------------
        #           K.sqrt(K.sum(K.square(y_true_mean - y_true)/(N-1)))
        #
        #
        #           K.sqrt(K.mean(K.square(y_true - y_pred)))
        #        = ------------------------------------------
        #           K.std(y_true)
        #
        num = K.sqrt(K.mean(K.square(y_true - y_pred), axis=None))
        den = K.std(y_true, axis=None)
        
        return num / den
    
    
class MovingAverageBasedModels(object):
    def __init__(self, df):
        self.df = df
    
    def moving_average(df):

        # initialize an empty list to store cumulative moving averages of each session (measurements-15 sec)
        moving_averages = []

        # start and end of each session
        start = 0
        end = 5

        # loop through the elements array
        while end <= df.shape[0]:
            # calculate the cumulative average using pandas.Series.rolling method
            moving_average = df.iloc[start:end, -1].rolling(5, min_periods=1).mean()
            # print(moving_average)
            moving_averages.append(moving_average)

            end += 5
            start += 5
            
        return moving_averages
    
    
    def harmonic_mean(df):

        # initialize an empty list to store cumulative harmonic mean of each session (measurements-15 sec)
        hm = []

        # start and end of each session
        start = 0
        end = 5
        count = 1

        # loop through the array elements
        while end <= df.shape[0]:
            # calculate the cumulative harmonic mean using hmean method
            harm_mean = df.iloc[start:end, -1].values
            for i in range(1, len(harm_mean)):
                hm_value = hmean(harm_mean[:i])
                # print(harm_mean)
                hm.append(hm_value)
            end += 5
            start += 5
            
        return hm
    
    def exponentially_weighted_moving_average(df):

        # initialize an empty list to store cumulative EWMA of each session (measurements - 15 sec)
        EWMA = []

        # start and end of each session
        start = 0
        end = 5

        # sliding window 
        window = 5

        # loop through the array elements
        while end <= df.shape[0]:
            # calculate the cumulative EWMA using pandas.Series.rolling method
            moving_average = df.iloc[start:end, -1].ewm(span= window, adjust=False).mean()
            # print(moving_average)
            EWMA.append(moving_average)

            end += 5
            start += 5
            
        return EWMA


# This class implement the initial layer for AR component 
class PreARTrans(tf.keras.layers.Layer):
    def __init__(self, hw, **kwargs):
        #
        # hw: Highway = Number of timeseries values to consider for the linear layer (AR layer)
        #
        self.hw = hw
        super(PreARTrans, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(PreARTrans, self).build(input_shape)
    
    def call(self, inputs):
        # Get input tensors; in this case it's just one tensor: X = the input to the model
        x = inputs

        # Get the batchsize which is tf.shape(x)[0]
        batchsize = tf.shape(x)[0]

        # Get the shape of the input data
        input_shape = K.int_shape(x)
        
        # Select only 'highway' length of input to create output
        output = x[:,-self.hw:,:]
        
        # Permute axis 1 and 2. axis=2 is the the dimension having different time-series
        # This dimension should be equal to 'm' which is the number of time-series.
        output = tf.transpose(output, perm=[0,2,1])
        
        # Merge axis 0 and 1 in order to change the batch size
        output = tf.reshape(output, [batchsize * input_shape[2], self.hw])
        
        # Adjust the output shape by setting back the batch size dimension to None
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        
        return output
    
    def compute_output_shape(self, input_shape):
        # Set the shape of axis=1 to be hw since the batchsize is NULL
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.hw
        
        return tf.TensorShape(shape)

    def get_config(self):
        config = {'hw': self.hw}
        base_config = super(PreARTrans, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# posterior layer for AR component   
class PostARTrans(tf.keras.layers.Layer):
    def __init__(self, m, **kwargs):
        #
        # m: Number of timeseries
        #
        self.m = m
        super(PostARTrans, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(PostARTrans, self).build(input_shape)
    
    def call(self, inputs):
        # Get input tensors
        # - First one is the output of the Dense(1) layer which we will operate on
        # - The second is the original model input tensor which we will use to get
        #   the original batchsize
        x, original_model_input = inputs

        # Get the batchsize which is tf.shape(original_model_input)[0]
        batchsize = tf.shape(original_model_input)[0]

        # Get the shape of the input data
        input_shape = K.int_shape(x)
        
        # Reshape the output to have the batch size equal to the original batchsize before PreARTrans
        # and the second dimension as the number of timeseries
        output = tf.reshape(x, [batchsize, self.m])
        
        # Adjust the output shape by setting back the batch size dimension to None
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        
        return output
    
    def compute_output_shape(self, input_shape):
        # Adjust shape[1] to be equal 'm'
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.m
        
        return tf.TensorShape(shape)

    def get_config(self):
        config = {'m': self.m}
        base_config = super(PostARTrans, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# load weights of the model
def LoadModel(filename, custom_objects):
    model = None

    if filename is not None:
        try:
            log.info("Loading model and weights ...")
        except NameError:
            pass

        # load and create model from json file
        file = filename + ".json"
        if os.path.exists(file):
            with open(file, "r") as json_file:
                try:
                    log.debug("Loading model from: %s", file)
                except NameError:
                    pass

                model = model_from_json(json_file.read(), custom_objects=custom_objects)
        
            # load weights from h5 file into new model
            file = filename + ".h5"
            if os.path.exists(file):
                try:
                    log.debug("Loading weights from: %s", file)
                except NameError:
                    pass

                model.load_weights(file)
            else:
                try:
                    log.critical("File %s does not exist", file)
                except NameError:
                    print("File %s does not exist" % (file))
        else:
            try:
                log.critical("File %s does not exist", file)
            except NameError:
                print("File %s does not exist" % (file))
    
    return model