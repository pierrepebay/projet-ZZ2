import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import callbacks
from keras import models
from keras import utils
from keras import optimizers
from keras import initializers
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Classes import Quote
from model_validation import get_train_val_test_idx_rolling, get_train_val_test_idx_regular


class LongShortTermMemory:
  def __init__(self, data: pd.DataFrame, label_fit: pd.Series, feature: str, symbol: str, 
               strategy_code: str, model_parameters: Dict[str, Any], save_path: str) -> None:
    """
    Fit an LSTM on a given symbol.
    The data is assumed to be standardized with respect to the basket of stocks at each date
    """
    self.symbol = symbol
    self.strategy_code = strategy_code
    self.model_parameters = model_parameters
    self.series = data[data.index.get_level_values("symbol") == self.symbol][feature]
    self.y = label_fit[label_fit.index.get_level_values("symbol") == self.symbol]
    self.feature = feature
    self.model = None
    self.h_callback = None

  def split_series(self) -> Tuple[List[pd.MultiIndex], List[pd.MultiIndex], List[pd.MultiIndex]]:
    """
    Split the given time series in 3 sub series (train, val, test) according to the given ratios

    Returns
    -------
    train_idx, val_idx, test_idx : Tuple[List[pd.MultiIndex], List[pd.MultiIndex], List[pd.MultiIndex]]
      Each list contains relevant indexes to extract the data in the original time series.
    """
    ratio_train = self.model_parameters["split_ratio"]["train"]
    ratio_val = self.model_parameters["split_ratio"]["val"]
    train_idx, val_idx, test_idx = get_train_val_test_idx_regular(data=self.series, ratio_train=ratio_train, ratio_val=ratio_val)
    return train_idx, val_idx, test_idx

  def get_data(self, train_idx: List[pd.MultiIndex], val_idx: List[pd.MultiIndex], test_idx: List[pd.MultiIndex]) -> Dict[str, Dict[str, pd.Series]]:
    """
    Parameters
    ----------
    train_idx : List[pd.MultiIndex]
      Contain train indexes to extract in the original time series
    val_idx : List[pd.MultiIndex]
      Contain val indexes to extract in the original time series  
    test_idx : List[pd.MultiIndex]
      Contain test indexes to extract in the original time series  
    Returns
    -------
    res : Dict[str, Dict[str, pd.Series]]
      Global dictionary which contain relevant sets of data like x_train, x_val, etc.
    """
    res = {}
    # Features Data
    x_train = self.series.loc[train_idx[0]]
    x_val = self.series.loc[val_idx[0]]
    x_test = self.series.loc[test_idx[0]]

    # Target Data
    y_train = self.y.loc[train_idx[0]]
    y_val = self.y.loc[val_idx[0]]
    y_test = self.y.loc[test_idx[0]]

    res_x = {"train" : x_train, "val" : x_val, "test" : x_test}
    res_y = {"train" : y_train, "val" : y_val, "test" : y_test}
    res["X"] = res_x
    res["y"] = res_y

    return res
  
  @staticmethod
  def create_samples(sequence: np.array, label: np.array, n_steps: int) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    sequence : np.array
      time series representing a training, val or test set to split in sub sequences
    label : np.array
      time series containing the labels (target) 
    n_steps :  int
      Length of the sequence considered in input in the LSTM 
    Returns
    -------
    np.vstack(X), np.vstack(y) : Tuple[np.array, np.array]
      Containers of observations (each obs is a sequence of length 'n_steps')                           
    """
    # split a univariate sequence into samples (modified internet version)
    X, y = list(), list()
    for i in range(len(sequence)):
      # find the end of this pattern
      end_ix = i + n_steps
      # check if we are beyond the sequence
      if end_ix > len(sequence)-1:
        break
      # gather input and output parts of the pattern
      seq_x, seq_y = sequence[i:end_ix], label[end_ix - 1]
      X.append(seq_x)
      y.append(seq_y)
    return np.vstack(X), np.vstack(y)
    
  def shape_data(self, res:  Dict[str, Dict[str, pd.Series]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Parameters
    ----------
    res : Dict[str, Dict[str, pd.Series]]
      Global dictionary which contain relevant sets of data like x_train, x_val, etc.
    Returns
    -------
    shaped_data : Dict[str, Dict[str, np.ndarray]]
      Returns a global dictionary containing the training, val and test set shaped to be passed in the LSTM
    """
    shaped_data = {}
    n_steps = self.model_parameters["n_steps"]
    # Create observation of length 'n_steps'
    trainX, trainY = LongShortTermMemory.create_samples(res["X"]["train"].values, res["y"]["train"].values, n_steps)
    validX, validY = LongShortTermMemory.create_samples(res["X"]["val"].values, res["y"]["val"].values, n_steps)
    testX,  testY = LongShortTermMemory.create_samples(res["X"]["test"].values,  res["y"]["test"].values, n_steps)
    # reshape input to be [samples, time steps, features]
    n_features = 1
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], n_features))
    validX = validX.reshape((validX.shape[0], validX.shape[1], n_features))
    testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))

    sh_x = {"train" : trainX, "val" : validX, "test" : testX}
    sh_y = {"train" : trainY, "val" : validY, "test" : testY}
    shaped_data["X"] = sh_x
    shaped_data["y"] = sh_y

    return shaped_data
    
  
  def fit(self,  shaped_data: Dict[str, Dict[str, np.ndarray]]):
    """
    Fit the LSTM model on the shaped data
    Parameters
    ----------
    shaped_data : Dict[str, Dict[str, np.ndarray]]
      global dictionary containing the training, val and test set shaped to be passed in the LSTM
    """
    # Retrieve model hyperparameters and other
    n_steps = self.model_parameters["n_steps"]

    path_model = self.model_parameters["path_model"]

    early_stopp_patience = self.model_parameters["early_stopp_patience"]
    early_stopp = callbacks.EarlyStopping(monitor = 'val_loss', patience = early_stopp_patience)

    opt = self.model_parameters["optimizer"]
    metrics = self.model_parameters["metrics"]
    batch_size = self.model_parameters["batch_size"]
    epochs = self.model_parameters["epochs"]
    dropout = self.model_parameters["dropout"]
    recurrent_dropout = self.model_parameters["recurrent_dropout"]
    n_neurons = self.model_parameters["n_neurons"]
    kernel_init = self.model_parameters["kernel_init"]

    n_features = 1

    # create the LSTM network
    self.model = models.Sequential()
    self.model.add(layers.LSTM(n_neurons, input_shape=(n_steps, n_features), kernel_initializer = kernel_init, 
                        recurrent_dropout = recurrent_dropout))
    self.model.add(layers.Dropout(dropout))
    self.model.add(layers.Dense(1))

    # Compile the model
    self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)
    # Fit the model
    self.h_callback = self.model.fit(shaped_data["X"]["train"], shaped_data["y"]["train"], epochs=epochs, batch_size=batch_size, 
                            verbose=2, validation_data = (shaped_data["X"]["val"], shaped_data["y"]["val"]), 
                            callbacks = [early_stopp])
    
      
  def load_lstm_model(self, path_model: str):
    """Load a trained model from a given path"""
    # load json and create model
    json_file = open(f"{path_model}model_{self.symbol}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.model = models.model_from_json(loaded_model_json)

    # load weights into new model
    self.model.load_weights(f"{path_model}weights_{self.symbol}.h5")
    print("Loaded model from disk")
  
  def save_lstm_model(self, path_model: str):
    """Save a trained model on a given path"""
    # serialize model to JSON
    model_json = self.model.to_json()
    with open(f"{path_model}model_{self.symbol}.json", "w") as json_file:
      json_file.write(model_json)

    # serialize weights to HDF5
    self.model.save_weights(f"{path_model}weights_{self.symbol}.h5")
    print("Saved model to disk")

  def predict(self, testX: np.ndarray, threshold: float = 0.5):
    """
    Predict the label for a given number of samples contained in testX
    Parameters
    ----------
    testX : np.ndarray
      Test data shaped as mentionned in the shape_data method
    threshold : float
      Set the threshold from which we make predictions in the LSTM
    """
    assert self.model is not None, "You must fit the model or load one before trying to make predictions"
    # Prediction of the model on the overall test set
    testPredict = self.model.predict(testX)
    # Prediction of the model
    y_pred = np.where(testPredict > threshold, 1, 0)
    return y_pred

  
  def plot_metrics(self):
    """
    Display the learning and validation curves to monitor the training process and avoid overfitting
    """
    assert self.h_callback is not None, "You must fit the model before trying to print the results"

    plt.plot(self.h_callback.history['loss'])
    plt.plot(self.h_callback.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()