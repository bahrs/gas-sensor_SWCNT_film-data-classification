import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import LSTM, Dense
from tf.keras.optimizers import Adam
from tf.keras.callbacks import EarlyStopping

import numpy as np
from sklearn.metrics import mean_squared_error, f1_score


def make_LSTM(X, y, n_LSTM_layers: int = 1, n_units: int = 64, dropout: float = 0, optimizer: tf.keras.optimizers = Adam, learning_rate: float = 0.001, loss: str = 'mean_squared_error'):
    input_shape = ((X[0]).shape[1], X[0].shape[2])
    output_shape = y[0].shape[1]
    to_return_sequences = lambda n_LSTM_layers_left: True if n_LSTM_layers_left > 1 else False

    model = Sequential()
    for n_LSTM_layers_left in range(n_LSTM_layers, 0, -1):
        model.add(LSTM(units=n_units, 
                       return_sequences=to_return_sequences(n_LSTM_layers_left), 
                       input_shape=input_shape, 
                       dropout = dropout, 
                       recurrent_dropout=dropout))
        n_units = n_units // 2 if n_units > 8 else n_units
    model.add(Dense(units=output_shape))

    model.compile(loss=loss, optimizer=optimizer(learning_rate=learning_rate), weighted_metrics = [])

    return model

def fit_LSTM(model, X_train, y_train, X_test, y_test, sw_train, sw_test, epochs: int = 100, batch_size: int = 32, return_history: bool = False):
    patience = 30
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.1, restore_best_weights=True)]
    history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        #validation_data=(X_test, y_test, sw_test),
        #sample_weight = sw_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        callbacks = callbacks,
        shuffle = False,
        #workers = 32,
        #use_multiprocessing = True
        )
    #test_loss = model.evaluate(X_test, y_test, sample_weight = sw_test, batch_size = batch_size)
    y_pred = model.predict(X_test, batch_size = batch_size, verbose = 0,
                           #workers = 16, use_multiprocessing = True,
                           )
    test_rmse = mean_squared_error(y_test, y_pred, squared = False, multioutput = 'uniform_average')
    #test_rmse = mean_squared_error(y_test, y_pred, sample_weight= sw_test, squared = False, multioutput = 'uniform_average')
    ### taking history into account and trying to minimize the gap between train and validation loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    mse_epochs = [(x-y)**2 for x, y in zip(train_loss, val_loss)]
    history_slice = slice(-patience,len(mse_epochs),1)# slice(-3*patience//2,-patience//2,1)  # apparently it doesn't bug out if len(mse_epochs) < 2* patience
    historical_loss_rmse = np.sqrt(np.nanmean(mse_epochs[history_slice]))  # this slice was chosen as it contains the most optimized hyperparameters
    # here a new metric - historical_loss_rmse was introduced, that shows how far validation and training loss at the last patience epochs are

    hist_test_weights = [10,0]  # completely arbitrary parameter that sets up a weighted sum of traditional RMSE and HLRMSE with weights 2 and 1 respectively
    test_loss = (np.array([test_rmse, historical_loss_rmse]) * np.array(hist_test_weights)).sum() / sum(hist_test_weights)
    if return_history:
        return test_loss, history.history
    else:
        return test_loss, None
