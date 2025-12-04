###
### DRAFT VERSION 0.0.0.0.1
###

params = dict(
    dedrifting_method = 'SavGol',
    envelope_choice = 'multienv',  # trial.suggest_categorical('dedrifting_method', ['multienv', 'topenv', 'None']),  #'multienv',  # 
    window_length = 250, #trial.suggest_int('window_length', 100, 700, step = 10),
    alpha = 0.022,
    look_back = 35, #trial.suggest_int('look_back', 5, 250),
    n_components = 110, # trial.suggest_int('n_components', 2, 70),
    do_PCA = True, # trial.suggest_categorical('do_PCA', [True, False]),
    n_LSTM_layers = 1, #trial.suggest_int('n_LSTM_layers', 1, 3),
    n_units = 35, # trial.suggest_int('n_units', 4, 256, log=True),
    dropout = 0.2, # trial.suggest_float('dropout', 0.05, 0.5, log=True),
    learning_rate= 0.02, #trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
    epochs = 200, #trial.suggest_int('epochs', 50, 300),
    batch_size = 128, #trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256]),
    )
DF = load_full_dedrifted_dataset(dedrifting_method=params['dedrifting_method'], envelope_choice=params['envelope_choice'], window_length=params['window_length'])
# Split data and perform PCA
train_X, test_X, train_y, test_y, train_SW, test_SW = train_test_RNN(DF, 
                                                    look_back= params['look_back'],
                                                    n_components= params['n_components'],
                                                    do_PCA= params['do_PCA'])
model = make_LSTM(train_X, 
                    train_y, 
                    optimizer = Adam, 
                    loss = 'mean_squared_error',
                    n_LSTM_layers = params['n_LSTM_layers'],
                    n_units = params['n_units'],
                    dropout = params['dropout'],
                    learning_rate = params['learning_rate'],)
# train the model
mse = []
#for X_train, X_test, y_train, y_test in zip(train_X[::-1], test_X[::-1], train_y[::-1], test_y[::-1]):  # reversed the order so the the trial be pruned earlier and more consistent
mse_, history_ = fit_LSTM(
    model, train_X[0], train_y[0], test_X[0], test_y[0], train_SW, test_SW,
    epochs = params['epochs'],
    batch_size= params['batch_size'], 
    return_history=True)
plot_history(history_, params)
#mse.append(mse_) 
print(mse_)