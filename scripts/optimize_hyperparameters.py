###
### SUPER RAW DRAFT VERSION
###


def objective_LSTM(trial):
    # Load_data
    data_load_params = dict(
        dedrifting_method = trial.suggest_categorical('dedrifting_method', ['SavGol', 'exp', "none"]), # 'SavGol', 
        )
    if data_load_params['dedrifting_method'] == 'SavGol':
        data_load_params['window_length'] = trial.suggest_int('window_length', 150, 500, step = 10)
        data_load_params['envelope_choice'] = trial.suggest_categorical('envelope_choice', ['multienv', 'topenv'])
        data_load_params['alpha'] = 1
    elif data_load_params['dedrifting_method'] == 'exp':
        data_load_params['alpha'] = trial.suggest_float('alpha', 0.001, 0.1, log = True)
        data_load_params['window_length'] = 1
        data_load_params['envelope_choice'] = trial.suggest_categorical('envelope_choice', ['multienv', 'topenv'])
    elif data_load_params['dedrifting_method'] == 'none':
        data_load_params['envelope_choice'] = 'none'
        data_load_params['window_length'] = 1
        data_load_params['alpha'] = 1
    
    params = dict(
        look_back = trial.suggest_int('look_back', 20, 57, log=False),
        n_components = trial.suggest_int('n_components', 15, 150),
        do_PCA = True,  # trial.suggest_categorical('do_PCA', [True, False]),
        n_LSTM_layers = 1,  # trial.suggest_int('n_LSTM_layers', 1, 2),
        n_units = trial.suggest_int('n_units', 16, 128, log=False),  # trial.suggest_categorical('n_units', [16, 32, 64, 96, 128]),  
        dropout = trial.suggest_float('dropout', 0.005, 0.5, log=False),  # 0.162
        learning_rate= trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        epochs = 150,  # trial.suggest_int('epochs', 50, 300),
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256]),
        )
    

    DF = load_full_dedrifted_dataset(**data_load_params)
    # Split data and perform PCA
    train_X, test_X, train_y, test_y, train_SW, test_SW = train_test_RNN(
        DF, 
        look_back= params['look_back'],
        n_components= params['n_components'],
        do_PCA= params['do_PCA'],
        start=8)
    model = make_LSTM(train_X, 
                      train_y, 
                      optimizer = Adam, 
                      loss = 'mean_squared_error',
                      n_LSTM_layers = params['n_LSTM_layers'],
                      n_units = params['n_units'],
                      dropout = params['dropout'],
                      learning_rate = params['learning_rate'],)
    # train the model
    #mse = []
    #pruning_flag = 0
    #for X_train, X_test, y_train, y_test in zip(train_X[::-1], test_X[::-1], train_y[::-1], test_y[::-1]):  # reversed the order so the the trial be pruned earlier and more consistent
    rmse_, history_ = fit_LSTM(
        model, train_X[0], train_y[0], test_X[0], test_y[0], 
        train_SW, test_SW,
        epochs = params['epochs'],
        batch_size= params['batch_size'], 
        return_history=True)
    plot_history(history_, params)
    
    return rmse_

def objective_CBR(trial):
    # Load_data
    data_load_params = dict(dedrifting_method= trial.suggest_categorical('dedrifting_method', ['exp', 'none']),  # 'SavGol', 
                            envelope_choice = trial.suggest_categorical('envelope_choice', ['multienv', 'topenv']),
                            )
    if data_load_params['dedrifting_method'] == 'SavGol':
        data_load_params['window_length'] = trial.suggest_int('window_length', 150, 400, step = 10)
    elif data_load_params['dedrifting_method'] == 'exp':
        data_load_params['alpha'] = trial.suggest_float('alpha', 0.001, 0.1, log = True)
    
    DF = load_full_dedrifted_dataset(**data_load_params)
    # Split data and perform PCA
    train_X, test_X, train_y, test_y = train_test_TS(DF, n_components=trial.suggest_int('n_components', 5, 150), start = 6)
    # train the model
    cb_params = {
        'iterations': trial.suggest_int('iterations', 100, 1800, step = 50),
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log = True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 25)
    }
    catboost_model = CatBoostRegressor(**cb_params, loss_function='MultiRMSE', verbose=0)
    pruning_callback = CatBoostPruningCallback(trial, "MultiRMSE")
    mse = []
    #pruning_flag = 0
    for X_train, X_test, y_train, y_test in zip(train_X[::-1], test_X[::-1], train_y[::-1], test_y[::-1]):  # reversed the order so the the trial be pruned earlier and more consistent
        catboost_model.fit(X_train, y_train, 
                           eval_set=[(X_test, y_test)], 
                           early_stopping_rounds=200, # need to implement good pruning by starting from the most information train|test pair and not calculate the whole sequence if this pair is pruned
                           callbacks=[pruning_callback],
                           verbose=0)
        pruning_callback.check_pruned()
        y_pred = catboost_model.predict(X_test)  # add native penalization (sample_weights) according to TLV_TWA values for gases, NO2 - 0.2, H2S - 1, Acet - 
        mse_ = mean_squared_error(y_test, y_pred, multioutput='raw_values', squared = False) # substitute this with native CatBoost stuff
        TLV_weighted_mean = np.mean([x/y for x, y in zip(mse_,[0.2, 1, 250])])  # Assuming column order to be NO2 H2S Acetone
        mse.append(TLV_weighted_mean) 
    return np.mean(mse_)  # np.mean(mse)

def objective_CBC(trial):
    # Load_data
    data_load_params = dict(dedrifting_method= trial.suggest_categorical('dedrifting_method', ['SavGol', 'exp', None]),
                            envelope_choice = trial.suggest_categorical('envelope_choice', ['multienv', 'topenv']),
                            )
    if data_load_params['dedrifting_method'] == 'SavGol':
        data_load_params['window_length'] = trial.suggest_int('window_length', 150, 400, step = 10)
        data_load_params['alpha'] = 1
    elif data_load_params['dedrifting_method'] == 'exp':
        data_load_params['alpha'] = trial.suggest_float('alpha', 0.001, 0.1, log = True)
    DF = load_full_dedrifted_dataset(**data_load_params)
    # Split data and perform PCA
    train_X, test_X, train_y, test_y = train_test_TS_class(add_class_column(DF), trial.suggest_int('n_components', 5, 200), start = 7)
    # train the model
    cb_params = {
        'iterations': trial.suggest_int('iterations', 100, 2000, step = 50),
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.7, log = True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 15)
    }
    
    catboost_model = CatBoostClassifier(**cb_params, loss_function="MultiClass", eval_metric="Accuracy", use_best_model=True, silent=True)
    pruning_callback = CatBoostPruningCallback(trial, "MultiClass")
    mse = []
    #pruning_flag = 0
    f1_macro_scores = []
    for X_train, X_test, y_train, y_test in zip(train_X[::-1], test_X[::-1], train_y[::-1], test_y[::-1]):  # reversed the order so the the trial be pruned earlier and more consistent
        catboost_model.fit(X_train, y_train, 
                           eval_set=[(X_test, y_test)], 
                           early_stopping_rounds=200, # need to implement good pruning by starting from the most information train|test pair and not calculate the whole sequence if this pair is pruned
                           callbacks=[pruning_callback],
                           verbose=0)
        pruning_callback.check_pruned()
        '''if not pruning_flag:
            pruning_callback.check_pruned()
            pruning_flag += 1
            if trial.should_prune():
                break'''
        y_pred = catboost_model.predict(X_test)  # add native penalization (sample_weights) according to TLV_TWA values for gases, NO2 - 0.2, H2S - 1, Acet - 
        f1_macro_scores.append(f1_score(y_test, y_pred, average='macro'))
    
    return np.mean(f1_macro_scores)  # np.mean(mse)

import warnings
import optuna
warnings.filterwarnings("ignore")
study = optuna.create_study(direction= 'minimize',  # minimize  for regression minimize  for classification 'maximize'
                            pruner= optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1, n_min_trials=1),
                            #optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', reduction_factor=3, min_early_stopping_rate=4, bootstrap_count=0) 
                            # optuna.pruners.MedianPruner(n_warmup_steps=10),
                            )
study.optimize(objective_LSTM, n_trials=10000, timeout=8*60*60, show_progress_bar = False, n_jobs=-1)  #n_jobs=-1

# Print the optimized hyperparameters and their values
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
    
    
optuna.visualization.plot_param_importances(study).update_layout(height = 300, margin = dict(t=50, l=20, r=10, b=20)).show()
optuna.visualization.plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration").update_layout(height = 300, 
                                                                                                                                      margin = dict(t=50, l=20, r=10, b=20)).show()
optuna.visualization.plot_slice(study).show() 