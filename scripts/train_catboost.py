from catboost import CatBoostClassifier, Pool, CatBoost, CatBoostRegressor
import matplotlib as plt


###
### LOAD DATA
###


model = CatBoostClassifier(
    iterations= 1300,
    depth=6,
    learning_rate=0.00937,
    l2_leaf_reg=13,
    loss_function="MultiClass",
    eval_metric="Accuracy",
)
f1_macro_scores = []
for i in range(len(train_X)):
    model.fit(
        train_X[i], train_y[i], 
        eval_set = (test_X[i], test_y[i]), 
        use_best_model=True,
        silent=True,
        plot=False)
    
    predictions = model.predict(test_X[i])
    f1_macro_scores.append(f1_score(test_y[i], predictions, average='macro'))

    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    labels = [r"$acetone$", r'$H_{2}S$', r'$NO_{2}$', r'$air$']
    fig = ConfusionMatrixDisplay.from_estimator(estimator = model, X = test_X[i], y = test_y[i], normalize = 'pred', xticks_rotation='horizontal', 
                                                ax = ax, cmap = 'viridis',
                                                display_labels=labels)
    
    # Set title with F1 score and number of measurement cycles
    title = f"Confusion Matrix \nF1 Score: {f1_score(test_y[i], predictions, average='macro'):.2f}\n meas_cycle {(5+i)}"
    
    ax.set_title(title, fontsize=10)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.show()