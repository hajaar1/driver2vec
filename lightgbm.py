# LightGBM classifier

# The following is the setup and the training of the LightGBM classifier.

def get_n_way_accuracy(n_way, test_dataset, train_dataset, model):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': n_way,
        'metric': 'multi_logloss',
        'num_leaves': 32,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'max_depth': 8,
        'num_trees': 30,
        'verbose': -1,
        'min_data_in_leaf': 2,  # May need to change that with a real test set
        'verbose': -1
    }

    model.eval()
    l = [0, 1, 2, 3, 4,5,6,7,8,9]
    accuracies = []
    for driver_list in itt.combinations(l, n_way):

        x_train_classifier, y_train_classifier = train_dataset.get_classifier_data(driver_list,
                                                                                   model)
        x_test_class, y_test_class = test_dataset.get_classifier_data(
            driver_list, model)
        lgb_train = lgbm.Dataset(x_train_classifier, y_train_classifier, params={
                                 'verbose': -1}, free_raw_data=False)

        clf = lgbm.train(params, lgb_train)

        pred = clf.predict(x_test_class)

        binar_pred = [0 if pred[index][truth] <
                      0.5 else 1 for index, truth in enumerate(y_test_class)]
        accuracies.append(np.mean(binar_pred))

    return np.mean(accuracies)
