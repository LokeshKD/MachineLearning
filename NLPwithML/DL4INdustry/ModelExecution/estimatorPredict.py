
## Predicting with TF Estimator for Regression Models.

preds = regressor.predict(input_fn)

###

preds = regressor.predict(
    input_fn,
    predict_keys=['prediction'])


