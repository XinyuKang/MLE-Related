# General Machine Learning
## [What's the trade-off between bias and variance?](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
1. Bias is an error from wrong assumptions about the data, e.g. assuming data is linear when in reality is complex. Model with high bias pays very little attention to the training data and oversimplifies the model. Leads to high error on training and test data.
2. Variance is an error from high sensitivity to small fluctuations in the training set. Model with high variance pays too much attention to the training data and does not generalize on the data it hasn't seen before. Thus performs very well on the training data but very bad on the test data.
3. Underfitting heppens when a model is unable to capture the underlying pattern of the data (high bias and low variance). Happens when training set is too small or training model is too simple.
4. Overfitting happens when model captures the noise along with the underlying pattern in data (low bias and high variance). Happens when train model a lot over noisy dataset or model is too complex.
5. Why is tradeoff? Simple model (few parameters) -> high bias and low variance. Complx model -> high variance and low bias. Thus need good balance without overfitting or underfitting. An optimal balance of bias and variance would never overfit or undefit the model.

# Computer Vision
