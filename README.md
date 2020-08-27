### Real Life Applications of SVM
Cancer Detection - Character Recognition

Support Vector Machine (SVM) is a supervised machine learning algorithm involving classification, regression or outliers detection. SVM is effective in high dimensionality and also in cases where the number of dimensions is greater than the number of samples. It is also memory efficient by using support vectors as a subset of training points.
In this program, I bring two simple examples to introduce SVM and it's applications in real life.

In the first example, Cancer Detection, by making a classifier we can predict whether the cancer is malignant or benign, which as a target 0 represents malignant, and 1 represents benign. For generating the model first import the SVM module from sklearn to create a support vector classifier in svc() by passing the argument kernel as the linear kernel. Then for evaluating the model, we can predict how accurate the model or classifier is by calculating the accuracy score, recall, and precision.

In Character Recognition example, we can use the existing digit data set and train the classifier. Next, by using the classifier we can predict a digit and plot the image to be more distinct.