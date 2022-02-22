import sys
import logging
import numpy as np
from optparse import OptionParser
from time import time
from matplotlib import pyplot as plot

#--------------------------------------------------------------------------------

def ex_1(descr=False):

    if descr:
        return """ 
Exercise 1: Plot 2D views of the iris dataset
==============================================
  Plot a simple scatter plot of 2 features of the iris dataset.
  Note that more elaborate visualization of this dataset is detailed
  in the :ref:`statistics` chapter.
"""
    from sklearn.datasets import load_iris

    # Load the data
    iris = load_iris()

    # The indices of the features that we are plotting
    x_index = 1
    y_index = 3

    # this formatter will label the colorbar with the correct target names
    formatter = plot.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plot.figure(figsize=(5, 4))
    plot.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
    plot.colorbar(ticks=[0, 1, 2], format=formatter)
    plot.xlabel(iris.feature_names[x_index])
    plot.ylabel(iris.feature_names[y_index])
    
    plot.tight_layout()
    plot.show()

#--------------------------------------------------------------------------------

def ex_2(descr=False):

    if descr:
        return """
Exercise 2: Introduction to scikit-learn estimator object
==========================================================
  Demonstrate simple fitting of data using LinearRegression model
"""
    #from sklearn.linear_model import LinearRegression
    #model = LinearRegression(normalize=True)
    #print(model.normalize)
    #print(model)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())

    import numpy as np
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    X = x[:, np.newaxis] # The input data for sklearn is 2D: (samples == 3 x features == 1)
    X
    model.fit(X, y)
    print(model)

#--------------------------------------------------------------------------------

def ex_3(descr=False):

    if descr:
        return """
Exercise 3: Simple Linear Regression exercise
=============================================
  Plot a basic example of fitting using Linear Regresssion
"""
    from sklearn.linear_model import LinearRegression

    # x from 0 to 30
    x = 30 * np.random.random((20, 1))

    # y = a*x + b with noise
    y = 0.5 * x + 1.0 + np.random.normal(size=x.shape)

    # create a linear regression model
    model = LinearRegression()
    model.fit(x, y)

    # predict y from the data
    x_new = np.linspace(0, 30, 100)
    y_new = model.predict(x_new[:, np.newaxis])

    # plot the results
    plot.figure(figsize=(4, 3))
    ax = plot.axes()
    ax.scatter(x, y)
    ax.plot(x_new, y_new)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('tight')

    plot.show()

#--------------------------------------------------------------------------------

def ex_4(descr=False):

    if descr:
        return """
Exercise 4: K Nearest-neighbour prediciton on Iris data
=====================================================
  Plot the decision boundary of nearest neighbor decision on iris, 
  first with a single nearest neighbor, and then using 3 nearest 
  neighbors.
"""

    from sklearn import neighbors, datasets
    from matplotlib.colors import ListedColormap
    iris = datasets.load_iris()

    A, b = iris.data, iris.target
    knnAb = neighbors.KNeighborsClassifier(n_neighbors=1)
    knnAb.fit(A,b)

    print("What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?")
    print(iris.target_names[knnAb.predict([[3, 5, 4, 2]])])

    # Create color maps for 3-class classification problem, as with iris
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    X = iris.data[:, :2]  # we only take the first two features. We could
                          # avoid this ugly slicing by using a two-dim dataset
    y = iris.target
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)

    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plot.figure()
    plot.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plot.xlabel('sepal length (cm)')
    plot.ylabel('sepal width (cm)')
    plot.axis('tight')
    plot.show()

    #And now, redo the analysis with 3 neighbors
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plot.figure()
    plot.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plot.xlabel('sepal length (cm)')
    plot.ylabel('sepal width (cm)')
    plot.axis('tight')

    plot.show()


#--------------------------------------------------------------------------------

def ex_5(descr=False):

    if descr:
        return """
Exercise 5: Scatter plot with labels
====================================
  Show a scatter plot of all countries ranked by overall happiness. The
  legend has all countries and their colour. Uses the 'seaborn' module.
"""

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    
    df = pd.read_csv('data\\2019.csv')

    print(df)

    sns.scatterplot(data = df, 
        x = "GDP per capita", 
        y = "Score", 
        hue = "Country or region", 
        size = "Freedom to make life choices")

    plt.show()


    # Do a 3D plot
    sns.set(style = "darkgrid")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    x = df['Score']
    y = df['GDP per capita']
    z = df['Healthy life expectancy']

    ax.set_xlabel("Happiness")
    ax.set_ylabel("Per Capita GDP")
    ax.set_zlabel("Life Expectancy")

    ax.scatter(x, y, z)

    plt.show()


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def is_interactive():
    return not hasattr(sys.modules["__main__"], "__file__")
#--------------------------------------------------------------------------------

op = OptionParser()
op.add_option(
    "-n", "--number",
    action="store",
    type="int",
    dest="exercise_number",
    help="Run the specified exercise.",
)

op.add_option(
    "-l", "--list",
    action="store_true",
    dest="list_exercises",
    help="List the available exercises with a description for each."
)

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)

if len(args) > 0:
    op.error("This script takes no arguments.")
    sys.exit(1)

if(opts.list_exercises):
    # Need to declare the variables here, otherwise locals() will cause a dictionary 
    # error due to new variables being added to the dictionary
    # The invocation of a function can be done 2 ways given it's string value, namely;
    #
    # import foo
    # result = getattr(foo, 'bar')()
    #
    # Or, for a local function bar(),
    # result = locals()["bar"]()

    ex = None
    item = None
    for ex in filter(lambda item: item.startswith("ex_"), locals()):
        print(locals()[ex](True))

elif(opts.exercise_number):
    locals()["ex_" + str(opts.exercise_number)]()

else:
    argv = ["-h"];
    op.parse_args(argv)

