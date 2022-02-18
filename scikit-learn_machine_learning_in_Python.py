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
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
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

