"""
    Brief Description:
        1. plot helper function to support stk_price_modeler, stk_price_predictor...

    ToDo's:
        1.

    Date: 2023/7/28
    Ver.: 0.1d
    Author: maoyi.fan@gmail.com
    Reference:

    Revision History:
        v. 0.1d: newly created
"""
import matplotlib.pyplot as plt


def hello():
    print("You've found plot_functions.py!")


def plot_series(x, y, format='-', start=0, end=None, title=None,
                xlabel=None, ylabel=None, legend=None):
    """
    Visualizes time series data

    Args:
      x (array of int) - contains values for the x-axis
      y (array of int or tuple of arrays) - contains the values for the y-axis
      format (string) - line style when plotting the graph
      start (int) - first time step to plot
      end (int) - last time step to plot
      title (string) - title of the plot
      xlabel (string) - label for the x-axis
      ylabel (string) - label for the y-axis
      legend (list of strings) - legend for the plot
    """

    # Setup figure dimension
    plt.figure(figsize=(5, 3))

    # Check if there are more than two series to plot
    if type(y) is tuple:
        for y_curr in y:
            # plot the x and current y values
            plt.plot(x[start:end], y_curr[start:end], format)
    else:
        plt.plot(x[start:end], y[start:end], format)

    # label the x-axis
    plt.xlabel(xlabel)

    # label the y-axis
    plt.ylabel(ylabel)

    # Set the legend
    if legend:
        plt.legend(legend)

    # Set the title
    plt.title(title)

    # Overlay a grid over the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()

    return
