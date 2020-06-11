# Data Mining and Machine Learning 2020
# Group Number 2    ->  [Raz Malka | Shoham Yamin | Raz Itzhak Afriat]
# Research Subject  ->  One-Class SVMs for Document Classification

# This file is meant to provide implementation of graphical figures

# GUI
import tkinter as tk; from tkinter import ttk

# MatplotLib and Embedding with tkinter
import matplotlib; matplotlib.use("TkAgg");
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure; import matplotlib.pyplot as plt; import matplotlib.font_manager
from sklearn.svm import OneClassSVM; import numpy as np

figure = canvas = toolbar = None

def createFigure(rightFrame: tk.Frame, item_xpos: int, item_ypos: int):
    global figure, canvas, toolbar

    # Destroy Previous Figure
    canvas.get_tk_widget().destroy()
    canvas._tkcanvas.destroy()
    toolbar.destroy()
    figure.clf()

    # Define "classifiers" to be used
    classifiers = {"One-Class SVM": OneClassSVM(nu=0.25, kernel="rbf", gamma=0.35)}
    colors = ['m', 'g', 'b']; legend1 = {}; legend2 = {}

    # Get data
    x_train = np.array([[2,2],[2,2.5],[2,3], [2,3.5], [3, 1.5], [2.5, 1.5], [2,3], [7,6]]) # two clusters
    x_test  = np.array([[1.5,1.5], [5,5]])

    # Learn a frontier for outlier detection with several classifiers
    xx1, yy1 = np.meshgrid(np.linspace(0, 6, 500), np.linspace(1, 4.5, 500))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        figure = plt.figure(1)
        y_train = clf.fit_predict(x_train)
        y_test = clf.predict(x_test)
        Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
        Z1 = Z1.reshape(xx1.shape)
        legend1[clf_name] = plt.contour(
            xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])

    legend1_values_list = list(legend1.values())
    legend1_keys_list = list(legend1.keys())

    # Plot the results (= shape of the data points cloud)
    plt.figure(1)  # two clusters
    plt.title("Outlier detection on a real data set")
    plt.scatter(x_train[:, 0], x_train[:, 1], color='black')
    plt.scatter(x_test[:, 0], x_test[:, 1], color='blue')
    bbox_args = dict(boxstyle="round", fc="0.8")
    arrow_args = dict(arrowstyle="->")
    plt.annotate("outlying points", xy=(6, 2),
                xycoords="data", textcoords="data",
                xytext=(0, 0.4), bbox=bbox_args, arrowprops=arrow_args)
    plt.xlim((xx1.min(), xx1.max() + 2))
    plt.ylim((yy1.min(), yy1.max() + 2))
    #plt.xlim(0, 8)
    #plt.ylim(0, 8)
    plt.legend(([legend1_values_list[0].collections[0]]),   #FIX
            ([legend1_keys_list[0]]), #FIX
            loc="upper left",
            prop=matplotlib.font_manager.FontProperties(size=11))
    plt.ylabel("Y Axis"); plt.xlabel("X Axis")
    canvas = FigureCanvasTkAgg(figure, rightFrame); canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, rightFrame)
    toolbar.config(bg="white")
    toolbar.update()
    toolbar._message_label.config(bg="white")

    canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True, padx=item_xpos, pady=item_ypos)
    canvas._tkcanvas.pack(side="top", fill="both", expand=True, padx=item_xpos, pady=item_ypos)
"""
def createFigure(rightFrame: tk.Frame, item_xpos: int, item_ypos: int):
    global figure, canvas, toolbar

    # Destroy Previous Figure
    canvas.get_tk_widget().destroy()
    canvas._tkcanvas.destroy()
    toolbar.destroy()

    # Figure
    figure = Figure(figsize=(5,5), dpi=100)
    a = figure.add_subplot(111)
    a.plot([11,8], [8,9], 'ro', label="BBB")
    a.plot([1,4], [5,3], 'go', label="AAA")
    a.plot([11,4], [8,3], 'bx', label="CCC")
    a.legend(loc="upper left")

    canvas = FigureCanvasTkAgg(figure, rightFrame)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, rightFrame)
    toolbar.config(bg="white")
    toolbar.update()
    toolbar._message_label.config(bg="white")
    
    canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True, padx=item_xpos, pady=item_ypos)
    canvas._tkcanvas.pack(side="top", fill="both", expand=True, padx=item_xpos, pady=item_ypos)
"""
def exampleFigure(rightFrame: tk.Frame, item_xpos: int, item_ypos: int):
    global figure, canvas, toolbar

    # Define "classifiers" to be used
    classifiers = {"One-Class SVM": OneClassSVM(nu=0.25, kernel="rbf", gamma=0.35)}
    colors = ['m', 'g', 'b']; legend1 = {}; legend2 = {}

    # Get data
    x_train = np.array([[2,2],[2,2.5],[2,3], [2,3.5], [3, 1.5], [2.5, 1.5], [2,3], [7,6]]) # two clusters
    x_test  = np.array([[1.5,1.5], [5,5]])

    # Learn a frontier for outlier detection with several classifiers
    xx1, yy1 = np.meshgrid(np.linspace(0, 6, 500), np.linspace(1, 4.5, 500))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        figure = plt.figure(1)
        y_train = clf.fit_predict(x_train)
        y_test = clf.predict(x_test)
        Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
        Z1 = Z1.reshape(xx1.shape)
        legend1[clf_name] = plt.contour(
            xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])

    legend1_values_list = list(legend1.values())
    legend1_keys_list = list(legend1.keys())

    # Plot the results (= shape of the data points cloud)
    plt.figure(1)  # two clusters
    plt.title("Outlier detection on a real data set")
    plt.scatter(x_train[:, 0], x_train[:, 1], color='black')
    plt.scatter(x_test[:, 0], x_test[:, 1], color='blue')
    bbox_args = dict(boxstyle="round", fc="0.8")
    arrow_args = dict(arrowstyle="->")
    plt.annotate("outlying points", xy=(6, 2),
                xycoords="data", textcoords="data",
                xytext=(0, 0.4), bbox=bbox_args, arrowprops=arrow_args)
    plt.xlim((xx1.min(), xx1.max() + 2))
    plt.ylim((yy1.min(), yy1.max() + 2))
    #plt.xlim(0, 8)
    #plt.ylim(0, 8)
    plt.legend(([legend1_values_list[0].collections[0]]),   #FIX
            ([legend1_keys_list[0]]), #FIX
            loc="upper left",
            prop=matplotlib.font_manager.FontProperties(size=11))
    plt.ylabel("Y Axis"); plt.xlabel("X Axis")
    canvas = FigureCanvasTkAgg(figure, rightFrame); canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, rightFrame)
    toolbar.config(bg="white")
    toolbar.update()
    toolbar._message_label.config(bg="white")

    canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True, padx=item_xpos, pady=item_ypos)
    canvas._tkcanvas.pack(side="top", fill="both", expand=True, padx=item_xpos, pady=item_ypos)

def main():
    createFigure(None, None, None)

if __name__ == "__main__":
    main()