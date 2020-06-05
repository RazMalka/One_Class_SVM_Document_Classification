# Data Mining and Machine Learning 2020
# Group Number 2    ->  [Raz Malka | Shoham Yamin | Raz Itzhak Afriat]
# Research Subject  ->  One-Class SVMs for Document Classification

# This file is meant to provide implementation of graphical figures

# GUI
import tkinter as tk; from tkinter import ttk

# MatplotLib and Embedding with tkinter
import matplotlib; matplotlib.use("TkAgg");
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure; import matplotlib.pyplot as plt;

figure = canvas = toolbar = None

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

def exampleFigure(rightFrame: tk.Frame, item_xpos: int, item_ypos: int):
    global figure, canvas, toolbar

    # Figure
    figure = Figure(figsize=(5,5), dpi=100)
    a = figure.add_subplot(111)

    a.plot([5,6,7,8],[8,9,3,5], 'ro', label="point A")
    a.plot([1,2,3,4],[5,6,1,3], 'go', label="point B")
    a.plot([5,6],[8,9], 'bx', label="point C")
    a.legend(loc="upper left")

    canvas = FigureCanvasTkAgg(figure, rightFrame)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, rightFrame)
    toolbar.config(bg="white")
    toolbar.update()
    toolbar._message_label.config(bg="white")

    canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True, padx=item_xpos, pady=item_ypos)
    canvas._tkcanvas.pack(side="top", fill="both", expand=True, padx=item_xpos, pady=item_ypos)