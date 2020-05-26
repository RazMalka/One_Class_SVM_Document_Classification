# Data Mining and Machine Learning 2020
# Group Number 2    ->  [Raz Malka | Shoham Yamin | Raz Itzhak Afriat]
# Research Subject  ->  One-Class SVMs for Document Classification

# This file is meant to set a graphical user interface,
# along with its handlers and widget template.

# -------------------------------------------------------------------
# Imports%
import tkinter as tk
from tkinter import ttk
import webbrowser as browser

# Data Analysis Libraries
import numpy as np
from scipy import stats

# MatplotLib and Embedding with tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------

def aboutWindow(root: tk.Toplevel):
    about = createWindow(400, 180, "About", root)

    # About Window Content
    title = tk.Label(master=about, text="\nOne-Class SVM Document Classification\nVersion 0.02", bg="white").pack()
    repository_link = tk.Label(master=about, text = "Open Source Repository", fg="Blue", cursor="hand2", bg="white")
    repository_link.bind("<Button-1>", lambda e: browser.open_new("https://github.com/RazMalka/SVM-DC"))
    footer = tk.Label(master=about, text="MIT License © 2020\n\nRaz Malka\tShoham Yamin\tRaz Itzhak Afriat", bg="white")

    # Pack Content into About Window
    repository_link.pack()
    footer.pack()

    # Display About Window
    about.mainloop()

def testFunc(str: str):
    print(str)

def destroyAll(root: tk.Toplevel):
    for widget in root.winfo_children():
        if (isinstance(widget, tk.Toplevel)):
            widget.destroy()
    root.destroy()

def createWindow(width: int, height: int, title: str, root: tk.Toplevel):
    # Create Window Instance and compute position and dimensions
    if root == None: window = tk.Tk()
    else: window = tk.Toplevel(root, bg="white")
    geometry_settings = "%dx%d+%d+%d" % (width, height, (window.winfo_screenwidth()-width)/2, (window.winfo_screenheight()-height)/2)

    # Basic Window Settings
    window.geometry(geometry_settings)
    window.wm_resizable(False, False)
    window.wm_title(title)

    # Return Instance
    return window

def configureRootMenu(root: tk.Toplevel):
    # -------------------------------------------------------------------
    # Menu Configuration
    menu = tk.Menu(root, tearoff=False)
    options_menu = tk.Menu(menu, tearoff=False)
    help_menu = tk.Menu(menu, tearoff=False)

    menu.add_cascade(label="Options", menu=options_menu)
    menu.add_cascade(label="Help", menu=help_menu)

    # -------------------------------------------------------------------
    # Menu Items
    options_menu.add_command(label="Test", command=lambda: testFunc("test"))
    options_menu.add_command(label="Quit", command=lambda: destroyAll(root))

    help_menu.add_command(label="v0.02", command=lambda: testFunc("test"))
    help_menu.entryconfig("v0.02", state="disabled")
    help_menu.add_separator()
    help_menu.add_command(label="About", command=lambda: aboutWindow(root))
    
    root.config(menu=menu, bg="white")

def configureRootContent(root: tk.Toplevel, width: int, height: int):
    # Pre-Definition of Initial Window

    # -------------------------------------------------------------------
    # Left Frame (Control Port)
    leftFrame = tk.Frame(master=root, bg="white")
    label_controlport = tk.Label(text="Control Port", bg="white").place(x=width/16, y=0, in_=leftFrame)

    label_algorithm = tk.Label(text="Algorithm:", bg="white").place(x=width/108,y=height/18, in_=leftFrame)
    combobox_algorithm = ttk.Combobox(width=int(width/54), values=["-", "Scholkopf", "Outlier-SVM"])
    combobox_algorithm.current(0)
    combobox_algorithm.place(x=width/108, y=height/10, in_=leftFrame)

    label_data = tk.Label(text="Data:", bg="white").place(x=width/108,y=height/6, in_=leftFrame)
    combobox_data = ttk.Combobox(width=int(width/54), values=["-", "Document A", "Document B"])
    combobox_data.current(0)
    combobox_data.place(x=width/108, y=height/4.8, in_=leftFrame)
    
    # -------------------------------------------------------------------
    # Right Frame (View Port)
    rightFrame = tk.Frame(master=root, bg="white")
    label_viewport = tk.Label(text="View Port", bg="white").place(x=width/3.25, y=0, in_=rightFrame)

    # Figure
    f = Figure(figsize=(5,5), dpi=100)
    a = f.add_subplot(111)

    a.plot([5,6,7,8],[8,9,3,5], 'ro', label="point A")
    a.plot([1,2,3,4],[5,6,1,3], 'go', label="point B")
    a.plot([5,6],[8,9], 'bx', label="point C")
    a.legend(loc="upper left")

    canvas = FigureCanvasTkAgg(f, rightFrame)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, rightFrame)
    toolbar.config(bg="white")
    toolbar.update()
    toolbar._message_label.config(bg="white")
    
    # -------------------------------------------------------------------
    # Event Handlers
    # ...

    # -------------------------------------------------------------------
    # Packing
    leftFrame.place(x=0, y=0, in_=root)
    rightFrame.place(x=width/4, y=0, in_=root)
    tk.Frame(master=root, height=height, bg="black").place(x=width/4, y=0, in_=root)

    canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True, padx=width/108, pady=height/22)
    canvas._tkcanvas.pack(side="top", fill="both", expand=True, padx=width/108, pady=height/22)

def main():
    width = 720 ; height = 588
    # Create Root Window
    root = createWindow(width, height, "One-Class SVMs for Document Classification - G2", None)
    
    # Configure Root Window
    configureRootMenu(root)
    configureRootContent(root, width, height)

    # Display Root Window
    root.mainloop()

if __name__ == "__main__":
    main()