# Data Mining and Machine Learning 2020
# Group Number 2    ->  [Raz Malka | Shoham Yamin | Raz Itzhak Afriat]
# Research Subject  ->  One-Class SVMs for Document Classification

# This file is meant to set a graphical user interface,
# along with its handlers and widget template.

import tkinter as tk
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib as mp
import webbrowser as browser

def aboutWindow(root):
    about = createWindow(400, 180, "About", root)

    # About Window Content
    title = tk.Label(master=about, text="\nOne-Class SVM Document Classification\nVersion 0.01")
    repository_link = tk.Label(master=about, text = "Open Source Repository", fg="Blue", cursor="hand2")
    repository_link.bind("<Button-1>", lambda e: browser.open_new("https://github.com/RazMalka/SVM-DC"))
    footer = tk.Label(master=about, text="MIT License © 2020\n\nRaz Malka\tShoham Yamin\tRaz Itzhak Afriat")

    # Pack Content into About Window
    title.pack()
    repository_link.pack()
    footer.pack()

    # Display About Window
    about.mainloop()

def testFunc(str):
    print(str)

def destroyAll(root):
    for widget in root.winfo_children():
        if (isinstance(widget, tk.Toplevel)):
            widget.destroy()
    root.destroy()

def createWindow(width, height, title, root):
    # Create Window Instance and compute position and dimensions
    if root == None:
        window = tk.Tk()
    else:
        window = tk.Toplevel(root)
    geometry_settings = "%dx%d+%d+%d" % (width, height, (window.winfo_screenwidth()-width)/2, (window.winfo_screenheight()-height)/2)

    # Basic Window Settings
    window.geometry(geometry_settings)
    window.wm_resizable(False, False)
    window.wm_title(title)

    # Return Instance
    return window

def configureRootMenu(root):
    # -------------------------------------------------------------------
    # Menu Configuration
    menu = tk.Menu(root, tearoff=False)
    options_menu = tk.Menu(menu, tearoff=False)
    help_menu = tk.Menu(menu, tearoff=False)

    menu.add_cascade(label="Options", menu=options_menu)
    menu.add_cascade(label="Help", menu=help_menu)

    # Menu Items
    options_menu.add_command(label="Test", command=lambda: testFunc("test"))
    options_menu.add_command(label="Quit", command=lambda: destroyAll(root))

    help_menu.add_command(label="Help", command=lambda: testFunc("test"))
    help_menu.add_separator()
    help_menu.add_command(label="About", command=lambda: aboutWindow(root))
    
    root.config(menu=menu)

def configureRootContent(root):
    # Pre-Definition of Initial Window
    # ...
    # Here will be initial main frame set-up, with its content and event handlers
    # Including graph panes, buttons, frames etc.

    # Additional Settings
    greeting = tk.Label(master=root, text="Hello, SVM")
    greeting.pack()

def main():
    # Create Root Window
    root = createWindow(1080, 720, "One-Class SVMs for Document Classification - G2", None)

    # Configure Root Window
    configureRootMenu(root)
    configureRootContent(root)

    # Display Root Window
    root.mainloop()

if __name__ == "__main__":
    main()