# Data Mining and Machine Learning 2020
# Group Number 2    ->  [Raz Malka | Shoham Yamin | Raz Itzhak Afriat]
# Research Subject  ->  One-Class SVMs for Document Classification

# This file is meant to set a graphical user interface,
# along with its handlers and widget template.

import tkinter as tk

def hello(str):
    print(str);

def createWindow(width, height):
    # Basic Setup and Creation
    window = tk.Tk();
    window.geometry(str(width) + "x" + str(height));
    window.wm_resizable(False, False);
    window.wm_title("One-Class SVMs for Document Classification - G2");

    # Menu Configuration
    menubar = tk.Menu(window)
    menubar.add_command(label="Hello!", command=lambda: hello("hi"))
    menubar.add_command(label="Quit", command=window.quit)
    window.config(menu=menubar);

    # Pre-Definition of Initial Window
    # ...
    # Here will be initial main frame set-up, with its content and event handlers

    # Additional Settings
    greeting = tk.Label(text="Hello, Tkinter")
    greeting.pack()

    # Return Instance
    return window;


def main():
    window = createWindow(1080, 720);
    window.mainloop()

if __name__ == "__main__":
    main()