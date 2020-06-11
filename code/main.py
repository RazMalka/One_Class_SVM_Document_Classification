# Data Mining and Machine Learning 2020
# Group Number 2    ->  [Raz Malka | Shoham Yamin | Raz Itzhak Afriat]
# Research Subject  ->  One-Class SVMs for Document Classification

# This file is meant to set a graphical user interface,
# along with its handlers and widget template.

# -------------------------------------------------------------------
# Imports
import const
import window

def main():
    width = const.root_width ; height = const.root_height
    # Create Root Window
    root = window.createWindow(width, height, const.root_title, None)
    
    # Configure Root Window
    window.configureRootMenu(root)
    window.configureRootContent(root, width, height)

    # Display Root Window
    root.mainloop()

if __name__ == "__main__":
    main()