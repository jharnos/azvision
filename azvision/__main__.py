import tkinter as tk
from azvision.gui.main_app import CNCVisionApp

def main():
    root = tk.Tk()
    app = CNCVisionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 