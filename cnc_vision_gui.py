import cv2
import numpy as np
import ezdxf
import tkinter as tk
from tkinter import filedialog, messagebox

# === GUI Application ===

class CNCVisionApp:
    def __init__(self, master):
        self.master = master
        master.title("CNC Table Image to DXF")
        master.geometry("400x300")

        # Default values
        self.image_path = None
        self.inches_per_pixel = tk.DoubleVar(value=0.04)

        # Widgets
        tk.Label(master, text="Inches per Pixel:").pack(pady=5)
        self.scale_entry = tk.Entry(master, textvariable=self.inches_per_pixel)
        self.scale_entry.pack(pady=5)

        self.select_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.process_button = tk.Button(master, text="Generate DXF", command=self.process_image)
        self.process_button.pack(pady=10)

        self.status_label = tk.Label(master, text="", fg="blue")
        self.status_label.pack(pady=10)

    def select_image(self):
        filetypes = (("JPEG files", "*.jpg *.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
        path = filedialog.askopenfilename(title="Select Edge Detected Image", filetypes=filetypes)
        if path:
            self.image_path = path
            self.status_label.config(text=f"Selected: {path.split('/')[-1]}")

    def process_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        inches_per_pixel = self.inches_per_pixel.get()
        if inches_per_pixel <= 0:
            messagebox.showerror("Error", "Inches per pixel must be greater than 0.")
            return

        try:
            # Process the image
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                messagebox.showwarning("No Contours", "No shapes were found in the image.")
                return

            # Create DXF
            doc = ezdxf.new()
            msp = doc.modelspace()

            for contour in contours:
                if len(contour) < 5:
                    continue

                points = []
                for point in contour:
                    x_pixel, y_pixel = point[0]
                    x_real = x_pixel * inches_per_pixel
                    y_real = y_pixel * inches_per_pixel
                    points.append((x_real, y_real))

                msp.add_lwpolyline(points, close=True)

            # Save DXF
            output_path = filedialog.asksaveasfilename(defaultextension=".dxf", filetypes=[("DXF files", "*.dxf")])
            if output_path:
                doc.saveas(output_path)
                messagebox.showinfo("Success", f"DXF saved to:\n{output_path}")
                self.status_label.config(text="DXF generation completed!")

        except Exception as e:
            messagebox.showerror("Error", str(e))

# === Run the app ===

if __name__ == "__main__":
    root = tk.Tk()
    app = CNCVisionApp(root)
    root.mainloop()
