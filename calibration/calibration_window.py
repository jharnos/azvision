import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import traceback

class CalibrationWindow:
    def __init__(self, parent, inches_per_pixel, on_calibration_complete):
        try:
            print("\nCalibrationWindow Initialization Debug:")
            print(f"Parent type: {type(parent)}")
            print(f"Parent has frame_buffer: {hasattr(parent, 'frame_buffer')}")
            if hasattr(parent, 'frame_buffer'):
                print(f"Frame buffer length: {len(parent.frame_buffer) if parent.frame_buffer else 0}")
                if parent.frame_buffer:
                    print(f"Latest frame shape: {parent.frame_buffer[-1].shape if hasattr(parent.frame_buffer[-1], 'shape') else 'No shape'}")
            
            # Create window with parent's master as the parent
            self.window = tk.Toplevel(parent.master)
            self.window.title("Camera Calibration")
            self.window.geometry("400x500")
            
            # Center the window on screen
            self.window.geometry("+%d+%d" % (
                parent.master.winfo_rootx() + 50,
                parent.master.winfo_rooty() + 50))
            
            # Store parameters
            self.parent = parent
            self.inches_per_pixel = inches_per_pixel
            self.on_calibration_complete = on_calibration_complete
            self.calibration_points = []
            self.picker_window = None
            
            # Create main frame with padding
            main_frame = ttk.Frame(self.window, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Current scale display
            scale_frame = ttk.LabelFrame(main_frame, text="Current Scale", padding="5")
            scale_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.scale_label = ttk.Label(scale_frame, 
                                       text=f"1 pixel = {inches_per_pixel.get():.6f} inches")
            self.scale_label.pack()
            
            # Calibration controls
            cal_frame = ttk.LabelFrame(main_frame, text="New Calibration", padding="5")
            cal_frame.pack(fill=tk.BOTH, expand=True)
            
            # Known distance input
            dist_frame = ttk.Frame(cal_frame)
            dist_frame.pack(fill=tk.X, pady=5)
            ttk.Label(dist_frame, text="Known Distance:").pack(side=tk.LEFT)
            self.known_distance = tk.DoubleVar(value=1.0)
            ttk.Entry(dist_frame, textvariable=self.known_distance, 
                     width=10).pack(side=tk.LEFT, padx=5)
            ttk.Label(dist_frame, text="inches").pack(side=tk.LEFT)
            
            # Instructions
            instruction_text = ("To calibrate:\n\n"
                              "1. Place an object with a known dimension in the camera view\n"
                              "2. Enter that dimension in inches above\n"
                              "3. Click 'Start Measurement'\n"
                              "4. Click the start and end points of your known dimension\n"
                              "5. The scale will be automatically calculated")
            
            ttk.Label(cal_frame, text=instruction_text, 
                     justify=tk.LEFT, wraplength=350).pack(pady=10)
            
            # Buttons
            btn_frame = ttk.Frame(cal_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(btn_frame, text="Start Measurement",
                      command=self.start_calibration).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Reset to Default",
                      command=self.reset_calibration).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Close",
                      command=self.close_window).pack(side=tk.RIGHT, padx=5)
            
            # Status
            self.status_label = ttk.Label(cal_frame, text="")
            self.status_label.pack(pady=5)
            
            # Make window modal
            self.window.transient(parent.master)
            self.window.grab_set()
            self.window.focus_set()
            
            # Bind window close event
            self.window.protocol("WM_DELETE_WINDOW", self.close_window)
            
            print("CalibrationWindow initialization completed successfully")
            
        except Exception as e:
            print(f"Error in CalibrationWindow initialization: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise the exception to be caught by the caller

    def start_calibration(self):
        """Start the calibration process"""
        print("\nCalibration Debug:")
        print(f"Parent type: {type(self.parent)}")
        print(f"Has frame_buffer attribute: {hasattr(self.parent, 'frame_buffer')}")
        if hasattr(self.parent, 'frame_buffer'):
            print(f"Frame buffer length: {len(self.parent.frame_buffer) if self.parent.frame_buffer else 0}")
            print(f"Frame buffer type: {type(self.parent.frame_buffer)}")
            if self.parent.frame_buffer:
                print(f"Latest frame shape: {self.parent.frame_buffer[-1].shape if hasattr(self.parent.frame_buffer[-1], 'shape') else 'No shape'}")
        
        if not hasattr(self.parent, 'frame_buffer') or not self.parent.frame_buffer:
            messagebox.showerror("Error", "No camera feed available")
            return
        
        self.show_calibration_picker()

    def show_calibration_picker(self):
        """Show the calibration picker window"""
        print("\nCalibration Picker Debug:")
        print(f"Parent type: {type(self.parent)}")
        print(f"Has frame_buffer attribute: {hasattr(self.parent, 'frame_buffer')}")
        if hasattr(self.parent, 'frame_buffer'):
            print(f"Frame buffer length: {len(self.parent.frame_buffer) if self.parent.frame_buffer else 0}")
            if self.parent.frame_buffer:
                print(f"Latest frame shape: {self.parent.frame_buffer[-1].shape if hasattr(self.parent.frame_buffer[-1], 'shape') else 'No shape'}")
        
        if self.picker_window:
            try:
                self.picker_window.close()
            except:
                pass
        
        # Create new picker window with the correct parent window and pass the app instance
        self.picker_window = CalibrationPicker(self.window,  # Use self.window as parent
                                             self.parent,    # Pass the app instance
                                             self.known_distance.get(),
                                             self.on_calibration_done)
        
        # Position picker window next to main calibration window
        self.picker_window.window.geometry("+%d+%d" % (
            self.window.winfo_rootx() + self.window.winfo_width() + 10,
            self.window.winfo_rooty()))
        
        # Ensure picker window is visible and focused
        self.picker_window.window.lift()
        self.picker_window.window.focus_force()

    def on_calibration_done(self, new_scale):
        if new_scale is not None:
            self.inches_per_pixel.set(new_scale)
            self.scale_label.config(text=f"1 pixel = {new_scale:.6f} inches")
            self.status_label.config(text="Calibration successful!")
            self.on_calibration_complete(new_scale)

    def reset_calibration(self):
        self.inches_per_pixel.set(0.04)  # Default value
        self.scale_label.config(text=f"1 pixel = 0.040000 inches")
        self.status_label.config(text="Reset to default scale")
        self.on_calibration_complete(0.04)

    def close_window(self):
        if self.picker_window:
            try:
                self.picker_window.close()
            except:
                pass
        self.window.grab_release()
        self.window.destroy()

class CalibrationPicker:
    def __init__(self, parent, app_instance, known_distance, callback):
        self.window = tk.Toplevel(parent)
        self.window.title("Measure Known Distance")
        
        # Get screen dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        self.known_distance = known_distance
        self.callback = callback
        self.points = []
        
        # Get current frame from camera using the app instance
        frame = app_instance.frame_buffer[-1].copy()
        height, width = frame.shape[:2]
        
        # Calculate preview size to be 80% of screen height while maintaining aspect ratio
        max_preview_height = int(screen_height * 0.8)
        max_preview_width = int(screen_width * 0.8)
        
        # Calculate preview dimensions maintaining aspect ratio
        preview_height = max_preview_height
        preview_width = int(width * (preview_height / height))
        
        # If width is too large, scale down based on width instead
        if preview_width > max_preview_width:
            preview_width = max_preview_width
            preview_height = int(height * (preview_width / width))
        
        # Resize frame for preview
        preview_frame = cv2.resize(frame, (preview_width, preview_height))
        
        # Draw measurement guide
        guide_frame = preview_frame.copy()
        cv2.putText(guide_frame, 
                   f"Click start and end points of {known_distance} inch dimension",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        preview_image = Image.fromarray(cv2.cvtColor(guide_frame, cv2.COLOR_BGR2RGB))
        self.imgtk = ImageTk.PhotoImage(image=preview_image)
        
        # Create main frame with minimal padding
        main_frame = ttk.Frame(self.window, padding="2")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions with smaller font
        self.instruction_label = ttk.Label(
            main_frame,
            text=f"Click the start point of your {known_distance} inch measurement",
            font=('Arial', 8)
        )
        self.instruction_label.pack(pady=(2,0))
        
        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, 
                              width=preview_width, 
                              height=preview_height,
                              bg='black')
        self.canvas.pack(pady=2)
        
        self.canvas.create_image(0, 0, anchor="nw", image=self.imgtk)
        
        # Button frame immediately after canvas
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=(0,2))
        
        # Reset and Cancel buttons
        ttk.Button(btn_frame, 
                  text="Reset Points",
                  command=self.reset_points).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, 
                  text="Cancel",
                  command=self.cancel).pack(side=tk.LEFT, padx=2)
        
        # Store original dimensions for scaling
        self.orig_width = width
        self.orig_height = height
        self.preview_width = preview_width
        self.preview_height = preview_height
        
        # Bind click event
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Make window modal
        self.window.transient(parent)
        self.window.grab_set()
        self.window.focus_force()
        
        # Center window on screen with minimal padding
        window_width = preview_width + 10
        window_height = preview_height + 50
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Bind window close event
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

    def reset_points(self):
        """Reset all points and clear drawings"""
        self.points = []
        self.canvas.delete("point", "line", "label")
        self.instruction_label.config(
            text=f"Click the start point of your {self.known_distance} inch measurement"
        )

    def on_click(self, event):
        # Map preview coordinates to original image coordinates
        x = event.x * (self.orig_width / self.preview_width)
        y = event.y * (self.orig_height / self.preview_height)
        
        self.points.append((x, y))
        
        # Draw point with smaller radius
        radius = 3
        self.canvas.create_oval(
            event.x-radius, event.y-radius,
            event.x+radius, event.y+radius,
            fill='red',
            outline='white',
            width=1,
            tags="point"
        )
        
        if len(self.points) == 1:
            self.instruction_label.config(
                text=f"Click the end point of your {self.known_distance} inch measurement"
            )
            # Draw start point label
            self.canvas.create_text(
                event.x + 8, event.y + 8,
                text="Start",
                fill='red',
                font=('Arial', 6),
                tags="label"
            )
        elif len(self.points) == 2:
            # Draw end point label
            self.canvas.create_text(
                event.x + 8, event.y + 8,
                text="End",
                fill='red',
                font=('Arial', 6),
                tags="label"
            )
            
            # Draw line between points in preview
            start_x = self.points[0][0] * (self.preview_width / self.orig_width)
            start_y = self.points[0][1] * (self.preview_height / self.orig_height)
            self.canvas.create_line(
                start_x, start_y,
                event.x, event.y,
                fill='red',
                width=3,
                tags="line"
            )
            
            # Calculate scale using original image coordinates
            dx = self.points[1][0] - self.points[0][0]
            dy = self.points[1][1] - self.points[0][1]
            pixel_distance = np.sqrt(dx*dx + dy*dy)
            
            # Calculate new scale (inches per pixel)
            new_scale = self.known_distance / pixel_distance
            
            print("\nCalibration Debug:")
            print(f"Point 1: ({self.points[0][0]:.2f}, {self.points[0][1]:.2f})")
            print(f"Point 2: ({self.points[1][0]:.2f}, {self.points[1][1]:.2f})")
            print(f"Pixel distance: {pixel_distance:.2f}")
            print(f"Known distance: {self.known_distance:.2f} inches")
            print(f"Calculated scale: {new_scale:.6f} inches/pixel")
            
            # Add verification calculation
            expected_distance = pixel_distance * new_scale
            print(f"Verification - measured distance: {expected_distance:.2f} inches")
            
            # Update instruction label to show measurement
            self.instruction_label.config(
                text=f"Measured {pixel_distance:.1f} pixels = {self.known_distance:.2f} inches\n"
                     f"Click 'Reset Points' to try again or close window to accept"
            )
            
            # Store the scale factor
            self.current_scale = new_scale

    def cancel(self):
        self.callback(None)
        self.close()

    def close(self):
        if hasattr(self, 'current_scale'):
            self.callback(self.current_scale)
        self.window.grab_release()
        self.window.destroy() 