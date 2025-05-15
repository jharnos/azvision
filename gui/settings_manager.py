import json
import os
from tkinter import messagebox

class SettingsManager:
    def __init__(self, app):
        self.app = app
        self.settings_dir = "settings"
        os.makedirs(self.settings_dir, exist_ok=True)

    def save_settings(self, filename):
        """Save current application settings to a JSON file"""
        try:
            settings = {
                # Edge detection settings
                'inches_per_pixel': self.app.inches_per_pixel.get(),
                'canny_low': self.app.canny_low.get(),
                'canny_high': self.app.canny_high.get(),
                'edge_scale': self.app.edge_scale.get(),
                'dxf_rotation': self.app.dxf_rotation.get(),
                
                # Reference point settings
                'use_reference_point': self.app.use_reference_point.get(),
                'reference_table_x': self.app.reference_table_x.get(),
                'reference_table_y': self.app.reference_table_y.get(),
                
                # Table boundary settings
                'add_table_boundary': self.app.add_table_boundary.get(),
                'table_width': self.app.table_width.get(),
                'table_height': self.app.table_height.get(),
                
                # Camera settings
                'auto_exposure': self.app.auto_exposure.get(),
                'exposure': self.app.exposure_var.get(),
                'brightness': self.app.brightness_var.get(),
                'contrast': self.app.contrast_var.get(),
                
                # Color detection settings
                'color_mode': self.app.color_mode.get(),
                'color_tolerance_h': self.app.color_tolerance_h.get(),
                'color_tolerance_s': self.app.color_tolerance_s.get(),
                'color_tolerance_v': self.app.color_tolerance_v.get(),
                'color_sample_radius': self.app.color_sample_radius.get(),
                
                # Camera selection
                'selected_camera': self.app.selected_camera.get(),
                'selected_resolution': self.app.selected_resolution.get()
            }
            
            # Save reference point if it exists
            if hasattr(self.app, 'reference_point') and self.app.reference_point is not None:
                settings['reference_point'] = self.app.reference_point
            
            filepath = os.path.join(self.settings_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(settings, f, indent=4)
            
            messagebox.showinfo("Success", f"Settings saved to {filename}")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            return False

    def load_settings(self, filename):
        """Load settings from a JSON file"""
        try:
            filepath = os.path.join(self.settings_dir, filename)
            with open(filepath, 'r') as f:
                settings = json.load(f)
            
            # Update application settings
            self.app.inches_per_pixel.set(settings.get('inches_per_pixel', 0.0604))
            self.app.canny_low.set(settings.get('canny_low', 50))
            self.app.canny_high.set(settings.get('canny_high', 150))
            self.app.edge_scale.set(settings.get('edge_scale', 1.0))
            self.app.dxf_rotation.set(settings.get('dxf_rotation', -0.25))
            
            self.app.use_reference_point.set(settings.get('use_reference_point', True))
            self.app.reference_table_x.set(settings.get('reference_table_x', 72.63324))
            self.app.reference_table_y.set(settings.get('reference_table_y', 30.54024))
            
            self.app.add_table_boundary.set(settings.get('add_table_boundary', True))
            self.app.table_width.set(settings.get('table_width', 144.0))
            self.app.table_height.set(settings.get('table_height', 61.0))
            
            self.app.auto_exposure.set(settings.get('auto_exposure', True))
            self.app.exposure_var.set(settings.get('exposure', -5))
            self.app.brightness_var.set(settings.get('brightness', 128))
            self.app.contrast_var.set(settings.get('contrast', 128))
            
            self.app.color_mode.set(settings.get('color_mode', False))
            self.app.color_tolerance_h.set(settings.get('color_tolerance_h', 15))
            self.app.color_tolerance_s.set(settings.get('color_tolerance_s', 100))
            self.app.color_tolerance_v.set(settings.get('color_tolerance_v', 100))
            self.app.color_sample_radius.set(settings.get('color_sample_radius', 2))
            
            # Load reference point if it exists in settings
            if 'reference_point' in settings:
                self.app.reference_point = tuple(settings['reference_point'])
            
            # Update camera settings
            if 'selected_camera' in settings:
                self.app.selected_camera.set(settings['selected_camera'])
            if 'selected_resolution' in settings:
                self.app.selected_resolution.set(settings['selected_resolution'])
            
            # Update camera settings
            self.app.update_camera_settings()
            
            messagebox.showinfo("Success", f"Settings loaded from {filename}")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")
            return False

    def get_saved_settings(self):
        """Get list of saved settings files"""
        try:
            return [f for f in os.listdir(self.settings_dir) if f.endswith('.json')]
        except Exception:
            return [] 