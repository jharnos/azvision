import tkinter as tk
from tkinter import ttk, messagebox
import os

class SettingsDialog:
    def __init__(self, parent, settings_manager):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings Manager")
        self.dialog.resizable(False, False)
        self.settings_manager = settings_manager
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Create main frame
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Save settings section
        save_frame = ttk.LabelFrame(main_frame, text="Save Settings", padding="5")
        save_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        ttk.Label(save_frame, text="Settings Name:").grid(row=0, column=0, sticky="w", padx=5)
        self.save_entry = ttk.Entry(save_frame, width=30)
        self.save_entry.grid(row=0, column=1, padx=5)
        self.save_entry.insert(0, "settings.json")
        
        ttk.Button(save_frame, text="Save", command=self.save_settings).grid(row=0, column=2, padx=5)
        
        # Load settings section
        load_frame = ttk.LabelFrame(main_frame, text="Load Settings", padding="5")
        load_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        # Create listbox for saved settings
        self.settings_listbox = tk.Listbox(load_frame, height=5, width=40)
        self.settings_listbox.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(load_frame, orient="vertical", command=self.settings_listbox.yview)
        scrollbar.grid(row=0, column=2, sticky="ns")
        self.settings_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Load button
        ttk.Button(load_frame, text="Load Selected", command=self.load_settings).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Delete button
        ttk.Button(load_frame, text="Delete Selected", command=self.delete_settings).grid(row=2, column=0, columnspan=2, pady=5)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=self.dialog.destroy).grid(row=2, column=0, pady=10)
        
        # Populate settings list
        self.refresh_settings_list()
        
        # Center the dialog on the parent window
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")

    def refresh_settings_list(self):
        """Refresh the list of saved settings"""
        self.settings_listbox.delete(0, tk.END)
        for settings_file in self.settings_manager.get_saved_settings():
            self.settings_listbox.insert(tk.END, settings_file)

    def save_settings(self):
        """Save current settings"""
        filename = self.save_entry.get().strip()
        if not filename:
            messagebox.showerror("Error", "Please enter a settings name")
            return
            
        if not filename.endswith('.json'):
            filename += '.json'
            
        if self.settings_manager.save_settings(filename):
            self.refresh_settings_list()

    def load_settings(self):
        """Load selected settings"""
        selection = self.settings_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a settings file to load")
            return
            
        filename = self.settings_listbox.get(selection[0])
        self.settings_manager.load_settings(filename)

    def delete_settings(self):
        """Delete selected settings file"""
        selection = self.settings_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a settings file to delete")
            return
            
        filename = self.settings_listbox.get(selection[0])
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {filename}?"):
            try:
                os.remove(os.path.join(self.settings_manager.settings_dir, filename))
                self.refresh_settings_list()
                messagebox.showinfo("Success", f"Deleted {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete settings: {str(e)}") 