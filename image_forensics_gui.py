import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from algorithms import preprocess_image, apply_glcm, apply_dct, apply_dwt, apply_lbp, apply_sobel

# Function to load an image from file
def load_image():
    """
    Opens a file dialog to select an image file.
    Loads the image using OpenCV and converts it to RGB format.
    Returns the loaded image as a NumPy array.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency
            return image
        else:
            messagebox.showerror("Error", "Failed to load image.")
            return None
    return None

# Function to display image in Tkinter label
def display_image(image, label, max_size=(400, 400)):
    """
    Displays the image in the given Tkinter label, resizing to fit max_size.
    """
    if image is None:
        label.config(image='')
        return

    # Convert to PIL Image
    if len(image.shape) == 3:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(image).convert('L')  # Grayscale

    # Resize to fit
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to PhotoImage
    photo = ImageTk.PhotoImage(pil_image)
    label.config(image=photo)
    label.image = photo  # Keep reference

# Main GUI class
class ImageForensicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Forensics Tool")

        # Variables
        self.original_image = None
        self.preprocessed_image = None

        # GUI Elements
        # Image Upload
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=10, pady=10)

        # Image Display
        self.image_label = tk.Label(root, text="No image loaded")
        self.image_label.grid(row=0, column=1, rowspan=4, padx=10, pady=10)

        self.preprocessed_label = tk.Label(root, text="Preprocessed image")
        self.preprocessed_label.grid(row=0, column=2, rowspan=2, padx=10, pady=10)

        # YCbCr channels
        self.ycbcr_y_label = tk.Label(root, text="Y Channel")
        self.ycbcr_y_label.grid(row=2, column=2, padx=10, pady=10)

        self.ycbcr_cb_label = tk.Label(root, text="Cb Channel")
        self.ycbcr_cb_label.grid(row=3, column=2, padx=10, pady=10)

        self.ycbcr_cr_label = tk.Label(root, text="Cr Channel")
        self.ycbcr_cr_label.grid(row=4, column=2, padx=10, pady=10)

        # Preprocessing Section
        tk.Label(root, text="Preprocessing:").grid(row=1, column=0, sticky='w', padx=10)
        self.conversion_var = tk.StringVar(value='none')
        tk.Radiobutton(root, text="None", variable=self.conversion_var, value='none').grid(row=2, column=0, sticky='w', padx=20)
        tk.Radiobutton(root, text="Grayscale", variable=self.conversion_var, value='grayscale').grid(row=3, column=0, sticky='w', padx=20)
        tk.Radiobutton(root, text="YCbCr", variable=self.conversion_var, value='ycbcr').grid(row=4, column=0, sticky='w', padx=20)

        tk.Label(root, text="Resize Width:").grid(row=5, column=0, sticky='w', padx=10)
        self.resize_width = tk.Entry(root)
        self.resize_width.insert(0, "256")
        self.resize_width.grid(row=5, column=1, sticky='w', padx=10)

        tk.Label(root, text="Resize Height:").grid(row=6, column=0, sticky='w', padx=10)
        self.resize_height = tk.Entry(root)
        self.resize_height.insert(0, "256")
        self.resize_height.grid(row=6, column=1, sticky='w', padx=10)

        # Algorithm Selection
        tk.Label(root, text="Select Algorithm:").grid(row=7, column=0, sticky='w', padx=10)
        self.algorithm_var = tk.StringVar(value='glcm')
        algorithms = ['GLCM', 'DCT', 'DWT', 'LBP', 'Sobel']
        self.algorithm_menu = tk.OptionMenu(root, self.algorithm_var, *algorithms)
        self.algorithm_menu.grid(row=7, column=1, sticky='w', padx=10)

        # Process Button
        self.preprocess_btn = tk.Button(root, text="Preprocess", command=self.preprocess_only)
        self.preprocess_btn.grid(row=8, column=0, pady=10)

        self.process_btn = tk.Button(root, text="Process", command=self.process_image)
        self.process_btn.grid(row=8, column=1, pady=10)

        # Output Section
        tk.Label(root, text="Processed Image:").grid(row=9, column=0, sticky='w', padx=10)
        self.processed_label = tk.Label(root, text="No processed image")
        self.processed_label.grid(row=9, column=1, columnspan=2, padx=10, pady=10)

        tk.Label(root, text="Features/Values:").grid(row=10, column=0, sticky='w', padx=10)
        self.values_text = tk.Text(root, height=5, width=50)
        self.values_text.grid(row=10, column=1, padx=10, pady=10)

        tk.Label(root, text="Log:").grid(row=11, column=0, sticky='w', padx=10)
        self.log_text = tk.Text(root, height=5, width=50)
        self.log_text.grid(row=11, column=1, padx=10, pady=10)

    def upload_image(self):
        """
        Handles image upload, displays it, preprocesses automatically, and logs the action.
        """
        self.original_image = load_image()
        if self.original_image is not None:
            display_image(self.original_image, self.image_label)
            # Automatically preprocess and display
            conversion = self.conversion_var.get()
            try:
                width = int(self.resize_width.get())
                height = int(self.resize_height.get())
            except ValueError:
                width = 256
                height = 256
            
            if conversion == 'ycbcr':
                # Display YCbCr channels separately
                self.preprocessed_image = preprocess_image(self.original_image, conversion, width, height)
                # Split channels
                y, cr, cb = cv2.split(self.preprocessed_image)
                display_image(y, self.preprocessed_label)
                display_image(cb, self.ycbcr_cb_label)
                display_image(cr, self.ycbcr_cr_label)
            else:
                self.preprocessed_image = preprocess_image(self.original_image, conversion, width, height)
                display_image(self.preprocessed_image, self.preprocessed_label)
                # Clear YCbCr labels
                self.ycbcr_y_label.config(image='')
                self.ycbcr_cb_label.config(image='')
                self.ycbcr_cr_label.config(image='')
            
            self.log("Image loaded and preprocessed.")
        else:
            self.log("Failed to load image.")

    def process_image(self):
        """
        Applies preprocessing, then the selected algorithm, displays results, and logs.
        """
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        # Get preprocessing options
        conversion = self.conversion_var.get()
        try:
            width = int(self.resize_width.get())
            height = int(self.resize_height.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid resize dimensions.")
            return

        # Preprocess
        self.preprocessed_image = preprocess_image(self.original_image, conversion, width, height)
        
        # Display YCbCr channels if selected
        if conversion == 'ycbcr':
            y, cr, cb = cv2.split(self.preprocessed_image)
            display_image(y, self.preprocessed_label)
            display_image(cb, self.ycbcr_cb_label)
            display_image(cr, self.ycbcr_cr_label)
        else:
            display_image(self.preprocessed_image, self.preprocessed_label)
            # Clear YCbCr labels
            self.ycbcr_y_label.config(image='')
            self.ycbcr_cb_label.config(image='')
            self.ycbcr_cr_label.config(image='')
        
        self.log("Preprocessing applied.")

        # Get algorithm
        algorithm = self.algorithm_var.get()

        # Apply algorithm
        if algorithm == 'GLCM':
            features, glcm = apply_glcm(self.preprocessed_image)
            if features:
                self.display_features(features)
                # For GLCM, display the GLCM as image? But it's 256x256, maybe not.
                # Just show features
                self.processed_label.config(text="GLCM computed (see values)")
                self.log("GLCM algorithm executed.")
            else:
                messagebox.showerror("Error", glcm)
        elif algorithm == 'DCT':
            dct = apply_dct(self.preprocessed_image)
            if dct is not None:
                # Display log-scaled DCT
                dct_display = np.log(np.abs(dct) + 1)
                dct_display = cv2.normalize(dct_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                display_image(dct_display, self.processed_label)
                self.values_text.delete(1.0, tk.END)
                self.log("DCT algorithm executed.")
        elif algorithm == 'DWT':
            LL, LH, HL, HH = apply_dwt(self.preprocessed_image)
            # Display LL as example
            LL_display = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display_image(LL_display, self.processed_label)
            self.values_text.delete(1.0, tk.END)
            self.values_text.insert(tk.END, f"Subbands shapes: LL {LL.shape}, LH {LH.shape}, HL {HL.shape}, HH {HH.shape}")
            self.log("DWT algorithm executed.")
        elif algorithm == 'LBP':
            lbp, hist = apply_lbp(self.preprocessed_image)
            lbp_display = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display_image(lbp_display, self.processed_label)
            self.values_text.delete(1.0, tk.END)
            self.values_text.insert(tk.END, "LBP Histogram:\n")
            for i, count in enumerate(hist):
                self.values_text.insert(tk.END, f"Pattern {i}: {count}\n")
            self.log("LBP algorithm executed.")
        elif algorithm == 'Sobel':
            sobel, features = apply_sobel(self.preprocessed_image)
            if sobel is not None:
                display_image(sobel, self.processed_label)
                self.display_features(features)
                self.log("Sobel algorithm executed.")
            else:
                messagebox.showerror("Error", "Failed to apply Sobel.")

    def preprocess_only(self):
        """
        Applies preprocessing to the image and displays the preprocessed image.
        """
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        # Get preprocessing options
        conversion = self.conversion_var.get()
        try:
            width = int(self.resize_width.get())
            height = int(self.resize_height.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid resize dimensions.")
            return

        # Preprocess
        self.preprocessed_image = preprocess_image(self.original_image, conversion, width, height)
        
        # Display YCbCr channels separately if selected
        if conversion == 'ycbcr':
            y, cr, cb = cv2.split(self.preprocessed_image)
            display_image(y, self.preprocessed_label)
            display_image(cb, self.ycbcr_cb_label)
            display_image(cr, self.ycbcr_cr_label)
        else:
            display_image(self.preprocessed_image, self.preprocessed_label)
            # Clear YCbCr labels
            self.ycbcr_y_label.config(image='')
            self.ycbcr_cb_label.config(image='')
            self.ycbcr_cr_label.config(image='')
        
        self.values_text.delete(1.0, tk.END)
        self.log("Preprocessing applied and displayed.")

    def display_features(self, features):
        """
        Displays the extracted features in the values text box.
        """
        self.values_text.delete(1.0, tk.END)
        for key, value in features.items():
            self.values_text.insert(tk.END, f"{key}: {value:.4f}\n")

    def log(self, message):
        """
        Adds a message to the log text box.
        """
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageForensicsApp(root)
    root.mainloop()