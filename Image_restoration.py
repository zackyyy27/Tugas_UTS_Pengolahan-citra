import cv2
import numpy as np
from scipy.signal import wiener
from scipy.fft import fft2, ifft2
from tkinter import filedialog, Tk, Button, Label, Scale, HORIZONTAL
from PIL import Image, ImageTk, ImageEnhance

# Global variables to store the original and processed images
original_image = None
processed_image = None

# Function to load image
def load_image():
    global img_path, original_image
    img_path = filedialog.askopenfilename()
    if img_path:
        original_image = cv2.imread(img_path)
        display_image(original_image, 'Original Image', 3, 0)

# Function to display image
def display_image(img_array, title, row, column):
    global processed_image
    processed_image = img_array  # Store the current processed image for saving
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img  # Keep a reference!
    panel.grid(row=row, column=column, columnspan=2)
    panel_text = Label(root, text=title)
    panel_text.grid(row=row-1, column=column, columnspan=2)

# Regularized inverse filter to reduce noise amplification
def regularized_inverse_filter(image, kernel_size, regularization_param=0.01):
    restored_channels = []
    for channel in cv2.split(image):
        kernel = np.zeros((abs(kernel_size), abs(kernel_size)))
        kernel[int((abs(kernel_size) - 1) / 2), :] = np.ones(abs(kernel_size))
        kernel /= abs(kernel_size)

        image_fft = fft2(channel)
        kernel_fft = fft2(kernel, s=channel.shape)
        
        inverse_kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + regularization_param)
        restored_fft = image_fft * inverse_kernel_fft
        restored_channel = np.abs(ifft2(restored_fft))
        restored_channels.append(np.clip(restored_channel, 0, 255).astype(np.uint8))
    
    return cv2.merge(restored_channels)

# Function to apply Regularized Inverse Filter
def apply_inverse_filter():
    if original_image is None:
        return
    kernel_size = kernel_slider.get()
    restored_image = regularized_inverse_filter(original_image, kernel_size)
    display_image(restored_image, 'Regularized Inverse Filtered Image', 3, 2)

# Function for Wiener Filtering per color channel
def apply_wiener_filter():
    if original_image is None:
        return
    noise_power = noise_slider.get() / 100
    restored_channels = [np.clip(wiener(channel, (5, 5), noise=noise_power), 0, 255).astype(np.uint8)
                         for channel in cv2.split(original_image)]
    restored_image = cv2.merge(restored_channels)
    display_image(restored_image, 'Wiener Filtered Image', 3, 2)

# Function to sharpen image after filtering
def sharpen_image(img_array):
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Sharpness(img)
    enhanced_image = enhancer.enhance(2.0)  # Adjust sharpness level as needed
    return np.array(enhanced_image)

# Function to auto-restore with optimal settings
def auto_restore():
    if original_image is None:
        return
    noise_power = 10 / 100
    restored_channels = [np.clip(wiener(channel, (5, 5), noise=noise_power), 0, 255).astype(np.uint8)
                         for channel in cv2.split(original_image)]
    restored_image = cv2.merge(restored_channels)
    final_image = sharpen_image(restored_image)
    display_image(final_image, 'Auto Restored Image', 3, 2)

# Function to save the processed image
def save_image():
    if processed_image is None:
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", ".png"), ("JPEG files", ".jpg")])
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

# GUI Tkinter Setup
root = Tk()
root.title("Color Image Restoration Application")

# Button to load image
load_btn = Button(root, text="Load Image", command=load_image)
load_btn.grid(row=0, column=0, padx=10, pady=10)

# Slider for kernel size in Inverse Filter (-21 to 21)
kernel_slider = Scale(root, from_=-21, to=21, orient=HORIZONTAL, label="Kernel Size (Inverse Filter)")
kernel_slider.set(15)
kernel_slider.grid(row=1, column=0, padx=5, pady=5)

# Button to apply Inverse Filter
inverse_btn = Button(root, text="Apply Regularized Inverse Filter", command=apply_inverse_filter)
inverse_btn.grid(row=0, column=1, padx=10, pady=10)

# Slider for noise power in Wiener Filter
noise_slider = Scale(root, from_=1, to=100, orient=HORIZONTAL, label="Noise Power (Wiener Filter)")
noise_slider.set(10)
noise_slider.grid(row=1, column=1, padx=5, pady=5)

# Button to apply Wiener Filter
wiener_btn = Button(root, text="Apply Wiener Filter", command=apply_wiener_filter)
wiener_btn.grid(row=0, column=2, padx=10, pady=10)

# Button to enhance sharpness
sharpen_btn = Button(root, text="Enhance Sharpness", command=lambda: display_image(sharpen_image(processed_image), 'Enhanced Sharpness', 3, 2))
sharpen_btn.grid(row=0, column=3, padx=10, pady=10)

# Button to auto-restore image with recommended settings
auto_restore_btn = Button(root, text="Auto Restore", command=auto_restore)
auto_restore_btn.grid(row=1, column=2, padx=10, pady=10)

# Button to save the processed image
save_btn = Button(root, text="Save Image", command=save_image)
save_btn.grid(row=1, column=3, padx=10, pady=10)

root.mainloop()