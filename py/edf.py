import numpy as np
import pywt
import os
from pathlib import Path
import tifffile
from skimage.filters import unsharp_mask
import argparse

def wavelet_edf(images):
    n_levels = 2  # Level of wavelet decomposition
    wavelet = 'dmey'  # Wavelet type
    
    # Initialize lists to store the maximum coefficients and the corresponding indices of the selected planes
    max_coefficients = None
    selected_planes = None

    # Variables for the maximum and minimum pixel values across the image stack
    max_pixel_values = None
    min_pixel_values = None
    
    # Process each image in the stack
    for z, img in enumerate(images):
        img = np.asarray(img, dtype=np.float32)

        # Update the max and min pixel values across the stack
        if max_pixel_values is None:
            max_pixel_values = img
            min_pixel_values = img
        else:
            max_pixel_values = np.maximum(max_pixel_values, img)
            min_pixel_values = np.minimum(min_pixel_values, img)

        # Perform wavelet decomposition on the current image
        coeffs = pywt.wavedec2((img), wavelet, level=n_levels)
        
        if max_coefficients is None:
            # Initialize max_coefficients and selected_planes with the structure of coeffs but filled with zeros or appropriate values
            max_coefficients = [coeff if idx == 0 else [np.zeros_like(c) for c in coeff] for idx, coeff in enumerate(coeffs)]
            selected_planes = [np.zeros_like(coeffs[0], dtype=int)] + [[np.zeros_like(c, dtype=int) for c in coeff] for coeff in coeffs[1:]]

        for j, coeff in enumerate(coeffs):
            if j == 0:  # Process approximation coefficients
                coeff_gpu = np.asarray(coeff)
                magnitude = np.abs(coeff_gpu)
                mask = magnitude > np.asarray(max_coefficients[j])
                max_coefficients[j] = np.where(mask, coeff_gpu, max_coefficients[j])
                selected_planes[j] = np.where(mask, z, selected_planes[j])
            else:  # Process detail coefficients
                for theta, sub_coeff in enumerate(coeff):
                    sub_coeff_gpu = np.asarray(sub_coeff)
                    magnitude = np.abs(sub_coeff_gpu)
                    mask = magnitude > np.asarray(max_coefficients[j][theta])
                    max_coefficients[j][theta] = np.where(mask, sub_coeff_gpu, max_coefficients[j][theta])
                    selected_planes[j][theta] = np.where(mask, z, selected_planes[j][theta])

    # Convert max_coefficients and selected_planes to the format expected by pywt for reconstruction
    coeffs_for_recon = [max_coefficients[0]] + [(c[0], c[1], c[2]) for c in max_coefficients[1:]]

    # Reconstruct the image from the modified coefficients
    final_image_cpu = pywt.waverec2(coeffs_for_recon, wavelet)

    # Post-processing: adjust the final image to match the original stack's intensity range
    final_image_cpu = np.clip(final_image_cpu, min_pixel_values.min(), max_pixel_values.max())

    # Post-processing: apply unsharp mask to the final image
    final_image_cpu = unsharp_mask(final_image_cpu, radius=2, amount=2)

    # convert to 16 bit and scale to 0-65535
    final_image_cpu = (final_image_cpu - min_pixel_values.min()) / (max_pixel_values.max() - min_pixel_values.min()) * 65535
    final_image_cpu = final_image_cpu.astype(np.uint16)

    return final_image_cpu
  
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tiff files using wavelet EDF")
    parser.add_argument("--input", action='store', type=str, help="Directory containing tiff files")
    parser.add_argument("--output", help="Output directory")
    args = parser.parse_args()
    directory = Path(args.input.strip().replace("\\", "\\\\"))
    extensions = [".tiff", ".tif"]
    files = list(directory.glob("*"))
    # filter the files to only include tiff files
    files = [file for file in files if file.suffix in extensions]
    for i, file in enumerate(files):
        print(f"Processing {file.stem} [{i + 1} / {len(files)}]")
        img = tifffile.imread(file)
        # split the image along the channel axis
        channels, height, width = img.shape
        # convert to 32 bit float from 16 bit
        img = (img / 65535).astype(np.float32)
        images = np.split(img, channels, axis=0)
        # get the final image
        final = wavelet_edf(images)
        # parent directory
        parent = file.parent if args.output is None else Path(args.output.strip().replace("\\", "\\\\"))
        new_name = file.stem + "_edf" + file.suffix

        tifffile.imwrite(parent / new_name, final)