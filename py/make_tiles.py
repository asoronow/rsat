import os
import uuid
from PIL import Image
import cv2
import keyboard

# Hardcoded input and output directories
input_directory = r"\\128.114.78.227\euiseokdataUCSC_3\Matt Jacobs\images and data\M_brains_3\M677_678\2024-01-05\M678\counting\03 - max\03e - sharp nuclei\\to_tile"
output_directory = r"\\128.114.78.227\euiseokdataUCSC_3\Matt Jacobs\images and data\M_brains_3\M677_678\2024-01-05\M678\counting\03 - max\03e - sharp nuclei\\tiles"
# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def process_image(file_path):
    try:
        # Load 16bit grayscale
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        width, height = img.shape
        # Calculate the number of tiles in both dimensions
        num_tiles_x = width // 640
        num_tiles_y = height // 640
        print(f"Viewing tiles of {file_path}")
        print("Press right arrow to keep, down arrow to discard.")
        tile_counter = 0
        for i in range(0, width, 640):
            for j in range(0, height, 640):
                print(f"Processing tile {tile_counter}/{num_tiles_x*num_tiles_y}")
                tile_counter += 1
                # Ensure the box is within the image bounds
                if i + 640 > width or j + 640 > height:
                    continue
                # Define box to crop
                tile = img[i:i+640, j:j+640]
                cv2.imshow('Tile', tile)                
                # Wait for user action
                key_press = False
                while not key_press:
                    if keyboard.is_pressed('right'):
                        # Save tile
                        tile_path = os.path.join(output_directory, f'tile__{uuid.uuid4()}_{i}_{j}.png')
                        cv2.imwrite(tile_path, tile)
                        print(f'Saved {tile_path}')
                        key_press = True
                    elif keyboard.is_pressed('down'):
                        # Discard tile
                        key_press = True
                    cv2.waitKey(100)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Iterate over all files in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_directory, file_name)
            process_image(file_path)
            print(f"Finished processing {file_name}")

if __name__ == '__main__':
    main()
