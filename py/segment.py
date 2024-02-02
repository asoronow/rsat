import cv2
import numpy as np
from pathlib import Path
import argparse
class SectionImage:
    """
    SectionImage Object

    Represents an image of a tissue section.

    Parameters
    ----------
    filepath : str
        The path to the image file.
    image : np.array
        The image.
    
    """
    def __init__(self, filepath, image):
        self.filepath = Path(filepath)
        self.image = image

    def make_tiles(self, tile_size):
        """
        Make tiles from the image.

        Parameters
        ----------
        tile_size : int
            The size of the tiles to be made.

        Returns
        -------
        list
            A list of tiles.

        """
        tiles = []
        for i in range(0, self.image.shape[0], tile_size):
            for j in range(0, self.image.shape[1], tile_size):
                tile = self.image[i:i+tile_size, j:j+tile_size]
                tiles.append(tile)
        return tiles
    
    def save_tiles(self, tile_size, output_dir):
        tiles = self.make_tiles(tile_size)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        row_count = self.image.shape[0] // tile_size + (1 if self.image.shape[0] % tile_size else 0)
        col_count = self.image.shape[1] // tile_size + (1 if self.image.shape[1] % tile_size else 0)

        for i in range(row_count):
            for j in range(col_count):
                tile_index = i * col_count + j
                if tile_index < len(tiles):  # Check to avoid index error
                    tile_path = output_path / f"{self.filepath.stem}_tile_{i}_{j}.tif"
                    cv2.imwrite(str(tile_path), tiles[tile_index])

    
    def reconstruct_from_tiles(self, input_dir):
        tile_paths = sorted(Path(input_dir).glob(f"{self.filepath.stem}_tile_*.tif"),
                            key=lambda x: (int(x.stem.split('_')[-2]), int(x.stem.split('_')[-1])))

        # Load the first tile to determine its properties for initialization
        first_tile = cv2.imread(str(tile_paths[0]), cv2.IMREAD_GRAYSCALE)
        tile_height, tile_width = first_tile.shape

        # Determine the max row and column indices from filenames
        max_row_idx = max_col_idx = 0
        for path in tile_paths:
            parts = path.stem.split('_')
            row_idx, col_idx = int(parts[-2]), int(parts[-1])  # Adjust according to actual naming convention
            max_row_idx = max(max_row_idx, row_idx)
            max_col_idx = max(max_col_idx, col_idx)

        # Calculate the dimensions of the reconstructed image
        reconstructed_height = (max_row_idx + 1) * tile_height
        reconstructed_width = (max_col_idx + 1) * tile_width

        # Initialize an empty array to hold the reconstructed image
        reconstructed_image = np.zeros((reconstructed_height, reconstructed_width), dtype=np.uint8)

        # Place each tile in the correct position
        for tile_path in tile_paths:
            parts = tile_path.stem.split('_')
            row_idx, col_idx = int(parts[-2]), int(parts[-1])  # Correct indices based on naming convention
            tile = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)

            start_row = row_idx * tile_height
            start_col = col_idx * tile_width

            # Calculate the end row and column indices based on the actual tile size
            # This handles edge cases where the tile might not be the full expected size
            end_row = start_row + tile.shape[0]
            end_col = start_col + tile.shape[1]

            # Adjust the placement logic to account for edge tiles
            reconstructed_image[start_row:end_row, start_col:end_col] = tile[:min(tile.shape[0], reconstructed_image.shape[0] - start_row),
                                                                           :min(tile.shape[1], reconstructed_image.shape[1] - start_col)]

        return reconstructed_image



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make tiles from an image.")
    parser.add_argument("--input", type=str, help="The input image file.")
    parser.add_argument("--output", type=str, help="The output directory.")
    parser.add_argument("--tile_size", type=int, help="The size of the tiles.", default=303)
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    section_image = SectionImage(args.input, image)
    section_image.save_tiles(args.tile_size, args.output)
    # reconstructed = section_image.reconstruct_from_tiles(args.output)
    # cv2.imwrite(f"{args.output}/{Path(args.input).stem}_reconstructed.png", reconstructed)
