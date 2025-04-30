import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

class ImageUtils:
    @staticmethod
    def load_image(image_path) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise FileNotFoundError
        Img = Image.open(image_path).convert('L')
        return np.array(Img).astype(np.float32)

    @staticmethod
    def save_image(image, path) -> None:
        if image.dtype != np.uint8:
            image = ImageUtils.normalize_image(image)
        Img = Image.fromarray(image)
        return Img.save(path)

    @staticmethod
    def normalize_image(image) -> np.ndarray[np.uint8]:
        max_value = np.max(image)
        min_value = np.min(image)
        if max_value == min_value:
            return np.full(image.shape, 127, dtype=np.uint8)
        norm = (image - min_value) / (max_value - min_value) * 255
        return norm.astype(np.uint8)

class ConvolutionProcessor:
    @staticmethod
    def create_sobel_kernel() -> Tuple[np.ndarray, np.ndarray]:
        Gx = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ]
        )
        Gy = np.array(
            [
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1],
            ]
        )
        return Gx, Gy

    @staticmethod
    def calculate_pad_size(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        if len(kernel_size) != 2:
            raise ValueError("Kernel size must be of length 2")
        pad_x_size = (kernel_size[0] - 1) // 2
        pad_y_size = (kernel_size[1] - 1) // 2
        return pad_x_size, pad_y_size

    @staticmethod
    def pad(image, pad_value: int, pad_size: Tuple[int, int] | None = None, kernel_size: Tuple[int, int] | None = None) -> np.ndarray:
        if pad_size is None:
            if kernel_size is not None:
                pad_size = ConvolutionProcessor.calculate_pad_size(kernel_size)
            else:
                raise ValueError("you must provide either kernel_size or pad_size.")
        if pad_value == 0:
            new_image_array = np.zeros((image.shape[0] + 2 * pad_size[0], image.shape[1] + 2 * pad_size[1]))
        else:
            new_image_array = np.ones((image.shape[0] + 2 * pad_size[0], image.shape[1] + 2 * pad_size[1])) * pad_value
        new_image_array[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]] = image
        return new_image_array

    @staticmethod
    def apply_convolution(image, kernel: np.ndarray, padded_image: np.ndarray | None = None) -> np.ndarray:
        new_image_array = np.zeros(image.shape)
        if padded_image is None:
            padded_image = ConvolutionProcessor.pad(image, pad_value=0, kernel_size=kernel.shape)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                roi = padded_image[x:x+kernel.shape[0], y:y+kernel.shape[1]]
                new_image_array[x,y] = np.sum(roi * kernel)
        return new_image_array

    @staticmethod
    def compute_block_variance(image: np.ndarray, block_size: int = 16) -> np.ndarray:
        h, w = image.shape
        nH, nW = h // block_size, w // block_size
        variance_map = np.zeros((nH, nW))
        for i in range(nH):
            for j in range(nW):
                blk = image[i * block_size:(i + 1) * block_size,
                      j * block_size:(j + 1) * block_size]
                variance_map[i, j] = np.var(blk)
        return variance_map

    @staticmethod
    def calculate_gradient_angle(gx_image: np.ndarray, gy_image: np.ndarray, window_size: int = 3) -> np.ndarray:
        if gx_image.shape != gy_image.shape:
            raise ValueError("Input gradient images must have the same shape")
        height, width = gx_image.shape
        angle_image = np.zeros(gx_image.shape)
        half_w = window_size // 2
        padded_gx = np.pad(gx_image, half_w, mode='constant')
        padded_gy = np.pad(gy_image, half_w, mode='constant')
        for y in range(height):
            for x in range(width):
                gx_window = padded_gx[y:y + window_size, x:x + window_size]
                gy_window = padded_gy[y:y + window_size, x:x + window_size]
                numerator = np.sum(2 * gx_window * gy_window)
                denominator = np.sum(gx_window ** 2 - gy_window ** 2)
                if denominator != 0:
                    angle = (np.pi / 2) + 0.5 * np.arctan2(numerator, denominator)
                else:
                    angle = np.pi / 2 if numerator >= 0 else -np.pi / 2
                angle_image[y, x] = angle
        return angle_image

    @staticmethod
    def compute_orientation_field(image: np.ndarray, window_size: int = 16) -> np.ndarray:
        Gx, Gy = ConvolutionProcessor.create_sobel_kernel()
        sobel_x = ConvolutionProcessor.apply_convolution(image, Gx)
        sobel_y = ConvolutionProcessor.apply_convolution(image, Gy)
        angle = ConvolutionProcessor.calculate_gradient_angle(sobel_x, sobel_y, window_size)
        return np.mod(angle, np.pi)

    @staticmethod
    def quantize_orientations(angle: np.ndarray, num_bins: int = 16) -> np.ndarray:
        bin_size = np.pi / num_bins
        idx = np.round(angle / bin_size) % num_bins
        return idx.astype(np.uint8)

    @staticmethod
    def compute_bof(quantized: np.ndarray, block_size: int = 16) -> np.ndarray:
        h, w = quantized.shape
        nH, nW = h // block_size, w // block_size
        bof = np.zeros((nH, nW), dtype=np.uint8)
        for i in range(nH):
            for j in range(nW):
                blk = quantized[i*block_size:(i+1)*block_size,
                                j*block_size:(j+1)*block_size]
                bof[i, j] = np.bincount(blk.ravel(), minlength=16).argmax()
        return bof

    @staticmethod
    def draw_bof_slope_image(
            bof: np.ndarray,
            original_image: np.ndarray,
            block_size: int = 16,
            variance_threshold: float = 5.0
    ) -> np.ndarray:
        H, W = bof.shape
        height, width = H * block_size, W * block_size
        canvas = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(canvas)

        # Compute variance map
        variance_map = ConvolutionProcessor.compute_block_variance(original_image, block_size)

        for i in range(H):
            for j in range(W):
                # Skip low-variance blocks
                if variance_map[i, j] < variance_threshold:
                    continue

                angle = bof[i, j] * (np.pi / 16)
                cx = j * block_size + block_size / 2
                cy = i * block_size + block_size / 2
                length = block_size * 0.9 / 2
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=0, width=1)
        return np.array(canvas, dtype=np.uint8)

    @staticmethod
    def overlay_bof_on_image(
            image: np.ndarray,
            bof: np.ndarray,
            original_image: np.ndarray,
            block_size: int = 16,
            line_color=(255, 0, 0),
            line_width: int = 1,
            variance_threshold: float = 5.0
    ) -> np.ndarray:
        gray = np.clip(image, 0, 255).astype(np.uint8)
        base = Image.fromarray(gray).convert('RGB')
        draw = ImageDraw.Draw(base)

        # Compute variance map
        variance_map = ConvolutionProcessor.compute_block_variance(original_image, block_size)

        H, W = bof.shape
        for i in range(H):
            for j in range(W):
                # Skip low-variance blocks
                if variance_map[i, j] < variance_threshold:
                    continue

                angle = bof[i, j] * (np.pi / 16)
                cx = j * block_size + block_size / 2
                cy = i * block_size + block_size / 2
                L = block_size * 0.45
                dx, dy = L * np.cos(angle), L * np.sin(angle)
                draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=line_color, width=line_width)
        return np.array(base)


def main():
    image_base_path = r"Images"
    os.makedirs(image_base_path, exist_ok=True)
    original_image_path = os.path.join(image_base_path, r"104_3.tif")

    # ==========================================================================================
    #          Loading the Original Image
    # ==========================================================================================
    image_array = ImageUtils.load_image(original_image_path)
    print(50 * "=", "\nOriginal Image Array:\n", image_array)

    # ==========================================================================================
    #          Padding the Original Image
    # ==========================================================================================
    padded_image_array = ConvolutionProcessor.pad(
        image=image_array,
        kernel_size=(3, 3),
        pad_value=255,
    )
    print(50 * "=", "\n255 Padded Original Image Array:\n", image_array)

    # ==========================================================================================
    #          Calculating the Sobel Gx and Gy
    # ==========================================================================================
    Gx, Gy = ConvolutionProcessor.create_sobel_kernel()
    image_sobel_x = ConvolutionProcessor.apply_convolution(
        padded_image=padded_image_array,
        image=image_array,
        kernel=Gx,
    )
    print(50 * "=", "\nSobel X Image Array:\n", image_sobel_x)
    plt.figure(1)
    plt.imshow(image_sobel_x, cmap="gray")
    plt.title("Sobel X")
    plt.savefig(r"Images/sobel_x.tiff")
    plt.show()
    # ImageUtils.save_image(np.abs(image_sobel_x), path=r"Images/sobel_x.tiff")

    image_sobel_y = ConvolutionProcessor.apply_convolution(
        padded_image=padded_image_array,
        image=image_array,
        kernel=Gy,
    )
    print(50 * "=", "\nSobel Y Image Array:\n",  image_sobel_y)

    plt.figure(2)
    plt.imshow(image_sobel_y, cmap="gray")
    plt.title("Sobel Y")
    plt.savefig(r"Images/sobel_y.tiff")
    plt.show()
    # ImageUtils.save_image(np.abs(image_sobel_x), path=r"Images/sobel_y.tiff")

    gradient_angle = ConvolutionProcessor.calculate_gradient_angle(
        image_sobel_x,
        image_sobel_y,
        window_size=16
    )
    plt.figure(3)
    plt.imshow(gradient_angle, cmap="gray")
    plt.title("Gradient Angle")
    plt.savefig(r"Images/gradient_angle.tiff")
    plt.show()

    gradient_angle = np.mod(gradient_angle, np.pi)
    quant = ConvolutionProcessor.quantize_orientations(
        gradient_angle,
        num_bins=16
    )
    bof = ConvolutionProcessor.compute_bof(quant, block_size=16)
    plt.figure()
    plt.title("Block Orientation Field")
    plt.imshow(bof, cmap='jet')
    plt.colorbar()
    plt.savefig(r"Images/bof.tiff")
    plt.show()

    slope_img = ConvolutionProcessor.draw_bof_slope_image(
        bof,
        original_image=image_array,
        block_size=16,
        variance_threshold=150.0
    )
    plt.figure()
    plt.title('BOF Slope Visualization')
    plt.imshow(slope_img, cmap='gray')
    plt.axis('off')
    plt.show()
    ImageUtils.save_image(slope_img, os.path.join(image_base_path, 'bof_slope.png'))

    overlay = ConvolutionProcessor.overlay_bof_on_image(
        image_array,
        bof,
        original_image=image_array,
        block_size=16,
        variance_threshold=150.0
    )
    plt.figure()
    plt.title('BOF Overlay on Fingerprint')
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()
    ImageUtils.save_image(overlay, os.path.join(image_base_path, 'bof_overlay.png'))

if __name__ == '__main__':
    main()
