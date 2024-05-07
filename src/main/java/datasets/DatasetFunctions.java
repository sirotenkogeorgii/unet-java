package main.java.datasets;

import javax.imageio.ImageIO;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Utility class providing static methods to manipulate image data.
 * Includes functions to resize images and convert them to a 3-dimensional array format,
 * suitable for use in various image processing and machine learning applications.
 */
public class DatasetFunctions {

    /**
     * Resizes a given image to specified dimensions.
     *
     * @param originalImage The {@link BufferedImage} to be resized.
     * @param targetWidth The desired width of the image after resizing.
     * @param targetHeight The desired height of the image after resizing.
     * @return A new {@link BufferedImage} that has been resized to the specified dimensions.
     * @throws IOException If an error occurs during the resizing process.
     */
    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
        Image resultingImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_DEFAULT);
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        outputImage.getGraphics().drawImage(resultingImage, 0, 0, null);
        return outputImage;
    }

    /**
     * Converts an image file to a 3-dimensional array where the third dimension represents
     * color channels (Red, Green, Blue): [height, width, 3]
     *
     * @param imagePath The path to the image file to be converted.
     * @param targetWidth The target width to which the image should be resized before conversion.
     * @param targetHeight The target height to which the image should be resized before conversion.
     * @return A 3-dimensional array of doubles, where each entry contains the RGB values of the corresponding pixel.
     * @throws IOException If there is an error reading the image file or processing it.
     */
    public static <Path> double[][][] convertImageTo3DArray(Path imagePath, int targetWidth, int targetHeight) throws IOException {
        BufferedImage image = ImageIO.read(new File(String.valueOf(imagePath)));
        image = resizeImage(image, targetWidth, targetHeight);

        int width = image.getWidth();
        int height = image.getHeight();

        double[][][] pixelData = new double[height][width][3];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int rgb = image.getRGB(j, i);

                pixelData[i][j][0] = ((rgb >> 16) & 0xFF); // Red
                pixelData[i][j][1] = ((rgb >> 8) & 0xFF);  // Green
                pixelData[i][j][2] = (rgb & 0xFF);         // Blue
            }
        }

        return pixelData;
    }
}
