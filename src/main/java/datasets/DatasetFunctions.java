package main.java.datasets;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

public class DatasetFunctions {

    public static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) throws IOException {
        Image resultingImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_DEFAULT);
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        outputImage.getGraphics().drawImage(resultingImage, 0, 0, null);
        return outputImage;
    }
    
    public static <Path> double[][][] convertImageTo3DArray(Path imagePath, int targetWidth, int targetHeight) throws IOException {
        BufferedImage image = ImageIO.read(new File(String.valueOf(imagePath)));
        image = resizeImage(image, targetWidth, targetHeight);

        int width = image.getWidth();
        int height = image.getHeight();

        // Assuming RGB image, change accordingly for grayscale or other formats
        double[][][] pixelData = new double[height][width][3];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int rgb = image.getRGB(j, i);

                // Extract RGB values
//                pixelData[i][j][0] = ((rgb >> 16) & 0xFF) / 255.; // Red
//                pixelData[i][j][1] = ((rgb >> 8) & 0xFF) / 255.;  // Green
//                pixelData[i][j][2] = (rgb & 0xFF) / 255.;         // Blue

                pixelData[i][j][0] = ((rgb >> 16) & 0xFF); // Red
                pixelData[i][j][1] = ((rgb >> 8) & 0xFF);  // Green
                pixelData[i][j][2] = (rgb & 0xFF);         // Blue
            }
        }

        return pixelData;
    }
}
