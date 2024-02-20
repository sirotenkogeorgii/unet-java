package main.java.datasets;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import javax.imageio.ImageIO;

public class PeopleSegmentationDataset implements IDataset {
    private Path data_path_;
    private Path label_path_;
    private ArrayList<IDatasetSample> data_samples_;
    private int size_;
    public PeopleSegmentationDataset(String data_path, String label_path) {
        if (data_path == null || label_path == null)
            throw new RuntimeException();
        data_samples_ = new ArrayList<>();
        data_path_ = Paths.get(data_path);
        label_path_ = Paths.get(label_path);
        retrieve_dataset(data_path_, label_path_);
        size_ = data_samples_.size();
    }

    private void retrieve_dataset(Path data_path, Path label_path) {
        try {
            final int[] counter = {0};
            int max_dataset_size = 10;
            Files.walk(data_path).filter(Files::isRegularFile).forEach(train_sample_path -> {
                if (counter[0] < max_dataset_size) {
                    Path file_name = train_sample_path.getFileName();
                    Path train_mask_path = label_path.resolve(file_name);

                    System.out.println(counter[0]);

                    try {
                        data_samples_.add(new SegmentationSample(train_sample_path, train_mask_path));
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                    counter[0]++;
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Path get_data_path() { return data_path_; }
    public Path get_label_path() { return label_path_; }
    public int get_size() { return size_; }
    public IDatasetSample get_sample(int index) {
        if (!is_valid_index(index))
            throw new RuntimeException();
        return data_samples_.get(index);
    }
    private boolean is_valid_index(int index) { return index >= 0 && index < get_size(); }
    public Iterator<Batch> batch_iterator(int batch_size) {
        if (batch_size < 1) throw new RuntimeException();
        return new Iterator<>() {
            private int current_index = 0;
            public boolean hasNext() { return current_index < size_; }
            public Batch next() {
                var batch_data = new ArrayList<IDatasetSample>();
                int batch_end_index = Math.min(current_index + batch_size, size_);
                for (; current_index < batch_end_index; ++current_index)
                    batch_data.add(data_samples_.get(current_index));
                return new Batch(batch_data);
            }
        };
    }
}

class Program {
    public static void main(String[] args) {
        String images = "/Users/georgiisirotenko/Downloads/people_segmentation/train/some_images";
        String masks = "/Users/georgiisirotenko/Downloads/people_segmentation/train/mask";
        var dataset = new PeopleSegmentationDataset(images, masks);

        for (Iterator<Batch> it = dataset.batch_iterator(16); it.hasNext(); ) {
            Batch batch = it.next();
            IDatasetSample sample = batch.iterator().next();
            int[] image_size = sample.get_data().get_size();
            System.out.printf("next batch size: %d, first image size: [%d, %d, %d]\n", batch.get_size(), image_size[0], image_size[1], image_size[2]);
        }
//        IDatasetSample sample = dataset.get_sample(0);
//        int[] image_size = sample.get_data().get_size();
//        System.out.printf("size: [%d, %d, %d]\n", image_size[0], image_size[1], image_size[2]);
    }
}













