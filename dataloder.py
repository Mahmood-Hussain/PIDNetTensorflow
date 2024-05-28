import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

class KashmirDataset:
    def __init__(self, 
                 root, 
                 split,
                 num_classes=2,
                 crop_size=(384, 512),
                 mean=[0.44669862, 0.4538911,  0.42983834], 
                 std=[0.26438654, 0.256789,   0.26210858],
                 label_mapping={}
                 ):
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.img_list = natsorted(os.listdir(os.path.join(root, 'images', self.split)))
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.img_list)

    def read_image_and_label(self, image_path, label_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.crop_size, method='bilinear')
        image = (image / 255.0 - self.mean) / self.std
        image = tf.cast(image, tf.float32)

        label = tf.io.read_file(label_path)
        label = tf.image.decode_png(label, channels=1)
        label = tf.image.resize(label, self.crop_size, method='nearest')
        label = self.convert_label(label)
        label = tf.cast(label, tf.int32)

        return image, label

    def convert_label(self, label):
        label = tf.where(label > 0, tf.constant(255, dtype=tf.uint8), label) # TODO: special for ITRI dataset REMOVE IT FOR OTHER DATASETS
        label = tf.cast(label, tf.int32)
        for k, v in self.label_mapping.items():
            label = tf.where(label == k, tf.constant(v, dtype=tf.int32), label)
        return label

    def preprocess_data(self, image_path, label_path):
        image, label = self.read_image_and_label(image_path, label_path)
        return image, label

    def map_function(self, image_path, label_path):
        return self.preprocess_data(image_path, label_path)
    
    def get_dataset(self):
        file_paths = [(os.path.join(self.root, 'images', self.split, img_name),
                    os.path.join(self.root, 'labels', self.split, img_name.split('.')[0] + ".png"))
                    for img_name in self.img_list]

        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(lambda x: self.map_function(x[0], x[1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset


def plot_samples(images, labels, mean, std, num_classes, show_binary=True, num_samples=3):
    mean = np.array(mean)
    std = np.array(std)
    
    plt.figure(figsize=(10, 5 * num_samples))
    for i in range(num_samples):
        img = images[i] * std + mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(num_samples, 2, 2 * i + 2)
        if show_binary:
            plt.imshow(labels[i], cmap='gray')
        else:
            cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'tab20')
            plt.imshow(labels[i], cmap=cmap)
        plt.title('Label')
        plt.axis('off')

    plt.show()
    

if __name__ == "__main__":
    # Usage CR_WUR Dataset
    # root = '/home/e300/mahmood/datasets/cr_wur_dataset'
    # label_map = { 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6 }
    # mean = [0.44669862, 0.4538911,  0.42983834] 
    # std = [0.26438654, 0.256789,   0.26210858]
    # split = 'train'  # or 'val', 'test'
    # num_classes = 7  # specify the number of classes
    # show_binary = False  # set to True to show binary images

    # Usage ITRI Insulator Dataset
    root = '/home/e300/mahmood/datasets/v1_itri_formatted'
    label_map = { 0:0, 255:1, 1:1 }
    mean = [0.44669862, 0.4538911,  0.42983834] 
    std = [0.26438654, 0.256789,   0.26210858]
    split = 'train'  # or 'val', 'test'
    num_classes = 2  # specify the number of classes
    show_binary = False  # set to True to show binary images
    
    dataset_tf = KashmirDataset(root, split, num_classes=num_classes, mean=mean, std=std, label_mapping=label_map)
    tf_dataset = dataset_tf.get_dataset()
    tf_dataset = tf_dataset.batch(3)
    
    # Take a batch from the dataset
    for images, labels in tf_dataset.take(3):
        print(images.shape, labels.shape)
        # Convert TensorFlow tensors to NumPy arrays for plotting
        images_np = images.numpy()
        labels_np = labels.numpy()
        print(images_np.shape, np.max(images_np), np.min(images_np))
        print(labels_np.shape, np.max(labels_np), np.min(labels_np))

        # Plot the samples
        plot_samples(images_np, labels_np, mean, std, num_classes, show_binary)
