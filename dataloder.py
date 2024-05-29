import os
import tensorflow as tf
from tensorflow.keras import layers
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
        # label = tf.where(label > 0, tf.constant(255, dtype=tf.uint8), label) # TODO: special for ITRI dataset REMOVE IT FOR OTHER DATASETS
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
    


class CityscapesDataset:
    def __init__(self, 
                 root, 
                 split,
                 num_classes=19,  # Cityscapes has 19 main classes
                 crop_size=(512, 1024),
                 mean=[0.28689554, 0.32513303, 0.28389177], 
                 std=[0.18696375, 0.19017339, 0.18720214],
                 label_mapping={}
                 ):
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.label_mapping = {
            -1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 
            7: 0, 8: 1, 9: 0, 10: 0, 11: 2, 12: 3, 13: 4, 14: 0, 
            15: 0, 16: 0, 17: 5, 18: 0, 19: 6, 20: 7, 21: 8, 22: 9, 
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 
            30: 0, 31: 16, 32: 17, 33: 18
        }

        self.img_list = self.get_image_list()
        
        self.rgb_augmentation = tf.keras.Sequential([
            layers.RandomBrightness(factor=0.2),
            layers.RandomContrast(factor=0.2),
        ])
        self.rescale = tf.keras.Sequential([
            layers.Rescaling(scale=1./255),
        ])
        self.common_augmentation = tf.keras.Sequential([
            layers.RandomFlip(mode='horizontal'),
            layers.RandomRotation(factor=0.3),
            layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        ])

    def get_image_list(self):
        img_list = []
        img_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        for city in os.listdir(img_dir):
            city_dir = os.path.join(img_dir, city)
            for img_name in os.listdir(city_dir):
                if img_name.endswith('_leftImg8bit.png'):
                    img_list.append(os.path.join(city_dir, img_name))
        return natsorted(img_list)

    def __len__(self):
        return len(self.img_list)

    def read_image_and_label(self, image_path, label_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, self.crop_size, method='nearest')

        label = tf.io.read_file(label_path)
        label = tf.image.decode_png(label, channels=1)
        label = tf.image.resize(label, self.crop_size, method='nearest')
        label = self.convert_label(label)
        label = tf.cast(label, tf.int32)

        return image, label

    def convert_label(self, label):
        label = tf.cast(label, tf.int32)
        for k, v in self.label_mapping.items():
            label = tf.where(label == k, tf.constant(v, dtype=tf.int32), label)
        return label

    def preprocess_data(self, image_path, label_path):
        image, label = self.read_image_and_label(image_path, label_path)
        if self.split == 'train':
            # Concatenate image and label for consistent augmentation
            combined = tf.concat([tf.cast(image, tf.float32), tf.cast(label, tf.float32)], axis=-1)
            combined = self.common_augmentation(combined)
            image, label = combined[..., :3], combined[..., 3:]
            image = self.rgb_augmentation(image)
            label = tf.cast(label, tf.int32)
        
        # Standardize image
        image = self.rescale(image)
        image = (image - self.mean) / self.std
        
        return image, label

    def map_function(self, image_path, label_path):
        return self.preprocess_data(image_path, label_path)
    
    def get_dataset(self):
        file_paths = [(img_path, img_path.replace('leftImg8bit', 'gtFine').replace('_gtFine.png', '_gtFine_labelIds.png'))
                      for img_path in self.img_list]

        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(lambda x: self.map_function(x[0], x[1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    

if __name__ == "__main__":
    # Usage CR_WUR Dataset
    root = '/home/e300/mahmood/datasets/cr_wur_ds/'
    label_map = { 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6 }
    mean = [0.44669862, 0.4538911,  0.42983834] 
    std = [0.26438654, 0.256789,   0.26210858]
    split = 'train'  # or 'val', 'test'
    num_classes = 7  # specify the number of classes
    show_binary = False  # set to True to show binary images

    # Usage ITRI Insulator Dataset
    # root = '/home/e300/mahmood/datasets/v1_itri_formatted'
    # label_map = { 0:0, 255:1, 1:1 }
    # mean = [0.44669862, 0.4538911,  0.42983834] 
    # std = [0.26438654, 0.256789,   0.26210858]
    # split = 'train'  # or 'val', 'test'
    # num_classes = 2  # specify the number of classes
    # show_binary = False  # set to True to show binary images
    
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


    # root = '/home/e300/mahmood/datasets/cityscapes'
    # label_map = {i: i for i in range(34)}  # Update label mapping if needed
    # mean = [0.28805044, 0.32632137, 0.2854135 ] 
    # std = [0.17725339, 0.18182704, 0.17837277]
    # split = 'train'  # or 'val', 'test'
    # num_classes = 19  # specify the number of classes
    # show_binary = False  # set to True to show binary images
    
    # dataset_tf = CityscapesDataset(root, split, num_classes=num_classes, mean=mean, std=std, label_mapping=label_map)
    # tf_dataset = dataset_tf.get_dataset()
    # tf_dataset = tf_dataset.batch(3)
    
    # # Take a batch from the dataset
    # for images, labels in tf_dataset.take(3):
    #     print(images.shape, labels.shape)
    #     # Convert TensorFlow tensors to NumPy arrays for plotting
    #     images_np = images.numpy()
    #     labels_np = labels.numpy()
    #     print(np.unique(labels_np))
    #     print(images_np.shape, np.max(images_np), np.min(images_np))
    #     print(labels_np.shape, np.max(labels_np), np.min(labels_np))

    #     # Plot the samples
    #     plot_samples(images_np, labels_np, mean, std, num_classes, show_binary)