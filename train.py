import os
import argparse
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from criterion import BasnetLoss
from pidnet import get_pred_model
from dataloder import KashmirDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train PIDNet on custom Dataset")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def get_latest_experiment_dir(base_dir):
    try:
        experiment_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not experiment_dirs:
            return None
        experiment_dirs.sort(reverse=True)
        return os.path.join(base_dir, experiment_dirs[0])
    except:
        print("Did you accidentally turn resume option on or wrong experiment directory name? 🤭🤭🤭 ")
        print("Startig training from scratch again ..!")

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if not checkpoints:
        return None
    checkpoints.sort(reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])

def train_pidnet(config):
    # Verify GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid allocating all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🔥🔥 GPUs: {gpus} 🔥🔥")
        except RuntimeError as e:
            print(e)

    base_experiment_dir = os.path.join("experiments", config['experiment_name'])
    if config['resume']:
        latest_experiment_dir = get_latest_experiment_dir(base_experiment_dir)
        if latest_experiment_dir:
            experiment_dir = latest_experiment_dir
            print(f"Resuming from latest experiment directory: {experiment_dir}")
        else:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_dir = os.path.join(base_experiment_dir, current_time)
            os.makedirs(experiment_dir, exist_ok=True)
            print(f"No previous experiment found. Starting new experiment at: {experiment_dir}")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = os.path.join(base_experiment_dir, current_time)
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"Starting new experiment at: {experiment_dir}")
    
    # save a copy of config back
    with open(os.path.join(experiment_dir, 'config_copy.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    train_ds = KashmirDataset(config['root'], 'train', crop_size=(config['img_height'], config['img_width']), num_classes=config['num_classes'], mean=config['mean'], std=config['std'], label_mapping=dict(zip(config['label_map'][::2], config['label_map'][1::2])))
    tf_train_ds = train_ds.get_dataset().batch(config['batch_size'])

    val_ds = KashmirDataset(config['root'], 'val', crop_size=(config['img_height'], config['img_width']), num_classes=config['num_classes'], mean=config['mean'], std=config['std'], label_mapping=dict(zip(config['label_map'][::2], config['label_map'][1::2])))
    tf_val_ds = val_ds.get_dataset().batch(config['batch_size'])

    pidnet_model = get_pred_model(config['model_type'], (config['img_height'], config['img_width'], 3), config['num_classes'])

    # Check for the latest checkpoint
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    initial_epoch = 0
    if config['resume'] and latest_checkpoint:
        print(f"🚀🚀 Resuming training from {latest_checkpoint} 🚀🚀🚀")
        pidnet_model = keras.models.load_model(latest_checkpoint, custom_objects={"BasnetLoss": BasnetLoss, "MeanIoU": tf.keras.metrics.MeanIoU})
        initial_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
    
    if config['show_model_summary']:
        pidnet_model.summary()  # Show model summary.

    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'], epsilon=config['epsilon'])

    # Compile model
    pidnet_model.compile(
        loss=BasnetLoss(),
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=config['num_classes'], name="mIoU", sparse_y_pred=False, sparse_y_true=True, axis=-1)
        ],
    )

    # TensorBoard callback
    log_dir = os.path.join(experiment_dir, "logs")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    # Custom callback to save images
    class SaveSamplesCallback(keras.callbacks.Callback):
        def __init__(self, val_ds, output_dir, mean, std, num_samples=2):
            super().__init__()
            self.val_ds = val_ds
            self.output_dir = output_dir
            self.mean = np.array(mean)
            self.std = np.array(std)
            self.num_samples = num_samples

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        def on_epoch_end(self, epoch, logs=None):
            for images, labels in self.val_ds.take(1):
                predictions = self.model.predict(images)
                predictions = tf.argmax(predictions, axis=-1)
                predictions = tf.expand_dims(predictions, axis=-1)
                images = (images * self.std + self.mean) * 255.0
                images = np.clip(images, 0, 255).astype(np.uint8)

                for i in range(min(self.num_samples, images.shape[0])):
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(images[i])
                    ax[0].set_title('Image')
                    ax[0].axis('off')
                    ax[1].imshow(predictions[i], cmap='gray')
                    ax[1].set_title('Prediction')
                    ax[1].axis('off')
                    plt.savefig(os.path.join(self.output_dir, f'epoch_{epoch + 1}_sample_{i + 1}.png'))
                    plt.close(fig)

    # Callbacks
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model_checkpoint_{epoch:02d}.h5'),
        save_freq='epoch',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_weights_only=False,
    )

    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    save_samples_callback = SaveSamplesCallback(val_ds=tf_val_ds, output_dir=os.path.join(experiment_dir, 'sample_images'), mean=config['mean'], std=config['std'])

    print("🚀🚀 Starting training now 🚀🚀🚀")
    # Train the model
    pidnet_model.fit(
        tf_train_ds,
        validation_data=tf_val_ds,
        epochs=config['epochs'],
        initial_epoch=initial_epoch, 
        callbacks=[checkpoint_callback, reduce_lr_callback, tensorboard_callback, save_samples_callback]
    )

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train_pidnet(config)

# Example usage
# python train.py --config config.json
