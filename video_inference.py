import cv2
import numpy as np
import argparse
import tensorflow as tf
from criterion import BasnetLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Perform video inference using TensorFlow checkpoint")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the TensorFlow checkpoint directory')
    return parser.parse_args()

def preprocess_frame(frame, input_shape, mean, std):
    resized_frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
    normalized_frame = (resized_frame / 255.0 - mean) / std
    return normalized_frame.astype(np.float32)

def denormalize_frame(frame, mean, std):
    denormalized_frame = frame * std + mean
    return np.clip(denormalized_frame * 255.0, 0, 255).astype(np.uint8)

def postprocess_frame(prediction, original_shape, num_classes):
    prediction = np.argmax(prediction, axis=-1)
    if num_classes > 1:
        prediction = prediction.astype(np.uint8)
    else:
        prediction = (prediction > 0.5).astype(np.uint8)
    resized_prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized_prediction

def video_inference(video_path, checkpoint_path, mean, std, num_classes):
    # Load the TensorFlow model from the checkpoint
    model = tf.keras.models.load_model(checkpoint_path, custom_objects={"BasnetLoss": BasnetLoss, "MeanIoU": tf.keras.metrics.MeanIoU})
    input_shape = model.input_shape

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        preprocessed_frame = preprocess_frame(rgb_frame, input_shape, mean, std)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

        # Perform inference
        predictions = model.predict(preprocessed_frame)
        postprocessed_frame = postprocess_frame(predictions[0], (frame.shape[0], frame.shape[1]), num_classes)

        # Convert postprocessed_frame to color for display
        if num_classes > 1:
            denormalized_prediction = postprocessed_frame * (255 // (num_classes - 1))
            denormalized_prediction = cv2.applyColorMap(denormalized_prediction, cv2.COLORMAP_JET)
        else:
            denormalized_prediction = postprocessed_frame * 255

        combined_frame = cv2.hconcat([frame, denormalized_prediction])

        cv2.imshow('Inference', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    mean = [0.44669862, 0.4538911,  0.42983834]
    std = [0.26438654, 0.256789,   0.26210858]
    num_classes = 2
    video_inference(args.video_path, args.checkpoint_path, mean, std, num_classes)
