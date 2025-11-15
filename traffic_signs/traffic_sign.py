# traffic_sign.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report


# here I just set some basic parameters for my project: path to data folder, image size, batch size and how many epochs to train
DATA_DIR = "data"  
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
EPOCHS = 8

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def main():
    if not os.path.isdir(TRAIN_DIR):
        print("ERROR: folder", TRAIN_DIR, "not found.")
        print("Please put GTSRB Train folder inside", DATA_DIR)
        return
    
# load images from Train folder. tensorflow automatically reads subfolders 0..42 as classes
    print("Loading dataset from:", TRAIN_DIR)

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,    
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="int"
    )

    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="int"
    )

# class_names is just list of folder names like 0, 1, 2 ... 42
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Number of classes:", num_classes)
    print("Example class names:", class_names[:5])
    print("Showing some example images...")
    for images, labels in train_ds.take(1):
        
# I want to see some random images from dataset for check
        plt.figure(figsize=(6, 6))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(str(labels[i].numpy()))
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# here I build CNN model. first layer rescales pixels from 0-255 to 0-1 and add small augmentations
    model = keras.Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),

            layers.Conv2D(32, (3, 3), activation="relu"), # first conv layer, 32 filters
            layers.MaxPooling2D((2, 2)), # make image smaller and keep important features

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(model.summary())
    print("Starting training...")
    
# now I train my model on train_ds and check quality on val_ds, history object will contain accuracy and loss for each epoch
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )
    print("Training finished.")
    print("Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print("Computing F1-score and confusion matrix...")

# after training I want to see more detailed evaluation. F1 score shows quality for all classes not only overall accuracy. confusion matrix shows which traffic signs are mixed by model
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        preds_labels = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds_labels)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=range(num_classes))
    disp.plot(include_values=False, cmap="Blues", ax=plt.gca(), xticks_rotation="vertical")
    plt.title("Confusion matrix (validation)")
    plt.tight_layout()
    plt.show()
    model_path = "gtsrb_cnn_simple.keras"
    model.save(model_path)
    print("Model saved to:", model_path)
    print("\nDone")


if __name__ == "__main__":
    main()
