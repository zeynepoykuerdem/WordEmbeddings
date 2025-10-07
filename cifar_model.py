import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, BatchNormalization
from tensorflow.keras import models,Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import LeakyReLU,ReLU
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import keras_tuner as kt
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shap

def load_data():
    """
    Loads CFIR dataset from keras datasets and return train and test
    :return: Train and test datasets
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train=x_train.astype('float32')/255
    x_test=x_test.astype('float32')/255
    return (x_train, y_train), (x_test, y_test)


def building_model():
    model=Sequential()
    model.add(Input(shape=(32,32,3)))
    model.add(Conv2D(64, (5, 5), padding='same',kernel_regularizer=regularizers.l2(0.00026882)))
    model.add(LeakyReLU(alpha=0.12941))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.36567))
    model.add(Conv2D(256, (5, 5), padding='same',kernel_regularizer=regularizers.l2(0.00058105)))
    model.add(LeakyReLU(alpha=0.28939))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(256,kernel_regularizer=regularizers.l2(0.0001625)))
    model.add(LeakyReLU(alpha=0.025772))
    model.add(Dropout(0.32478))
    model.add(Dense(10, activation="softmax"))
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00042514)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    return model


def explainable_ai(model, x_train, x_test):
    # Fix: Use a small subset of training data as background
    background = x_train[:10]

    explainer = shap.GradientExplainer(model, background)
    test_samples=x_test[:3]
    shap_values = explainer.shap_values(test_samples)
    print("SHAP values shape:", shap_values[0].shape)
    print("Min SHAP value:", np.min(shap_values[0]))
    print("Max SHAP value:", np.max(shap_values[0]))
    
    plt.figure(figsize=(15, 5))
    
    # Plot original images
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.imshow(test_samples[i])
        plt.title(f'Original Image {i+1}')
        plt.axis('off')
    
    # Plot SHAP attributions for the top predicted class
    for i in range(3):
        plt.subplot(2, 3, i + 4)
        # Sum across color channels and take absolute values for visualization
        shap_img = np.sum(np.abs(shap_values[0][i]), axis=-1)
        plt.imshow(shap_img, cmap='hot')
        plt.title(f'SHAP Attribution {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return shap_values

def model_prediction(model,x_test,y_test,class_labels):
    y_test_preds=model.predict(x_test)
    y_test_preds=np.argmax(y_test_preds,axis=1)

    test_acc=accuracy_score(y_test,y_test_preds)
    

    print("Test Accuracy : ",(test_acc))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_test, y_test_preds))
    print("\nClassification Report :")
    print(classification_report(y_test, y_test_preds, target_names=class_labels))


def plotting_metrics(history):
     #plotting
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    plt.show()