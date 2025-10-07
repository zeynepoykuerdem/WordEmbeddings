import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers, Sequential
from tensorflow.keras.datasets import cifar10
import shap

def load_data():
    """
    Loads CIFAR dataset from keras datasets and return train and test
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
    plt.savefig('/workspace/shap_visualization.png')  # Save instead of show for headless environment
    print("SHAP visualization saved to shap_visualization.png")
    
    return shap_values

# Test the explainable AI function
def test_explainable_ai():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Build and compile model
    model = building_model()
    
    # Train for a few epochs just to have some weights
    print("Training model briefly...")
    model.fit(x_train[:1000], y_train[:1000], epochs=2, batch_size=32, verbose=1)
    
    # Test explainable AI
    print("\nTesting explainable AI...")
    try:
        shap_values = explainable_ai(model, x_train, x_test)
        print("Explainable AI function completed successfully!")
        print(f"SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"Number of classes: {len(shap_values)}")
            print(f"Shape of SHAP values for first class: {shap_values[0].shape}")
    except Exception as e:
        print(f"Error in explainable AI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_explainable_ai()