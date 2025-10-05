import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, BatchNormalization
from tensorflow.keras import models, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import RandomRotation, RandomFlip, Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shap


def load_data():
    """
    Loads CIFAR-10 dataset from keras datasets and return train and test
    :return: Train and test datasets
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)


def explainable_ai(model, train_images, test_images):
    """
    Generate SHAP explanations for model predictions
    """
    explainer = shap.GradientExplainer(model, train_images)
    shap_values = explainer.shap_values(test_images[:3])
    # We are gonna have 10 outputs, 10 explanations (one for each output)
    shap.image_plot([shap_values[i] for i in range(10)], test_images[:3])


def build_hypermodel(hp):
    """
    Build a CNN model with hyperparameters for Bayesian optimization
    """
    inputs = Input(shape=(32, 32, 3))
    x = inputs
    
    # First Conv Block
    filters_1 = hp.Choice('conv1_filters', values=[32, 64, 128])
    kernel_size_1 = hp.Choice('conv1_kernel_size', values=[3, 5, 7])
    x = Conv2D(filters_1, (kernel_size_1, kernel_size_1), padding='same',
               kernel_regularizer=regularizers.l2(hp.Float('l2_reg_1', 1e-5, 1e-2, sampling='log')))(x)
    x = tf.keras.layers.LeakyReLU(alpha=hp.Float('leaky_relu_alpha_1', 0.01, 0.3))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=hp.Choice('pool_size_1', values=[2, 3]))(x)
    
    # Dropout after first block
    x = Dropout(hp.Float('dropout_1', 0.1, 0.5))(x)
    
    # Second Conv Block
    filters_2 = hp.Choice('conv2_filters', values=[64, 128, 256])
    kernel_size_2 = hp.Choice('conv2_kernel_size', values=[3, 5, 7])
    x = Conv2D(filters_2, (kernel_size_2, kernel_size_2), padding='same',
               kernel_regularizer=regularizers.l2(hp.Float('l2_reg_2', 1e-5, 1e-2, sampling='log')))(x)
    x = tf.keras.layers.LeakyReLU(alpha=hp.Float('leaky_relu_alpha_2', 0.01, 0.3))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=hp.Choice('pool_size_2', values=[2, 3]))(x)
    
    # Optional third Conv Block
    if hp.Boolean('use_third_conv'):
        filters_3 = hp.Choice('conv3_filters', values=[128, 256, 512])
        kernel_size_3 = hp.Choice('conv3_kernel_size', values=[3, 5])
        x = Conv2D(filters_3, (kernel_size_3, kernel_size_3), padding='same',
                   kernel_regularizer=regularizers.l2(hp.Float('l2_reg_3', 1e-5, 1e-2, sampling='log')))(x)
        x = tf.keras.layers.LeakyReLU(alpha=hp.Float('leaky_relu_alpha_3', 0.01, 0.3))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(hp.Float('dropout_3', 0.1, 0.5))(x)
    
    # Flatten and Dense layers
    x = Flatten()(x)
    
    # First Dense layer
    dense_units_1 = hp.Choice('dense_units_1', values=[64, 128, 256, 512])
    x = Dense(dense_units_1, kernel_regularizer=regularizers.l2(hp.Float('dense_l2_reg_1', 1e-5, 1e-2, sampling='log')))(x)
    x = tf.keras.layers.LeakyReLU(alpha=hp.Float('dense_leaky_relu_alpha_1', 0.01, 0.3))(x)
    x = Dropout(hp.Float('dense_dropout_1', 0.2, 0.7))(x)
    
    # Optional second Dense layer
    if hp.Boolean('use_second_dense'):
        dense_units_2 = hp.Choice('dense_units_2', values=[32, 64, 128, 256])
        x = Dense(dense_units_2, kernel_regularizer=regularizers.l2(hp.Float('dense_l2_reg_2', 1e-5, 1e-2, sampling='log')))(x)
        x = tf.keras.layers.LeakyReLU(alpha=hp.Float('dense_leaky_relu_alpha_2', 0.01, 0.3))(x)
        x = Dropout(hp.Float('dense_dropout_2', 0.2, 0.7))(x)
    
    # Output layer
    outputs = Dense(10, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Optimizer selection and hyperparameters
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Float('adam_lr', 1e-5, 1e-2, sampling='log'),
            beta_1=hp.Float('adam_beta_1', 0.85, 0.95),
            beta_2=hp.Float('adam_beta_2', 0.99, 0.999)
        )
    elif optimizer_choice == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=hp.Float('sgd_lr', 1e-4, 1e-1, sampling='log'),
            momentum=hp.Float('sgd_momentum', 0.8, 0.99)
        )
    else:  # rmsprop
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.Float('rmsprop_lr', 1e-5, 1e-2, sampling='log'),
            rho=hp.Float('rmsprop_rho', 0.8, 0.95)
        )
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def create_data_generators(x_train, y_train):
    """
    Create data generators with augmentation
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )
    
    train_gen = datagen.flow(x_train, y_train, batch_size=64, subset="training")
    validation_gen = datagen.flow(x_train, y_train, batch_size=64, subset="validation")
    
    return train_gen, validation_gen


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model and print detailed metrics
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Predictions for detailed metrics
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_flat = y_test.flatten()
    
    # Calculate additional metrics
    f1 = f1_score(y_test_flat, y_pred_classes, average='weighted')
    precision = precision_score(y_test_flat, y_pred_classes, average='weighted')
    recall = recall_score(y_test_flat, y_pred_classes, average='weighted')
    
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    return test_acc, test_loss, f1, precision, recall


def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run Bayesian optimization and model training
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    print("Creating data generators...")
    train_gen, validation_gen = create_data_generators(x_train, y_train)
    
    print("Setting up Bayesian Optimization with Keras Tuner...")
    
    # Create the tuner
    tuner = kt.BayesianOptimization(
        build_hypermodel,
        objective='val_accuracy',
        max_trials=20,  # Number of different hyperparameter combinations to try
        directory='bayesian_optimization',
        project_name='cnn_cifar10_optimization',
        overwrite=True
    )
    
    # Display search space
    tuner.search_space_summary()
    
    # Define callbacks for the search
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max',
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    print("Starting hyperparameter search...")
    tuner.search(
        train_gen,
        epochs=30,  # Reduced epochs for faster search
        validation_data=validation_gen,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nBest hyperparameters found:")
    print("=" * 50)
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    print("=" * 50)
    
    # Build the model with best hyperparameters
    print("\nTraining the best model...")
    best_model = tuner.hypermodel.build(best_hps)
    best_model.summary()
    
    # Train the best model with more epochs
    history = best_model.fit(
        train_gen,
        epochs=50,
        validation_data=validation_gen,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=8, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
        ],
        verbose=1
    )
    
    print("\nEvaluating the best model...")
    test_acc, test_loss, f1, precision, recall = evaluate_model(best_model, x_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the best model
    best_model.save('best_cnn_model_bayesian_opt.h5')
    print("Best model saved as 'best_cnn_model_bayesian_opt.h5'")
    
    # Optional: Generate SHAP explanations (commented out due to computational cost)
    # print("Generating SHAP explanations...")
    # explainable_ai(best_model, x_train[:100], x_test[:10])
    
    return best_model, best_hps, history


if __name__ == "__main__":
    best_model, best_hyperparameters, training_history = main()