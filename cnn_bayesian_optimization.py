import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, BatchNormalization
from tensorflow.keras import models, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import RandomRotation, RandomFlip, Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shap
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
import warnings
warnings.filterwarnings('ignore')

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
    Generate SHAP explanations for the model
    """
    explainer = shap.GradientExplainer(model, train_images)
    shap_values = explainer.shap_values(test_images[:3])
    # we are gonna have 10 outputs, 10 explanations(one for each output)
    shap.image_plot([shap_values[i] for i in range(10)], test_images[:3])

def building_model(learning_rate=0.001, dropout_rate=0.2, l2_reg=0.0005, 
                  conv1_filters=64, conv2_filters=128, dense_units=64,
                  batch_size=64, epochs=20):
    """
    Build CNN model with hyperparameters
    """
    inputs = Input(shape=(32, 32, 3))
    x = inputs
    
    # CONV LAYERS
    x = Conv2D(conv1_filters, (7, 7), padding='same', 
               kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(conv2_filters, (7, 7), padding='same', 
               kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(dense_units)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(10, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    return model

def objective_function(params, x_train, y_train, x_val, y_val):
    """
    Objective function for Bayesian optimization
    Returns negative validation accuracy (since we want to maximize accuracy)
    """
    learning_rate, dropout_rate, l2_reg, conv1_filters, conv2_filters, dense_units = params
    
    # Convert to appropriate types
    conv1_filters = int(conv1_filters)
    conv2_filters = int(conv2_filters)
    dense_units = int(dense_units)
    
    try:
        # Build model with current hyperparameters
        model = building_model(
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            conv1_filters=conv1_filters,
            conv2_filters=conv2_filters,
            dense_units=dense_units
        )
        
        # Train model with early stopping
        history = model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=20,  # Reduced for faster optimization
            validation_data=(x_val, y_val),
            callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, 
                                   mode='max', restore_best_weights=True)],
            verbose=0
        )
        
        # Get best validation accuracy
        best_val_acc = max(history.history['val_accuracy'])
        
        # Return negative accuracy (since we want to maximize)
        return -best_val_acc
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1.0  # Return high loss for failed runs

def bayesian_optimization(x_train, y_train, x_val, y_val, n_calls=20):
    """
    Perform Bayesian optimization to find best hyperparameters
    """
    # Define search space
    dimensions = [
        Real(0.0001, 0.01, name='learning_rate', prior='log-uniform'),
        Real(0.1, 0.5, name='dropout_rate'),
        Real(0.0001, 0.01, name='l2_reg', prior='log-uniform'),
        Integer(32, 128, name='conv1_filters'),
        Integer(64, 256, name='conv2_filters'),
        Integer(32, 128, name='dense_units')
    ]
    
    # Create objective function with fixed data
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        return objective_function(
            [params['learning_rate'], params['dropout_rate'], params['l2_reg'],
             params['conv1_filters'], params['conv2_filters'], params['dense_units']],
            x_train, y_train, x_val, y_val
        )
    
    # Run Bayesian optimization
    print("Starting Bayesian optimization...")
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=42,
        acq_func='EI'  # Expected Improvement acquisition function
    )
    
    return result

def main():
    """
    Main function to run Bayesian optimization and model training
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Split training data into train and validation
    split_idx = int(0.8 * len(x_train))
    x_train_split = x_train[:split_idx]
    y_train_split = y_train[:split_idx]
    x_val_split = x_train[split_idx:]
    y_val_split = y_train[split_idx:]
    
    print(f"Training set: {x_train_split.shape[0]} samples")
    print(f"Validation set: {x_val_split.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    
    # Perform Bayesian optimization
    print("\n" + "="*50)
    print("BAYESIAN OPTIMIZATION")
    print("="*50)
    
    optimization_result = bayesian_optimization(
        x_train_split, y_train_split, x_val_split, y_val_split, n_calls=15
    )
    
    # Get best parameters
    best_params = optimization_result.x
    best_score = -optimization_result.fun  # Convert back to positive accuracy
    
    print(f"\nBest validation accuracy: {best_score:.4f}")
    print("Best hyperparameters:")
    param_names = ['learning_rate', 'dropout_rate', 'l2_reg', 
                   'conv1_filters', 'conv2_filters', 'dense_units']
    for name, value in zip(param_names, best_params):
        if name in ['conv1_filters', 'conv2_filters', 'dense_units']:
            print(f"  {name}: {int(value)}")
        else:
            print(f"  {name}: {value:.6f}")
    
    # Train final model with best parameters
    print("\n" + "="*50)
    print("TRAINING FINAL MODEL")
    print("="*50)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )
    
    train_gen = datagen.flow(x_train, y_train, batch_size=64, subset="training")
    validation_gen = datagen.flow(x_train, y_train, batch_size=64, subset="validation")
    
    # Build final model with best parameters
    final_model = building_model(
        learning_rate=best_params[0],
        dropout_rate=best_params[1],
        l2_reg=best_params[2],
        conv1_filters=int(best_params[3]),
        conv2_filters=int(best_params[4]),
        dense_units=int(best_params[5])
    )
    
    print("\nFinal model architecture:")
    final_model.summary()
    
    # Train final model
    history = final_model.fit(
        train_gen,
        epochs=50,
        validation_data=validation_gen,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=5, mode='max', 
                         restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                             min_lr=1e-7)
        ],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=0)
    print(f'\nFinal Test Accuracy: {test_acc:.4f}')
    print(f'Final Test Loss: {test_loss:.4f}')
    
    # Generate predictions for detailed metrics
    y_pred = final_model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate additional metrics
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    
    print(f'\nDetailed Metrics:')
    print(f'F1-Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot optimization convergence
    plt.figure(figsize=(10, 6))
    plot_convergence(optimization_result)
    plt.title('Bayesian Optimization Convergence')
    plt.show()
    
    # Optional: Generate SHAP explanations (uncomment if needed)
    # print("\nGenerating SHAP explanations...")
    # explainable_ai(final_model, x_train[:100], x_test[:3])
    
    return final_model, optimization_result

if __name__ == "__main__":
    model, opt_result = main()