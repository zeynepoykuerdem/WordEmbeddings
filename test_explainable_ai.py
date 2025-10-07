import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cifar_model import load_data, building_model, explainable_ai

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