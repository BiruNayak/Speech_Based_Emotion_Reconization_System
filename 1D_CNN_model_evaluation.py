import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from config import SAVE_DIR_PATH, MODEL_DIR_PATH, RESULT_DIR_PATH

def load_data():
    """
    Load test data and labels.

    Returns:
        X_test (numpy.ndarray): Test data.
        y_test (numpy.ndarray): Test labels.
    """
    X = joblib.load(os.path.join(SAVE_DIR_PATH, 'X.joblib'))
    y = joblib.load(os.path.join(SAVE_DIR_PATH, 'y.joblib'))
    X = X.reshape(X.shape[0], 3, 1, 40)  # Reshape the data for CNN input
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test

def load_trained_model(model_folder):
    """
    Load the trained model from the specified folder.

    Parameters:
        model_folder (str): Folder path containing the trained model file.

    Returns:
        model (keras.models.Sequential): Loaded trained model.
    """
    model_name = 'Emotion_Voice_Detection_Model.h5'
    model_path = os.path.join(model_folder, model_name)
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file '{model_name}' not found in the specified folder.")

def generate_human_readable_labels(labels):
    """
    Generate human-readable labels.

    Parameters:
        labels (numpy.ndarray): Numeric labels.

    Returns:
        human_readable_labels (list): Human-readable labels.
    """
    label_map = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}
    return [label_map[label] for label in labels]

def generate_classification_report(model, X_test, y_test, save_folder=None):
    """
    Generate a classification report and save it.

    Parameters:
        model (keras.models.Sequential): Trained model.
        X_test (numpy.ndarray): Test data.
        y_test (numpy.ndarray): Test labels.
        save_folder (str): Folder path to save the report.
    """
    predictions = model.predict(X_test)
    new_max_values = [np.argmax(sublist) for sublist in predictions]
    new_y_test = y_test.astype(int)
    target_names = generate_human_readable_labels(np.unique(y_test))
    report = classification_report(new_y_test, new_max_values, target_names=target_names)
    print(report)
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        report_text_path = os.path.join(save_folder, 'classification_report.txt')
        with open(report_text_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved as text at: {report_text_path}")

        # Save classification report as PNG
        plt.figure(figsize=(8, 6))
        plt.text(0, 1, report, va='top', fontsize=10, family='monospace')
        plt.axis('off')
        report_image_path = os.path.join(save_folder, 'classification_report.png')
        plt.savefig(report_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Classification report saved as image at: {report_image_path}")

def generate_confusion_matrix(y_test, predictions, save_folder=None):
    """
    Generate a confusion matrix and save it.

    Parameters:
        y_test (numpy.ndarray): Test labels.
        predictions (numpy.ndarray): Model predictions.
        save_folder (str): Folder path to save the confusion matrix plot.
    """
    labels = generate_human_readable_labels(np.unique(y_test))
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'))
        print(f"Confusion matrix saved as image at: {os.path.join(save_folder, 'confusion_matrix.png')}")
    plt.show()

def evaluate_results(model_folder, save_folder):
    """
    Evaluate the trained model and save the evaluation results.

    Parameters:
        model_folder (str): Folder path containing the trained model file.
        save_folder (str): Folder path to save the evaluation results.
    """
    X_test, y_test = load_data()
    model = load_trained_model(model_folder)
    generate_classification_report(model, X_test, y_test, save_folder)
    predictions = model.predict(X_test)
    new_max_values = [np.argmax(sublist) for sublist in predictions]
    generate_confusion_matrix(y_test, new_max_values, save_folder)

if __name__ == "__main__":
    evaluate_results(MODEL_DIR_PATH, RESULT_DIR_PATH)
