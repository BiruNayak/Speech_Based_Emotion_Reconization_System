import os
import joblib
import numpy as np
from keras.layers import Dense, Conv1D, Flatten, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from config import SAVE_DIR_PATH, MODEL_DIR_PATH, RESULT_DIR_PATH

class TrainModel:

    @staticmethod
    def train_neural_network(X, y, model_folder=None) -> None:
        """
        This function trains the neural network.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        model = Sequential()
        model.add(Conv1D(64, 5, padding='same', input_shape=(3, 1, 40)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(8))
        model.add(Activation('softmax'))

        print(model.summary())

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=RMSprop(learning_rate=0.001),
                      metrics=['accuracy'])

        cnn_history = model.fit(x_traincnn, y_train,
                                batch_size=32, epochs=500,
                                validation_data=(x_testcnn, y_test))

        if model_folder:
            model_name = 'Emotion_Voice_Detection_Model.h5'
            model_path = os.path.join(model_folder, model_name)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model.save(model_path)
            print(f'Saved trained model at {model_path}')
        
        # Plot loss
        plt.plot(cnn_history.history['loss'])
        plt.plot(cnn_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        loss_plot_path = os.path.join(RESULT_DIR_PATH, 'model_loss.png')
        if not os.path.exists(RESULT_DIR_PATH):
            os.makedirs(RESULT_DIR_PATH)
        plt.savefig(loss_plot_path)  # Save loss plot
        plt.show()

        # Plot accuracy
        plt.plot(cnn_history.history['accuracy'])
        plt.plot(cnn_history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        accuracy_plot_path = os.path.join(RESULT_DIR_PATH, 'model_accuracy.png')
        if not os.path.exists(RESULT_DIR_PATH):
            os.makedirs(RESULT_DIR_PATH)
        plt.savefig(accuracy_plot_path)  # Save accuracy plot
        plt.show()


if __name__ == '__main__':
    print('Training started')
    X = joblib.load(os.path.join(SAVE_DIR_PATH, 'X.joblib'))
    y = joblib.load(os.path.join(SAVE_DIR_PATH, 'y.joblib'))
    model_folder = MODEL_DIR_PATH   # Specify the folder where you want to save the model
    TrainModel.train_neural_network(X=X, y=y, model_folder=model_folder)
