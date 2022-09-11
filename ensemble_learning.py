from tensorflow import keras
from keras.metrics import accuracy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import librosa
import pandas as pd

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result


# loading the models
CNN_model = keras.models.load_model('/ser_cnn_4.h5')
LSTM_model = keras.models.load_model('/ser_lstm_4.h5')
CNN_SVM_model = keras.models.load_model('/ser_cnnsvm_6.h5')

# loading the new data to be predected 
path ="/dataset/new_sample.wav"
X_new = []

data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

features_extracted = extract_features(data, sample_rate)
X_new = np.array(features_extracted)
Features = pd.DataFrame(X_new)

Y = np.load("features_values.npy", allow_pickle=True)
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
lables = encoder.get_feature_names()
scaler = StandardScaler()
x_test = scaler.fit_transform(Features)

# making our data compatible to model.
x_test = np.expand_dims(x_test, axis=0)


# performing the majority voting
models = [CNN_SVM_model, CNN_model, LSTM_model]
predictions = [model_chosen.predict(x_test) for model_chosen in models]
predictions = np.array(predictions)
sum_pred = np.sum(predictions, axis=0)

ensemble_prediction = np.argmax(sum_pred, axis=1)
y_pred = lables[ensemble_prediction[0]]
print('The predicted class when using the majority voting : ', y_pred[3:])


predict_cnn_svm= CNN_SVM_model.predict(x_test)
prediction_1 = np.argmax(predict_cnn_svm,axis=1)
y_pred_1 = encoder.inverse_transform(prediction_1)
print('The predicted class when using the CNN-SVM model : ', y_pred_1[0][0])

predict_cnn= CNN_model.predict(x_test)
prediction_2 = np.argmax(predict_cnn,axis=1)
y_pred_2 = encoder.inverse_transform(prediction_2)
print('The predicted class when using the CNN model : ', y_pred_2[0][0])

predict_lstm= LSTM_model.predict(x_test)
prediction_3 = np.argmax(predict_lstm,axis=1)
y_pred_3 = encoder.inverse_transform(prediction_3)
print('The predicted class when using the LSTM model : ', y_pred_3[0][0])
	
