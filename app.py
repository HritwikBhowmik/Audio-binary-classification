
from flask import Flask, request
from flask import url_for
import torch
import torch.nn as nn
import torchaudio
from Model import AudioBinaryClassifierV2
from Model import CNNNetwork
from AudioFolder import audioFileTransformer

app = Flask('myApp')

model = CNNNetwork(in_shapes=1, out_shapes=1) # this model with 87%
# model_1 = AudioBinaryClassifierV2(in_shapes=1, out_shapes=1) # this model with 75%


def getPrediction(filePath:str):

    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    N_FFT = 1064
    melSpec_transform = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                                            n_fft = N_FFT,
                                                                            hop_length=512,
                                                                            n_mels=64),
                                     torchaudio.transforms.AmplitudeToDB())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformed_audio = audioFileTransformer(filePath=filePath,
                                         transform=melSpec_transform,
                                         targ_sr=SAMPLE_RATE,
                                         num_samples=NUM_SAMPLES,
                                         device=device)
    signal = transformed_audio[0]

    # load trained model
    model.load_state_dict(state_dict=torch.load(f="Model_Binary_Audio_Classification.pth"))
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y_logit = model(signal.unsqueeze(dim=0))
        y_pred = torch.round(torch.sigmoid(y_logit))
    
    if y_pred == 0:
        pred_label = 'non-horn'
    if y_pred == 1:
        pred_label = 'horn'

    return pred_label

@app.route('/prediction', methods=['GET'])
def APICall():
    filePath = request.args.get('filePath')
    return getPrediction(filePath=filePath)

with app.test_request_context():
    print(url_for('APICall', filePath="audio/input/myAudioInput.wav"))

if __name__ == '__main__':
    print()
    print("[+] API key: http://127.0.0.1:1010//prediction?filePath=audio/input/myAudioInput.wav")
    print()
    app.run(port=1010)