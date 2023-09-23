import librosa
import os

folder1 = '/media/jaehee/LargeDB/VCTK/original_wav/clean_trainset_28spk_wav_16KHz'
folder2 = '/media/jaehee/LargeDB/VCTK/original_wav/clean_testset_wav_16KHz'
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

wav_len = []
for f in files1:
    wav, _ = librosa.load(folder1+'/'+f, sr=16000)
    print(len(wav))
    wav_len.append(len(wav))
print('------------------------')

for f in files2:
    wav, _ = librosa.load(folder2+'/'+f, sr=16000)
    print(len(wav))
    wav_len.append(len(wav))

print("min : ", min(wav_len))
print("max : ", max(wav_len))
print("count : ", len(wav_len))