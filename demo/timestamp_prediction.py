from funasr import AutoModel

model = AutoModel(model="fa-zh")
wav_file = f"{model.model_path}/example/asr_example.wav"
text_file = f"{model.model_path}/example/text.txt"
print('text_file:' + text_file)
res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
print(res)