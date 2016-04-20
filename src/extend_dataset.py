import librosa
from numpy import append, split

def extend_dataset(y, sr):
	# Make 2x faster
	D       = librosa.stft(y, n_fft=2048, hop_length=512)
	
	D_fast  = librosa.phase_vocoder(D, 2.0, hop_length=512)
	y_fast  = librosa.istft(D_fast, hop_length=512)

	# Concatenate two 2x frames together
	y_fast = append(y_fast, y_fast)
	y_fast = y_fast[:len(y)]

	# Make 2x slower
	D_slow  = librosa.phase_vocoder(D, 0.5, hop_length=512)
	y_slow  = librosa.istft(D_slow, hop_length=512)

	# split two 0.5x frames together
	y_slow1, y_slow2 = split(y_slow, 2)
	y_slow1 = y_slow1[:len(y)]
	y_slow2 = y_slow2[:len(y)]

	## Frequency scaling
	y_pitch_up = librosa.effects.pitch_shift(y, sr, n_steps=4)
	y_pitch_up = y_pitch_up[:len(y)]
	y_pitch_down = librosa.effects.pitch_shift(y, sr, n_steps=-4)
	y_pitch_down = y_pitch_down[:len(y)]

	return (y_fast, y_slow1, y_slow2, y_pitch_up, y_pitch_down)

if __name__ == '__main__':
	y, sr   = librosa.load(librosa.util.example_audio_file())
	print(extend_dataset(y, sr))
