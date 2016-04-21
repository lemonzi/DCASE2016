import matplotlib
matplotlib.use('Agg')
import librosa
from numpy import append, split

def extend_dataset(y, sr):

        #return (y,)

	# Make 2x faster
	D       = librosa.stft(y, n_fft=2048, hop_length=512)

	D_fast  = librosa.phase_vocoder(D, 2.0, hop_length=512)
	y_fast  = librosa.istft(D_fast, hop_length=512)

	# Concatenate two 2x frames together
	y_fast = append(y_fast, y_fast)

	# Make 2x slower
	D_slow  = librosa.phase_vocoder(D, 0.5, hop_length=512)
	y_slow  = librosa.istft(D_slow, hop_length=512)

	# split two 0.5x frames together
	y_slow1, y_slow2 = split(y_slow, 2)

	## Frequency scaling
	#y_pitch_up = librosa.effects.pitch_shift(y, sr, n_steps=4)
	#y_pitch_down = librosa.effects.pitch_shift(y, sr, n_steps=-4)

        samples = min([len(y), len(y_fast), len(y_slow1), len(y_slow2)])
        y = y[:samples]
        y_fast = y_fast[:samples]
        y_slow1 = y_slow1[:samples]
        y_slow2 = y_slow2[:samples]

	return (y, y_fast, y_slow1, y_slow2)

if __name__ == '__main__':
	y, sr   = librosa.load(librosa.util.example_audio_file())
	print(extend_dataset(y, sr))
