from imports import tf
import skvideo
skvideo.setFFmpegPath('/Users/nuowenlei/audio-orchestrator-ffmpeg/bin/')
import skvideo.io as skvideo_io
import numpy as np



def frame_masking(frames, num_elements_front_and_back = 3):
	X_prev = []
	X_after = []
	y = []
	for i in range(num_elements_front_and_back, len(frames) - num_elements_front_and_back):

		previous_frames = frames[i-num_elements_front_and_back:i]

		# after frames specifically ordered to go from future to present (in that direction)
		after_frames = frames[i+num_elements_front_and_back:i:-1]
		X_prev.append(previous_frames)
		X_after.append(after_frames)

		y.append(frames[i])

	return np.array(X_prev), np.array(X_after), np.array(y)

def data_loader(video_path, image_height = 128, image_width = 128, num_elements_front_and_back = 3):
	video_data = skvideo_io.vread(video_path)
	video_data_numpy = np.array(video_data)
	video_data_resized = tf.image.resize_with_pad(video_data_numpy, image_height, image_width).numpy()
	print(video_data_resized.shape)
	X_prev, X_after, y = frame_masking(video_data_resized, num_elements_front_and_back=num_elements_front_and_back)
	return X_prev, X_after, y

