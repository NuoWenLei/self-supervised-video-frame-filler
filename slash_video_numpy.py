from data_loader import np

with open("long_video_X_previous-002.npy", "rb") as prev_saver:
	X_prev = np.load(prev_saver)

with open("long_video_X_after-003.npy", "rb") as after_saver:
	X_after = np.load(after_saver)

with open("long_video_y.npy", "rb") as y_saver:
	y = np.load(y_saver)

l = y.shape[0]

X_prev = X_prev[0: l // 2, ...]
X_after = X_after[0: l // 2, ...]
y = y[0: l // 2, ...]

with open("half_video_X_previous.npy", "wb") as prev_saver:
	np.save(prev_saver, X_prev)

with open("half_video_X_after.npy", "wb") as after_saver:
	np.save(after_saver, X_after)

with open("half_video_y.npy", "wb") as y_saver:
	np.save(y_saver, y)