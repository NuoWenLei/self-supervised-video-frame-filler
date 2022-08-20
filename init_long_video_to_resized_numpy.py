from data_loader import data_loader, np

X_prev, X_after, y = data_loader("6857.mp4")

with open("long_video_X_previous.npy", "wb") as prev_saver:
	np.save(prev_saver, X_prev)

with open("long_video_X_after.npy", "wb") as after_saver:
	np.save(after_saver, X_after)

with open("long_video_y.npy", "wb") as y_saver:
	np.save(y_saver, y)
