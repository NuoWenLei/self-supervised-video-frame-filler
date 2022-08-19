from imports import tf, NoDependency
from conv2dmhaunit import Conv2DMHAUnit

def bidirectional_conv_lstm_attention_bottleneck_model(
	# overall parameters
	d_model: int = 64,
	num_unet_sections: int = 3,
	num_layers_attention: int = 2,

	# conv lstm parameters
	convlstm_kernel_size: tuple = (3, 3),
	activation: str = "relu",
	image_dims: dict = {
		"0": 64,
		"1": 64,
		"2": 3
	},
	seq_len_prev_and_after: int = 3,

	# attention bottleneck parameters
	attention_bottleneck_multiple: int = 2,
	num_heads: int = 4,
	attention_kernel_size: tuple = (4, 4),
	attention_feature_activation: str = "relu",
	attention_output_activation: str = "linear",

	# UNet Upsample parameters
	unet_upsample_kernel: tuple = (3, 3),
	prediction_activation: str = "tanh",
	):

	residual_tensors = []

	# init variables to keep track of image dimension and size
	curr_image_dims = image_dims
	# curr_image_size = curr_image_dims["0"] * curr_image_dims["1"]

	prev_x = tf.keras.layers.Input(shape = (seq_len_prev_and_after,) + (image_dims["0"], image_dims["1"], image_dims["2"]))
	after_x = tf.keras.layers.Input(shape = (seq_len_prev_and_after,) + (image_dims["0"], image_dims["1"], image_dims["2"]))

	for i in range(num_unet_sections):	
		prev_x = tf.keras.layers.ConvLSTM2D(d_model, convlstm_kernel_size, activation = activation, return_sequences = True)(prev_x)
		after_x = tf.keras.layers.ConvLSTM2D(d_model, convlstm_kernel_size, activation = activation, return_sequences = True)(after_x)
		combined_x = tf.reduce_sum(tf.concat([prev_x, after_x], axis = -1), axis = -4) / tf.cast((d_model * 2) ** .5, tf.float32)
		residual_tensors.append(combined_x)
		prev_x = tf.keras.layers.MaxPool3D((1, 2, 2))(prev_x)
		after_x = tf.keras.layers.MaxPool3D((1, 2, 2))(after_x)
		curr_image_dims["0"] = curr_image_dims["0"] // 2
		curr_image_dims["1"] = curr_image_dims["1"] // 2

	prev_conv = tf.keras.layers.ConvLSTM2D(d_model, convlstm_kernel_size, activation = activation, return_sequences = False)(prev_x)
	after_conv = tf.keras.layers.ConvLSTM2D(d_model, convlstm_kernel_size, activation = activation, return_sequences = False)(after_x)

	"""
	* * * a
	* * * *
	p * * *

	  |
	  V

	* | * | * | a
	* | * | * | *
	* | * | * | *
	p | * | * | *
	"""

	channels = tf.shape(prev_conv)[-1]

	attention_bottleneck_block_width = tf.shape(prev_conv)[-2]

	attention_bottleneck_block_height = tf.shape(prev_conv)[-3]

	curr_image_dims["0"] = curr_image_dims["0"] * (attention_bottleneck_multiple + 2)
	curr_image_dims["1"] = curr_image_dims["1"] * (attention_bottleneck_multiple + 2)

	b = tf.shape(prev_conv)[0]

	vert_cols = []

	vert_strip_prev = tf.zeros((b, (attention_bottleneck_multiple + 1) * attention_bottleneck_block_height, attention_bottleneck_block_width, channels))

	vert_col_prev = tf.concat([vert_strip_prev, prev_conv], axis = -3)

	vert_cols.append(vert_col_prev)

	for i in range(attention_bottleneck_multiple):
		
		vert_strip = tf.zeros((b, (attention_bottleneck_multiple + 2) * attention_bottleneck_block_height, attention_bottleneck_block_width, channels))

		vert_cols.append(vert_strip)
	
	vert_strip_after = tf.zeros((b, (attention_bottleneck_multiple + 1) * attention_bottleneck_block_height, attention_bottleneck_block_width, channels))

	vert_col_after = tf.concat([after_conv, vert_strip_after], axis = -3)

	vert_cols.append(vert_col_after)

	attention_bottleneck = tf.concat(vert_cols, axis = -2)

	h = tf.shape(attention_bottleneck)[-3]
	w = tf.shape(attention_bottleneck)[-2]

	for i in range(num_layers_attention):

		attention_bottleneck = Conv2DMHAUnit(
			num_heads = num_heads,
			d_model = d_model,
			image_size = NoDependency((image_dims["0"], image_dims["1"])),
			kernel_size = attention_kernel_size,
			name = f"Conv2DMHAUnit_{i}",
			feature_activation = attention_feature_activation,
			output_activation = attention_output_activation
		)(attention_bottleneck)
	
	prev_attention = attention_bottleneck[..., (attention_bottleneck_multiple + 1) * attention_bottleneck_block_height:, :attention_bottleneck_block_width, :]

	after_attention = attention_bottleneck[..., :attention_bottleneck_block_height, (attention_bottleneck_multiple + 1) * attention_bottleneck_block_width:, :]

	concatted_attention = tf.concat([prev_attention, after_attention], axis = -1)

	att_upsample = concatted_attention

	for i in range(num_unet_sections):
		att_upsample = tf.keras.layers.UpSampling2D()(att_upsample)
		residual_section = residual_tensors.pop()
		residual_added_att = tf.concat([att_upsample, residual_section], axis = -1)
		att_upsample = tf.keras.layers.Conv2D(
			d_model,
			unet_upsample_kernel,
			activation = activation
			)(residual_added_att)
	
	final_conv_prediction = tf.keras.layers.Conv2D(3, unet_upsample_kernel, activation = prediction_activation)

	return final_conv_prediction

