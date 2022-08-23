from imports import tf, Iterable

class Conv2DMHAUnit(tf.keras.layers.Layer):

	def __init__(self,
	num_heads: int,
	d_model: int,
	image_size: tuple,
	kernel_size: Iterable,
	name: str,
	feature_activation: str = "relu",
	output_activation: str = "linear"):
		super().__init__(name = name)

		self.num_heads = num_heads
		self.d_model = d_model
		self.query_size = self.d_model // self.num_heads
		self.feature_activation = feature_activation
		self.output_activation = output_activation
		self.image_size = image_size
		self.kernel_size = kernel_size

		assert self.d_model % self.num_heads == 0, "D_model and Number of Heads do not match"

		self.num_blocks_y = self.image_size[0] // self.kernel_size[0]

		self.num_blocks_x = self.image_size[1] // self.kernel_size[1]

		self.y_complete = (self.image_size[0] % self.kernel_size[0] == 0)

		self.x_complete = (self.image_size[1] % self.kernel_size[1] == 0)

		self.y_pad_length = 0

		self.x_pad_length = 0

		if not self.y_complete:

			self.y_pad_length = self.kernel_size[0] - (self.image_size[0] % self.kernel_size[0])

			self.num_blocks_y += 1

		if not self.x_complete:

			self.x_pad_length = self.kernel_size[1] - (self.image_size[1] % self.kernel_size[1])

			self.num_blocks_x += 1

		self.y_padded = self.image_size[0] + self.y_pad_length

		self.x_padded = self.image_size[1] + self.x_pad_length
	
		self.total_num_blocks = self.num_blocks_x * self.num_blocks_y

		self.q_dense, self.k_dense, self.v_dense = [
			tf.keras.layers.Dense(self.d_model,
			activation = feature_activation,
			name = f"{name}_feature_dense_{i}") for i in range(3)]

	def call(self, X):

		X_shape = tf.shape(X)

		b, h, w, c = X_shape[0], X_shape[1], X_shape[2], X_shape[3]
		

		X_reshaped = tf.reshape(X, (b, h * w, c))

		# features: (b, h * w, d_model)

		q_features = self.q_dense(X_reshaped)
		k_features = self.k_dense(X_reshaped)
		v_features = self.v_dense(X_reshaped)

		q_features /= (self.d_model ** .5)

		# split heads
		q_heads = tf.reshape(q_features, (b, self.num_heads, h, w, self.query_size))
		k_heads = tf.reshape(k_features, (b, self.num_heads, h, w, self.query_size))
		v_heads = tf.reshape(v_features, (b, self.num_heads, h, w, self.query_size))

		# pad to allow for kernel splits
		if not (self.x_complete and self.y_complete):

			padding = tf.constant([
				[0, 0],
				[0, 0],
				[0, self.y_pad_length],
				[0, self.x_pad_length],
				[0, 0]
			])

			padded_q_heads = tf.pad(q_heads, padding) # output: (batch_size, num_heads, h padded, w padded, query_size)
			padded_k_heads = tf.pad(k_heads, padding) # output: (batch_size, num_heads, h padded, w padded, query_size)
			padded_v_heads = tf.pad(v_heads, padding) # output: (batch_size, num_heads, h padded, w padded, query_size)

		else:
			
			padded_q_heads = q_heads
			padded_k_heads = k_heads
			padded_v_heads = v_heads
		
		# reshape to add kernels
		padded_q_heads = tf.reshape(
			padded_q_heads,
			(b,
			self.num_heads,
			self.total_num_blocks,
			self.query_size,
			self.kernel_size[0],
			self.kernel_size[1])
		)

		padded_k_heads = tf.reshape(
			padded_k_heads,
			(b,
			self.num_heads,
			self.total_num_blocks,
			self.query_size,
			self.kernel_size[0],
			self.kernel_size[1])
		)

		padded_v_heads = tf.reshape(
			padded_v_heads,
			(b,
			self.num_heads,
			self.total_num_blocks,
			self.query_size,
			self.kernel_size[0],
			self.kernel_size[1])
		)

		# create attention score
		attention = tf.einsum("...ijk,...njk->...injk", padded_q_heads, padded_k_heads)
		softmax_attention_score = tf.math.softmax(attention)

		# Use einsum for matmul with different ranked tensors
		# output shape: (b, num_heads, num_blocks, query_size, kernel_size_h, kernel_size_w)
		self_attention_value_padded_unreshaped = tf.einsum("...injk,...njk->...ijk", softmax_attention_score, padded_v_heads)

		self_attention_value_padded = tf.reshape(self_attention_value_padded_unreshaped,
		(b, self.y_padded, self.x_padded, self.d_model))

		self_attention_value = self_attention_value_padded[:, :h, :w, :]

		return self_attention_value
	








		
