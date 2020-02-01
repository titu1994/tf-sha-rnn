import tensorflow as tf


class VarianceScalingV2(tf.keras.initializers.VarianceScaling):

    def __init__(self, std=1.0,
                 variance_scale=1.0,
                 mode="fan_in",
                 distribution="truncated_normal",
                 seed=None):
        super(VarianceScalingV2, self).__init__(scale=variance_scale, mode=mode,
                                                distribution=distribution,
                                                seed=seed)

        self.std = std

    def __call__(self, shape, dtype=tf.float32):
        out = super(VarianceScalingV2, self).__call__(shape, dtype=dtype)
        scaled_out = self.std * out
        return scaled_out


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None, training=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
      dropout: Either a Dropout layer or None. If Dropout layer is provider,
            ensure to pass `training` flag as well.

    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    if dropout is not None:
        attention_weights = dropout(attention_weights, training=training)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# From: https://github.com/Smerity/sha-rnn/blob/master/model.py#L326
class GELU(tf.keras.layers.Layer):

    def __init__(self):
        super(GELU, self).__init__()
        self.supports_masking = True

    def call(self, x):
        # Approximate form GELU
        return x * tf.nn.sigmoid(1.702 * x)


class Overparam(tf.keras.layers.Layer):

    def __init__(self, num_hidden):
        super(Overparam, self).__init__()

        self.num_hidden = num_hidden
        self.l1 = tf.keras.layers.Dense(2 * num_hidden, kernel_initializer=VarianceScalingV2(0.1))

    def call(self, inputs, training=None, mask=None):
        activation = self.l1(inputs)
        c, f = tf.split(activation, num_or_size_splits=2, axis=-1)
        f = tf.nn.sigmoid(f)
        c = tf.nn.tanh(c)
        return f * c


class Attention(tf.keras.layers.Layer):
    
    def __init__(self, num_hidden, num_heads, q=True, k=False, v=False, r=False, dropout=None):
        super(Attention, self).__init__()

        assert num_hidden % num_heads == 0, 'Heads must divide vector evenly'

        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.depth = num_hidden // num_heads

        self.qs = tf.Variable(tf.zeros([1, 1, num_hidden]))
        self.ks = tf.Variable(tf.zeros([1, 1, num_hidden]))
        self.vs = tf.Variable(tf.zeros([1, 1, num_hidden]))

        self.qkvs = tf.Variable(tf.zeros([1, 3, num_hidden]))

        self.dropout = tf.keras.layers.Dropout(dropout) if dropout is not None else tf.keras.layers.Dropout(0.)
        self.gelu = GELU()

        self.q = tf.keras.layers.Dense(num_hidden, kernel_initializer=VarianceScalingV2(0.1)) if q else None
        self.k = tf.keras.layers.Dense(num_hidden, kernel_initializer=VarianceScalingV2(0.1)) if k else None
        self.v = tf.keras.layers.Dense(num_hidden, kernel_initializer=VarianceScalingV2(0.1)) if v else None
        self.r = tf.keras.layers.Dense(num_hidden, kernel_initializer=VarianceScalingV2(0.1)) if r else None
        self.r_gate = tf.Variable(tf.ones([1, 1, num_hidden]))

        self.qln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                      gamma_initializer=VarianceScalingV2(0.1))

        self.vq = Overparam(num_hidden)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, training=None, mask=None):
        qs = tf.nn.sigmoid(self.qs)
        ks = tf.nn.sigmoid(self.ks)
        vs = tf.nn.sigmoid(self.vs)
        vs = self.vq(vs)

        if self.q:
            query = self.q(query)
            query = self.qln(query)

        if self.k:
            key = self.k(key)

        if self.v:
            value = self.v(value)

        q = qs * query
        k = ks * key
        v = vs * value

        q = self.dropout(q, training=training)
        v = self.dropout(v, training=training)

        original_q = tf.identity(q)

        batch_size = tf.shape(q)[0]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        mix, focus = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout, training=training)
        mix = tf.transpose(mix, [0, 2, 1, 3])
        mix = tf.reshape(mix, [batch_size, -1, self.num_hidden])

        if self.r:
            r = tf.concat([mix, original_q], axis=-1)
            r = self.dropout(r, training=training)
            r = self.gelu(self.r(r))
            mix = tf.nn.sigmoid(self.r_gate) * mix + r

        return mix, focus


class Boom(tf.keras.layers.Layer):

    def __init__(self, d_model, d_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()

        self.shortcut = shortcut
        self.linear = tf.keras.layers.Dense(d_feedforward, kernel_initializer=VarianceScalingV2(0.1))
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout else tf.keras.layers.Dropout(0.)

        self.shortcut_ff = None
        if not self.shortcut:
            self.shortcut_ff = tf.keras.layers.Dense(d_model)

        self.act = GELU()

    def call(self, inputs, training=None):
        x = self.act(self.linear(inputs))
        x = self.dropout(x, training=training)

        if self.shortcut:
            num_inp = tf.shape(inputs)[-1]
            x_shape = tf.shape(x)
            num_out = x_shape[-1]
            clip_out = num_out // num_inp * num_inp

            x = x[..., 0:clip_out]
            x_shape = tf.concat([x_shape[:-1], [num_out // num_inp, num_inp]], axis=0)
            x = tf.reshape(x, x_shape)

            z = tf.reduce_sum(x, axis=-2)

        else:
            z = self.shortcut_ff(x)

        return z


class SHARNNBlock(tf.keras.layers.Layer):

    def __init__(self, inp_dim, hidden_dim, heads=1, dropout=None, rnn=False, residual=True, use_attn=True):
        super(SHARNNBlock, self).__init__()

        self.attn = None
        if use_attn:
            self.attn = Attention(inp_dim, num_heads=heads, r=False, dropout=dropout)

        self.ff = Boom(inp_dim, hidden_dim, dropout=dropout, shortcut=True)

        self.ln_start = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                           gamma_initializer=VarianceScalingV2(0.1))
        self.ln_mid = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                         gamma_initializer=VarianceScalingV2(0.1))
        self.ln_mem = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                         gamma_initializer=VarianceScalingV2(0.1))
        self.ln_out = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                         gamma_initializer=VarianceScalingV2(0.1))
        self.ln_ff = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                        gamma_initializer=VarianceScalingV2(0.1))
        self.ln_xff = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12,
                                                         gamma_initializer=VarianceScalingV2(0.1))
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout is not None else tf.keras.layers.Dropout(0.)
        self.gelu = GELU()
        self.residual = residual

        self.rnn = None
        if rnn:
            self.rnn = tf.keras.layers.LSTM(inp_dim, return_state=True, return_sequences=True)

    def call(self, h, positional_encoding, attention_mask, mem=None, hidden=None, training=None):
        new_mem = None

        h = self.ln_start(h)

        if self.rnn:
            out = self.rnn(h, training=training, initial_state=hidden)
            x, hidden = out[0], out[1:]

            num_inp = tf.shape(h)[-1]
            x_shape = tf.shape(x)
            num_out = x_shape[-1]
            clip_out = num_out // num_inp * num_inp

            z = x[..., 0:clip_out]
            z_shape = tf.concat([x_shape[:-1], [num_out // num_inp, num_inp]], axis=0)
            z = tf.reshape(z, z_shape)

            x = self.dropout(z, training=training)
            x = tf.reduce_sum(x, axis=-2)

            if self.residual:
                h = h + x
            else:
                h = x

        focus = None
        new_mem = []

        if self.attn is not None:
            mh = self.ln_mem(h)
            h = self.ln_mid(h)

            if mem is not None:
                bigh = tf.concat([mem, mh], axis=0)
            else:
                bigh = mh

            new_mem = bigh[-tf.shape(positional_encoding)[0]:]

            q, k = h, bigh
            x, focus = self.attn(q, k, bigh, mask=attention_mask, training=training)
            x = self.dropout(x, training=training)
            h = x + h

        if self.ff:
            h = self.ln_ff(h)
            x = self.ln_xff(h)

            x = self.ff(x, training=training)
            x = self.dropout(x)
            h = x + h

        return h, new_mem, hidden, focus


class SHARNN(tf.keras.Model):

    def __init__(self, num_token, embed_dim, num_hid, num_layers,
                 dropout=0.5, dropout_h=0.5, dropout_i=0.5,
                 return_hidden=True, return_mem=True):
        super(SHARNN, self).__init__()

        num_embeddings = num_token
        embed_dim = embed_dim
        hidden_dim = num_hid

        self.num_inp = embed_dim
        self.num_hidden = num_hid
        self.num_layers = num_layers

        self.num_max_positions = 5000
        self.num_heads = 1
        self.causal = True

        self.encoder = tf.keras.layers.Embedding(num_embeddings, embed_dim,
                                                 embeddings_initializer=VarianceScalingV2(0.1))

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.i_dropout = tf.keras.layers.Dropout(dropout_i)

        self.blocks = []
        for idx in range(num_layers):
            rnn = True
            block = SHARNNBlock(embed_dim, hidden_dim, self.num_heads, dropout_h, rnn=rnn, residual=False,
                                use_attn=True if idx == num_layers - 2 else False)

            self.blocks.append(block)

        self.pos_embedding = [0] * self.num_max_positions
        # self.decoder = tf.keras.layers.Dense(num_embeddings)

        self.return_hidden = return_hidden
        self.return_mem = return_mem

    def call(self, inputs, hidden=None, mems=None, training=None, mask=None):
        """ Input has shape [batch, seq length] """
        e = self.encoder(inputs)
        e = self.i_dropout(e, training=training)

        batchsize = tf.shape(inputs)[0]
        in_seqlen = tf.shape(inputs)[1]
        out_seqlen = tf.shape(e)[1]

        # TODO: See how to allow limiting the amount of memory
        # if mems is not None:
        #     max_mem = tf.cast(self.num_max_positions - out_seqlen, tf.int32)
        #     mems = [m[-max_mem:] for m in mems]

        # total_len = in_seqlen + (len(mems[0]) if mems else 0)

        positional_encoding = self.pos_embedding
        h = e

        new_hidden = []
        new_mems = []

        attn_mask = tf.ones([in_seqlen, in_seqlen])
        attn_mask = 1. - tf.linalg.band_part(attn_mask, -1, 0)

        if mems:
            m_shapes = [tf.shape(m) for m in mems]
            m_seqlen = [m[1] if len(m) > 1 else m[0] for m in m_shapes]
            max_mems = tf.reduce_max(m_seqlen)

            happy = tf.zeros([in_seqlen, max_mems])
            attn_mask = tf.concat([happy, attn_mask], axis=-1)

        for idx, block in enumerate(self.blocks):
            mem = mems[idx] if mems else None
            hid = hidden[idx] if hidden else None
            h, m, nh, f = block(h, positional_encoding, attn_mask, mem=mem, hidden=hid, training=training)

            new_hidden.append(nh)
            new_mems.append(m)

        h = self.dropout(h, training=training)

        output = [h]

        if self.return_hidden:
            output.append(new_hidden)

        if self.return_mem:
            output.append(new_mems)

        if len(output) == 1:
            output = output[0]

        return output


@tf.function
def model_forward_with_grads(model, x):
    with tf.GradientTape() as tape:
        h, new_hidden, new_mems = model(x, training=True)
        h, new_hidden, new_mems = model(x, hidden=new_hidden, mems=new_mems, training=True)
        # h = model(x, training=True)

        loss = tf.reduce_sum(h)

    grad = tape.gradient(loss, model.trainable_variables)

    return loss, grad


if __name__ == '__main__':

    model = SHARNN(num_token=1000, embed_dim=100, num_hid=200, num_layers=2,
                   return_hidden=True, return_mem=True)

    model.compile(optimizer='adam', loss='mse')

    # with tf.GradientTape() as tape:
    #     x = tf.zeros([10, 25], dtype=tf.int32)
    #
    #     h, new_hidden, new_mems = model(x, training=True)
    #     h, new_hidden, new_mems = model.call(x, hidden=new_hidden, mems=new_mems, training=True)
    #     # h = model(x, training=True)
    #
    #     loss = tf.reduce_sum(h)

    # # Test gradient tape
    # grad = tape.gradient(loss, model.trainable_variables)

    x = tf.zeros([10, 25], dtype=tf.int32)
    loss, grads = model_forward_with_grads(model, x)

    # Test predict
    model.predict(x)

    model.summary()
