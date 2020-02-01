# Single Headed Attention RNN for Tensorflow 2.0
For full details see the paper [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423).

Code ported from author's implementation here - https://github.com/Smerity/sha-rnn

# Usage
The `SHARNN` model class is a direct port in the most part of the codebase written in PyTorch.

In Tensorflow, it can be used either directly as a Keras Model, added as a sublayer of another Model. The model can be traced by tf.function, so performance degredation should be minimum even when custom training loops are being used.

## As a Keras Model
```python
from sharnn import SHARNN

model = SHARNN(num_token=1000, embed_dim=100, num_hid=200, num_layers=2,
               return_hidden=True, return_mem=True)

model.compile(optimizer='adam', loss='mse')

# Test predict
model.predict(x)

model.summary()    
```

## Inside a custom training loop

```python
@tf.function
def model_forward_with_grads(model, x):
    with tf.GradientTape() as tape:
        h, new_hidden, new_mems = model(x, training=True)
        h, new_hidden, new_mems = model(x, hidden=new_hidden, mems=new_mems, training=True)

        loss = tf.reduce_sum(h)  # Just for demonstration purposes

    grad = tape.gradient(loss, model.trainable_variables)

    return loss, grad
```


# Caveats
There is currently an issue with setting a maximum of the number of positions in `mems` (see TODO). Therefore there is currently no limit on the amount of memory that `mems` can take.

# Requirements
 - Tensorflow 2.0+
