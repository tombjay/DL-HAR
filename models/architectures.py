import gin
import tensorflow as tf
from keras import layers, Model

@gin.configurable
def build_LSTM_model(input_shape,num_classes,LSTM_units,Dense_units,s2s,dropout_rate):
    inputs = layers.Input(input_shape)
    x =layers.Bidirectional(layers.LSTM(units=LSTM_units,input_shape=input_shape, return_sequences =  s2s))(inputs)
    x = layers.Dropout(rate=dropout_rate) (x)
    x = layers.Dense(units=Dense_units, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs, name = "Single_LSTM_model")

@gin.configurable
def build_GRU_model(input_shape,num_classes,GRU_units,Dense_units,s2s,dropout_rate):
    
    inputs = layers.Input(input_shape)
    x =layers.Bidirectional(layers.GRU(units=GRU_units,input_shape=input_shape, return_sequences =  s2s))(inputs)
    x = layers.Dropout(rate=dropout_rate) (x)
    x = layers.Dense(units=Dense_units, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs,name = "Single_GRU_model")

@gin.configurable
def build_conv_LSTM_model(input_shape,num_classes,fltrs,kernel_size,LSTM_units,Dense_units,s2s,dropout_rate):
    inputs = layers.Input(input_shape)
    x = layers.Conv1D(filters = fltrs, kernel_size=kernel_size, padding="same")(inputs)
    x =layers.Bidirectional(layers.LSTM(units=LSTM_units,input_shape=input_shape, return_sequences =  s2s))(x)
    x = layers.Dropout(rate=dropout_rate) (x)
    x = layers.Dense(units=Dense_units, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs,name = "Conv_LSTM_model")

@gin.configurable
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    return x + res

@gin.configurable
def build_transformer_model(input_shape,num_classes,head_size,num_heads,ff_dim,num_transformer_blocks,mlp_units,dropout,mlp_dropout):
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs,name = "Transformer")