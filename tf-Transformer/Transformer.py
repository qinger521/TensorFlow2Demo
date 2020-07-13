import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import tensorflow_datasets as tfds
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
    1、载入数据
    2、预处理 -> dataset
    3、tools
        3.1 generates position embedding
        3.2 create mask
        3.3 scaled_dot_product_attention
    4、builds model
        4.1 MultiheadAttention
        4.2 EncoderLayer
        4.3 DecoderLayer
        4.4 EncoderModel
        4.5 DecoderModel
        4.6 Transformer
    5、 optimizer & loss
    6、 train
    7、 Evaluate and visualize
'''

# 1、载入数据 2、pre
examples, info = tfds.load('ted_hrlr_translate/pt_to_en',
                           with_info=True,
                           as_supervised=True)
train_examples, val_examples = examples["train"],examples['validation']


en_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt ,en in train_examples),
                                                                       target_vocab_size=2 ** 13)
pt_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt ,en in train_examples),
                                                                       target_vocab_size=2 ** 13)

sample_string = "Transformer is awesome."
tokenized_string = en_tokenizer.encode(sample_string)
print("Tokenized string is {}".format(tokenized_string))

origin_string = en_tokenizer.decode(tokenized_string)
print("Origin string is {}".format(origin_string))

for token in tokenized_string:
    print('{} -> {}'.format(token,en_tokenizer.decode([token])))

buffer_size = 20000
batch_size = 64
max_length = 40

def encode_to_subword(pt_sentence,en_sentence):
    pt_sentence = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(pt_sentence.numpy()) + [pt_tokenizer.vocab_size + 1]
    en_sentence = [en_tokenizer.vocab_size] + en_tokenizer.encode(en_sentence.numpy()) + [en_tokenizer.vocab_size + 1]
    return  pt_sentence,en_sentence

def fileter_by_max_length(pt,en):
    return tf.logical_and(tf.size(pt) <= max_length,
                          tf.size(en) <= max_length)

def tf_encode_to_subword(pt_sentence,en_sentence):
    return tf.py_function(encode_to_subword,[pt_sentence,en_sentence],
                          [tf.int64,tf.int64])

train_dataset = train_examples.map(tf_encode_to_subword)
train_dataset = train_dataset.filter(fileter_by_max_length)
train_dataset = train_dataset.shuffle(buffer_size).padded_batch(batch_size,padded_shapes=([-1],[-1]))

val_dataset = val_examples.map(tf_encode_to_subword)
val_dataset = val_dataset.filter(fileter_by_max_length).padded_batch(batch_size,padded_shapes=([-1],[-1]))

for pt,en in val_dataset:
    print(pt.shape,en.shape)

# 3、tool
# PE(pos,2i) = sin(pos/10000^(2i/d_model))
# PE(pos,2i+1) = cos(pos/10000^(2i/d_model))


# sin里面的部分
# pos : [sentence_length,1]
# i.shape : [1,d_model]
# result.shape : [sentence_length,d_model]
def get_angles(pos,i,d_model):
    angle_rates = 1 / np.power(10000,(2*(i//2)) / np.float32(d_model))
    return pos * angle_rates

def get_positional_embedding(sentence_length,d_model):
    angle_rads = get_angles(np.arange(sentence_length)[:,np.newaxis],
                            np.arange(d_model)[np.newaxis,:],
                            d_model)
    sines = np.sin(angle_rads[:,0::2])  # shape = [sentence_length,d_model/2]
    cosines = np.cos(angle_rads[:,1::2]) # shape = [sentence_length,d_model/2]

    # position_embedding : [sentence_length,d_model]
    position_embedding = np.concatenate([sines,cosines],axis=-1)
    position_embedding = position_embedding[np.newaxis,...]
    # position_embedding : [1,sentence_length,d_model]
    return tf.cast(position_embedding,dtype=tf.float32)

def plot_position_embedding(position_embedding):
    plt.pcolormesh(position_embedding[0],cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0,512))
    plt.ylabel('position')
    plt.colorbar()
    plt.show()


position_embedding = get_positional_embedding(50,512)
plot_position_embedding(position_embedding)

# 3： 1.padding mask 2. look ahead mask
# batch_data.shape : [batch_size,seq_len]
def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data,0),tf.float32)
    return padding_mask[:,np.newaxis,np.newaxis,:]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)),-1,0)
    return mask

'''
    scaled_dot_attention
'''
def scaled_dot_product_attention(q,k,v,mask):
    '''
    :param q: shape->[...,seq_len_q,depth]
    :param k: shape->[...,seq_len_k,depth]
    :param v: shape->[...,seq_len_v,depth]
    :param mask:  shape->[...,seq_len_q,seq_len_k]
    :return:
            -output:weight sum
            -attention_weight
    '''

    # shape->[...,seq_len_q,seq_len_k]
    matmul_qk = tf.matmul(q,k,transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits,axis=-1)

    output = tf.matmul(attention_weights,v)

    return output,attention_weights

def print_scaled_dot_product_attention(q,k,v):
    pass

class MultiHeadAttention(tf.keras.layers.Layer):
    '''
        q->wq0->q0
        k->wk0->k0
        v->wv0->v0

        q->Wq->Q->split->q1,q2,q3....
    '''
    def __init__(self,d_model,num_head):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_head
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.WQ = tf.keras.layers.Dense(self.d_model)
        self.Wk = tf.keras.layers.Dense(self.d_model)
        self.Wv = tf.keras.layers.Dense(self.d_model)

        self.dense = tf.keras.layers.Dense(self.d_model)

    def split_heads(self,x,batch_size):
        '''
        :param x: shape->[batch_size,seq_len,d_model]
        want: x->[batch_size,num_heads,seq_len,depth]
        '''
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm=[0,2,1,3])

    def call(self,q,k,v,mask):
        batch_size = tf.shape(q)[0]
        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        q = self.split_heads(q,batch_size)
        k = self.split_heads(k,batch_size)
        v = self.split_heads(v,batch_size)
        # scaled_attention_outputs->shape : [batch_size,num_heads,seq_len_q,depth]
        # attention_weight->shape : [batch_size,num_heads,seq_len_q,seq_len_k]
        scaled_attention_outputs,attention_weight = scaled_dot_product_attention(q,k,v,mask)

        scaled_attention_outputs = tf.transpose(scaled_attention_outputs,
                                                perm = [0,2,1,3])
        concat_attetion = tf.reshape(scaled_attention_outputs,
                                     (batch_size,-1,self.d_model))
        output = self.dense(concat_attetion)

        return output,attention_weight

def feed_forward_network(d_model,dff):
    '''diff : dim of feed forward network'''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff,activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    '''
        x -> self attention -> add & normalize & dropout -> feed forward -> add & normalize & dropout-> ...
    '''
    def __int__(self,d_model,num_heads,dff,rate = 0.1):
        super(EncoderLayer,self).__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ffn = feed_forward_network(d_model,dff)
        self.layer_normal1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layer_normal2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)


    def call(self,x,training,mask):
        '''x :shape [batch_size,seq_len,dim]'''
        '''attn_out :shape [batch_size,seq_len,d_model]'''
        attn_out, _ = self.mha(x,x,x,mask)
        attn_out = self.dropout1(attn_out,training=training)

        out1  =self.layer_normal1(x+attn_out)
        '''ffn_output :shape [batch_size,seq_len,d_model]'''
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output,training=training)
        out2 = self.layer_normal2(out1+ffn_output)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    '''
        x -> self attention -> add & normalize & dropout -> out1
        out1,encoding_outputs -> attention -> add & normalize & dropout -> out2
        out2 -> fnn -> add & normalize & dropout -> out3
    '''
    def __int__(self,d_model,num_heads,dff,rate = 0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_normal1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normal2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normal3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self,x,encoding_outputs,training,look_ahead_mask,padding_mask):
        '''x :shape [batch_size,seq_len,d_model] '''
        '''encoding_output :shape [batch_size,input_seq_len,d_model] '''
        attn1,attn_weights1 = self.mha1(x,x,x,look_ahead_mask)
        attn1 = self.dropout1(attn1,training=training)

        out1 = self.layer_normal1(attn1 + x)

        attn2, attn_weights2= self.mha2(out1,encoding_outputs,encoding_outputs,padding_mask)
        attn2 = self.dropout2(attn2, training=training)

        out2 = self.layer_normal2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output,training=training)

        out3 = self.layer_normal3(ffn_output + out2)

        return out3,attn_weights1,attn_weights2


class EncoderModel(tf.keras.layers.Layer):
    def __int__(self,num_layer,input_vocab_size,max_length,d_model,num_heads,dff,rate=0.1):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layer

        self.embedding = tf.keras.layers.Embedding(input_vocab_size,self.d_model)
        # position_embedding [1,max_length,d_model]
        self.position_embedding = get_positional_embedding(max_length,self.d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

        self.encoding_layers = [
            EncoderLayer(d_model,num_heads,dff,rate) for _ in range(self.num_layers)
        ]

    def call(self,x,training,mask):
        ''' x [batch_size,input_seq_len] '''
        input_seq_len = tf.shape(x)[1]

        ''' x [batch_size,input_seq_len,d_model] '''
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))

        x += self.position_embedding[:,:input_seq_len,:]

        x = self.dropout(x,training=training)

        for i in range(self.num_layers):
            x = self.encoding_layers[i](x,training,mask)
        return x

class DecoderModel(tf.keras.layers.Layer):
    def __int__(self,num_layer,target_vocab_size,max_length,d_model,num_heads,dff,rate=0.1):
        super(DecoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layer
        self.max_length = max_length

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        # position_embedding [1,max_length,d_model]
        self.position_embedding = get_positional_embedding(max_length, self.d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

        self.decoding_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)
        ]


    def call(self,x,encoding_output,training,decoder_mask,encoder_decoder_padding_mask):
        output_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(output_seq_len,max_length)

        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x += self.position_embedding[:, :output_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x,attn1,attn2 = self.decoding_layers[i](x, encoding_output, training, decoder_mask,encoder_decoder_padding_mask)
            attention_weights['decoder_layer{}_att1'.format(i+1)] = attn1
            attention_weights['decoder_layer{}_att2'.format(i+1)] = attn2
        return x , attention_weights

class Transformer(keras.Model):
    def __int__(self,num_layers,input_vocab_size,target_vocab_size,max_length,d_model,num_heads,dff,rate = 0.1):
        super(Transformer,self).__init__()
        self.encoder_model = EncoderModel(num_layers,input_vocab_size,max_length,d_model,num_heads,dff,rate)
        self.decoder_model = DecoderModel(num_layers,target_vocab_size,max_length,d_model,num_heads,dff,rate)
        self.final_layer = keras.layers.Dense(target_vocab_size)

    def call(self,inp,tar,training,encoder_padding_mask,look_ahead_mask,decoder_padding_mask):
        encoding_outputs = self.encoder_model(inp,training,encoder_padding_mask)
        decoding_outputs,attention_weights = self.decoder_model(tar,encoding_outputs,training,look_ahead_mask,decoder_padding_mask)
        prediction = self.final_layer(decoding_outputs)
        return  prediction,attention_weights

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = pt_tokenizer.vocab_size+2
target_vocab_size = en_tokenizer.vocab_size+2

dropout_rate = 0.1

transformer = Transformer(enumerate,input_vocab_size,target_vocab_size,max_length,d_model,num_heads,dff,dropout_rate)

# lrate = (d_model ** -0.5) * min(step_num ** -0.5,step_num * warm_up_steps ** -1.5)

class CustomizedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,d_model,warm_up_steps=40000):
        super(CustomizedSchedule,self).__init__()
        self.d_model = tf.cast(d_model,tf.float32)
        self.warm_up_steps = warm_up_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warm_up_steps ** (-1.5))
        arg3 = tf.math.rsqrt(self.d_model)

        return arg3 * tf.math.minimum(arg1,arg2)

learning_rate = CustomizedSchedule(d_model)
optimizer = keras.optimizers.Adam(learning_rate,beta_1 = 0.9,beta_2 = 0.98,epsilon = 1e-9)

# 可视化学习率
temp_learning_rate_schedule = CustomizedSchedule(d_model)
plt.plot(temp_learning_rate_schedule(tf.range(40000,dtype=tf.float32)))
plt.ylabel('learning rate')
plt.xlabel('training step')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True,reduction='none')
def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    mask = tf.cast(mask,dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def create_masks(inp,tar):
    '''
        Encoder:
            -> encoder padding mask (self attention of EncoderLayer)
        Decoder:
            ->look_ahead_mask, (self attention of DecoderLayer)
            ->encoder_decoder_padding_mask, (encoder-decoder attention of DecoderLayer)
            ->decoder_padding_mask (self attention of DecoderLayer)
    '''
    encoder_padding_mask = create_padding_mask(inp)
    encoder_decoder_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_padding_mask = create_padding_mask(tar)

    decoder_mask = tf.maximum(decoder_padding_mask,look_ahead_mask)
    return encoder_padding_mask,decoder_mask,encoder_decoder_mask

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inp,tar):
    tar_inp = tar[:,:-1]
    tar_real = tar[:,1:]

    encoder_padding_mask,decoder_mask,encoder_decoder_padding_mask = create_masks(inp,tar_inp)
    with tf.GradientTape() as tape:
        predictions,_ = transformer(inp,tar_inp,True,encoder_padding_mask,
                                    decoder_mask,
                                    encoder_decoder_padding_mask)
        loss = loss_function(tar_real,predictions)
    gradients = tape.gradient(loss,transformer.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients,transformer.trainable_variables)
    )
    train_loss(loss)
    train_accuracy(tar_real,predictions)

epochs = 20
print("------------.------------.------------START------------.------------.------------")
for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch,(inp,tar)) in enumerate(train_dataset):
        train_step(inp,tar)
        if batch % 100 ==0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch+1,batch,train_loss.result(),train_accuracy.result()))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch+1,train_loss.result(),train_accuracy.result()))


def evaluate(inp_sentence):
    input_id_sentence = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(inp_sentence) + [pt_tokenizer.vocab_size+1]
    encoder_input = tf.expand_dims(input_id_sentence,0) # shape: (1,input_sentence_length)

    decoder_input = tf.expand_dims([en_tokenizer.vocab_size],0) #shape: (1,1)

    for i in range(max_length):
        encoder_padding_mask,decoder_mask,encoder_decoder_padding_mask = create_masks(encoder_input,decoder_input)
        #predictions shape:(batch_size,output_target_len,target_vocab_size)
        predictions,attention_weights = transformer(encoder_input,decoder_input,False,encoder_padding_mask,decoder_mask,encoder_decoder_padding_mask)
        # predictions shape [batch_size,target_vocab_size]
        predictions = predictions[:,-1,:]

        predicted_id = tf.cast(tf.argmax(predictions,axis=-1),tf.int32)

        if tf.equal(predicted_id,en_tokenizer.vocab_size + 1):
            return tf.squeeze(decoder_input,axis=0),attention_weights
        decoder_input = tf.concat([decoder_input,predicted_id],axis=-1)
    return tf.squeeze(decoder_input,axis=0),attention_weights
