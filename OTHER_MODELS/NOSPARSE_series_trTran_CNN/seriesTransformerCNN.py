import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, BatchNormalization
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import *

def sep(x):
    if x==0 or x ==2 or x ==6:
      return 0
    else:
      return 1
	  
#model structure
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
		
#read data


xtrain = r"/uac/cprj/cprj2716/borrow_from_ESETR_gpmate_tokaho/training_sample_NoSparse.csv.gz"
ytrain = r"/uac/cprj/cprj2716/borrow_from_ESETR_gpmate_tokaho/training_label_NoSparse.csv.gz"
xtest = r"/uac/cprj/cprj2716/borrow_from_ESETR_gpmate_tokaho/testing_sample_NoSparse.csv.gz"
ytest = r"/uac/cprj/cprj2716/borrow_from_ESETR_gpmate_tokaho/testing_label_NoSparse.csv.gz"

"""
xtrain = r"/content/drive/My Drive/estr3108 project/ESTR3108/training_sample_NoSparse.csv.gz"
ytrain = r"/content/drive/My Drive/estr3108 project/ESTR3108/training_label_NoSparse.csv.gz"
xtest = r"/content/drive/My Drive/estr3108 project/ESTR3108/testing_sample_NoSparse.csv.gz"
ytest = r"/content/drive/My Drive/estr3108 project/ESTR3108/testing_label_NoSparse.csv.gz"
"""


samplesdf = pd.read_csv(xtrain,compression ="gzip",delimiter=',')
x_train = samplesdf.to_numpy()

samplesdf = pd.read_csv(ytrain,compression ="gzip",delimiter=',')
y_train = samplesdf.to_numpy()

samplesdf = pd.read_csv(xtest,compression ="gzip",delimiter=',')
x_test = samplesdf.to_numpy()

samplesdf = pd.read_csv(ytest,compression ="gzip",delimiter=',')
y_test = samplesdf.to_numpy()
print("done")
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

#further separate to sepsis(1) and non sepsis(0)
y_train=np.array(list(map(sep, y_train)))
y_test=np.array(list(map(sep, y_test)))

#class weight
count=[0 for i in range(2)]
total=0
for x in y_train:
  x=int(x)
  count[x]=count[x]+1
  total = total + 1

count = list(map(lambda x: (total/x)/2, count))
class_weight={0:count[0],1:count[1]}
print(class_weight)

#model structure
def gen_model(dim):
#hyper parameter
	vocab_size = 30  #after K-fold , seems the size here will not make a significant different on accuracy
	maxlen = 3273  # length of input
	embed_dim = 32+32*dim  # Embedding size for each token
	num_heads = 2   # Number of attention heads
	ff_dim = 128  # Hidden layer size in feed forward network inside transformer


 
	inputs = layers.Input(shape=(maxlen,))
	embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
	x = embedding_layer(inputs)
	transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
	x = transformer_block(x)
	x = layers.BatchNormalization()(x)
	x = layers.GlobalMaxPooling1D()(x)

	x = layers.Reshape((352, 1))(x)

	x = layers.Conv1D(filters=32, kernel_size=2, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = MaxPooling1D(pool_size=2)(x)
	
	x = layers.Conv1D(filters=32, kernel_size=2, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = MaxPooling1D(pool_size=2)(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.1)(x)
	x = layers.Dense(200, activation="relu")(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)

	
	
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
	return model

model=gen_model(10)
history = model.fit(
				x=x_train, 
				y=y_train, 
				class_weight=class_weight,
				verbose=2,
				validation_data=(x_test, y_test),
				batch_size=32,
				shuffle=True,
				epochs=20)
				
testresult=model.predict(x_test)


i = 0
correct = 0
for x in testresult:
    if x >=0.5 and y_test[i] == 1:
        correct = correct + 1
    elif x < 0.5 and y_test[i] == 0:
        correct = correct + 1
    i = i + 1
testacc = correct/i
testacc

fpr, tpr, _ = roc_curve(y_test,testresult)
roc_auc = auc(fpr,tpr)
plt.figure()
lw = 2
plt.plot(fpr,tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC curve for Series traditional Transformer + CNN')
plt.legend(loc="lower right")
plt.savefig("/uac/cprj/cprj2716/borrow_from_ESETR_gpmate_tokaho/TransformerAUROC.png")

plt.figure()
precision, recall, _ = precision_recall_curve(y_test,testresult)
prc_auc = auc(recall,precision)
plt.plot(recall,precision, color='darkorange',
         lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUPRC curve for Series traditional Transformer + CNN')
plt.legend(loc="lower right")
plt.savefig("/uac/cprj/cprj2716/borrow_from_ESETR_gpmate_tokaho/TransformerAUPRC.png")
print("AUPRC = %.02f"% prc_auc)
ss = np.zeros((len(testresult)))
i = 0
for x in testresult:
    if x >= 0.5:
        ss[i] = 1
    else:
        ss[i] = 0
    i = i + 1
f1s = f1_score(y_test,ss)
print("f1_score = %.02f"% f1s)
model.save('/uac/cprj/cprj2716/borrow_from_ESETR_gpmate_tokaho/seriesTrCNN_nosparse')
