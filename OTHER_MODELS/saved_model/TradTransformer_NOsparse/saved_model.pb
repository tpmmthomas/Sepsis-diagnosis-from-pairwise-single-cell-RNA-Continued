Թ0
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d388��,
�
batch_normalization/gammaVarHandleOp**
shared_namebatch_normalization/gamma*
dtype0*
_output_shapes
: *
shape:�
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
dtype0*
_output_shapes	
:�
�
batch_normalization/betaVarHandleOp*)
shared_namebatch_normalization/beta*
dtype0*
_output_shapes
: *
shape:�
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
dtype0*
_output_shapes	
:�
�
batch_normalization/moving_meanVarHandleOp*0
shared_name!batch_normalization/moving_mean*
dtype0*
_output_shapes
: *
shape:�
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes	
:�
�
#batch_normalization/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes	
:�
z
dense_6/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
dtype0* 
_output_shapes
:
��
q
dense_6/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
dtype0*
_output_shapes	
:�
y
dense_7/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	�*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
dtype0*
_output_shapes
:	�
p
dense_7/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
�
1token_and_position_embedding/embedding/embeddingsVarHandleOp*
dtype0*
_output_shapes
: *
shape:	�*B
shared_name31token_and_position_embedding/embedding/embeddings
�
Etoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp1token_and_position_embedding/embedding/embeddings*
dtype0*
_output_shapes
:	�
�
3token_and_position_embedding/embedding_1/embeddingsVarHandleOp*
shape:
��*D
shared_name53token_and_position_embedding/embedding_1/embeddings*
dtype0*
_output_shapes
: 
�
Gtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp3token_and_position_embedding/embedding_1/embeddings*
dtype0* 
_output_shapes
:
��
�
8transformer_block/multi_head_self_attention/dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*I
shared_name:8transformer_block/multi_head_self_attention/dense/kernel
�
Ltransformer_block/multi_head_self_attention/dense/kernel/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/dense/kernel*
dtype0* 
_output_shapes
:
��
�
6transformer_block/multi_head_self_attention/dense/biasVarHandleOp*
shape:�*G
shared_name86transformer_block/multi_head_self_attention/dense/bias*
dtype0*
_output_shapes
: 
�
Jtransformer_block/multi_head_self_attention/dense/bias/Read/ReadVariableOpReadVariableOp6transformer_block/multi_head_self_attention/dense/bias*
dtype0*
_output_shapes	
:�
�
:transformer_block/multi_head_self_attention/dense_1/kernelVarHandleOp*K
shared_name<:transformer_block/multi_head_self_attention/dense_1/kernel*
dtype0*
_output_shapes
: *
shape:
��
�
Ntransformer_block/multi_head_self_attention/dense_1/kernel/Read/ReadVariableOpReadVariableOp:transformer_block/multi_head_self_attention/dense_1/kernel*
dtype0* 
_output_shapes
:
��
�
8transformer_block/multi_head_self_attention/dense_1/biasVarHandleOp*
shape:�*I
shared_name:8transformer_block/multi_head_self_attention/dense_1/bias*
dtype0*
_output_shapes
: 
�
Ltransformer_block/multi_head_self_attention/dense_1/bias/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/dense_1/bias*
dtype0*
_output_shapes	
:�
�
:transformer_block/multi_head_self_attention/dense_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*K
shared_name<:transformer_block/multi_head_self_attention/dense_2/kernel
�
Ntransformer_block/multi_head_self_attention/dense_2/kernel/Read/ReadVariableOpReadVariableOp:transformer_block/multi_head_self_attention/dense_2/kernel*
dtype0* 
_output_shapes
:
��
�
8transformer_block/multi_head_self_attention/dense_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*I
shared_name:8transformer_block/multi_head_self_attention/dense_2/bias
�
Ltransformer_block/multi_head_self_attention/dense_2/bias/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/dense_2/bias*
dtype0*
_output_shapes	
:�
�
:transformer_block/multi_head_self_attention/dense_3/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*K
shared_name<:transformer_block/multi_head_self_attention/dense_3/kernel
�
Ntransformer_block/multi_head_self_attention/dense_3/kernel/Read/ReadVariableOpReadVariableOp:transformer_block/multi_head_self_attention/dense_3/kernel*
dtype0* 
_output_shapes
:
��
�
8transformer_block/multi_head_self_attention/dense_3/biasVarHandleOp*I
shared_name:8transformer_block/multi_head_self_attention/dense_3/bias*
dtype0*
_output_shapes
: *
shape:�
�
Ltransformer_block/multi_head_self_attention/dense_3/bias/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/dense_3/bias*
dtype0*
_output_shapes	
:�
�
+transformer_block/sequential/dense_4/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*<
shared_name-+transformer_block/sequential/dense_4/kernel
�
?transformer_block/sequential/dense_4/kernel/Read/ReadVariableOpReadVariableOp+transformer_block/sequential/dense_4/kernel*
dtype0* 
_output_shapes
:
��
�
)transformer_block/sequential/dense_4/biasVarHandleOp*:
shared_name+)transformer_block/sequential/dense_4/bias*
dtype0*
_output_shapes
: *
shape:�
�
=transformer_block/sequential/dense_4/bias/Read/ReadVariableOpReadVariableOp)transformer_block/sequential/dense_4/bias*
dtype0*
_output_shapes	
:�
�
+transformer_block/sequential/dense_5/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*<
shared_name-+transformer_block/sequential/dense_5/kernel
�
?transformer_block/sequential/dense_5/kernel/Read/ReadVariableOpReadVariableOp+transformer_block/sequential/dense_5/kernel*
dtype0* 
_output_shapes
:
��
�
)transformer_block/sequential/dense_5/biasVarHandleOp*:
shared_name+)transformer_block/sequential/dense_5/bias*
dtype0*
_output_shapes
: *
shape:�
�
=transformer_block/sequential/dense_5/bias/Read/ReadVariableOpReadVariableOp)transformer_block/sequential/dense_5/bias*
dtype0*
_output_shapes	
:�
�
+transformer_block/layer_normalization/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*<
shared_name-+transformer_block/layer_normalization/gamma
�
?transformer_block/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp+transformer_block/layer_normalization/gamma*
dtype0*
_output_shapes	
:�
�
*transformer_block/layer_normalization/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*;
shared_name,*transformer_block/layer_normalization/beta
�
>transformer_block/layer_normalization/beta/Read/ReadVariableOpReadVariableOp*transformer_block/layer_normalization/beta*
dtype0*
_output_shapes	
:�
�
-transformer_block/layer_normalization_1/gammaVarHandleOp*
shape:�*>
shared_name/-transformer_block/layer_normalization_1/gamma*
dtype0*
_output_shapes
: 
�
Atransformer_block/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp-transformer_block/layer_normalization_1/gamma*
dtype0*
_output_shapes	
:�
�
,transformer_block/layer_normalization_1/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*=
shared_name.,transformer_block/layer_normalization_1/beta
�
@transformer_block/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp,transformer_block/layer_normalization_1/beta*
dtype0*
_output_shapes	
:�
^
totalVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
�
 Adam/batch_normalization/gamma/mVarHandleOp*
shape:�*1
shared_name" Adam/batch_normalization/gamma/m*
dtype0*
_output_shapes
: 
�
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
dtype0*
_output_shapes	
:�
�
Adam/batch_normalization/beta/mVarHandleOp*
shape:�*0
shared_name!Adam/batch_normalization/beta/m*
dtype0*
_output_shapes
: 
�
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
dtype0*
_output_shapes	
:�
�
Adam/dense_6/kernel/mVarHandleOp*&
shared_nameAdam/dense_6/kernel/m*
dtype0*
_output_shapes
: *
shape:
��
�
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
dtype0* 
_output_shapes
:
��

Adam/dense_6/bias/mVarHandleOp*
shape:�*$
shared_nameAdam/dense_6/bias/m*
dtype0*
_output_shapes
: 
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
dtype0*
_output_shapes	
:�
�
Adam/dense_7/kernel/mVarHandleOp*&
shared_nameAdam/dense_7/kernel/m*
dtype0*
_output_shapes
: *
shape:	�
�
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
dtype0*
_output_shapes
:	�
~
Adam/dense_7/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
dtype0*
_output_shapes
:
�
8Adam/token_and_position_embedding/embedding/embeddings/mVarHandleOp*I
shared_name:8Adam/token_and_position_embedding/embedding/embeddings/m*
dtype0*
_output_shapes
: *
shape:	�
�
LAdam/token_and_position_embedding/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp8Adam/token_and_position_embedding/embedding/embeddings/m*
dtype0*
_output_shapes
:	�
�
:Adam/token_and_position_embedding/embedding_1/embeddings/mVarHandleOp*
shape:
��*K
shared_name<:Adam/token_and_position_embedding/embedding_1/embeddings/m*
dtype0*
_output_shapes
: 
�
NAdam/token_and_position_embedding/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOp:Adam/token_and_position_embedding/embedding_1/embeddings/m*
dtype0* 
_output_shapes
:
��
�
?Adam/transformer_block/multi_head_self_attention/dense/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense/kernel/m
�
SAdam/transformer_block/multi_head_self_attention/dense/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense/kernel/m*
dtype0* 
_output_shapes
:
��
�
=Adam/transformer_block/multi_head_self_attention/dense/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*N
shared_name?=Adam/transformer_block/multi_head_self_attention/dense/bias/m
�
QAdam/transformer_block/multi_head_self_attention/dense/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_block/multi_head_self_attention/dense/bias/m*
dtype0*
_output_shapes	
:�
�
AAdam/transformer_block/multi_head_self_attention/dense_1/kernel/mVarHandleOp*
shape:
��*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m*
dtype0*
_output_shapes
: 
�
UAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m*
dtype0* 
_output_shapes
:
��
�
?Adam/transformer_block/multi_head_self_attention/dense_1/bias/mVarHandleOp*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_1/bias/m*
dtype0*
_output_shapes
: *
shape:�
�
SAdam/transformer_block/multi_head_self_attention/dense_1/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_1/bias/m*
dtype0*
_output_shapes	
:�
�
AAdam/transformer_block/multi_head_self_attention/dense_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m
�
UAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m*
dtype0* 
_output_shapes
:
��
�
?Adam/transformer_block/multi_head_self_attention/dense_2/bias/mVarHandleOp*
shape:�*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_2/bias/m*
dtype0*
_output_shapes
: 
�
SAdam/transformer_block/multi_head_self_attention/dense_2/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_2/bias/m*
dtype0*
_output_shapes	
:�
�
AAdam/transformer_block/multi_head_self_attention/dense_3/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m
�
UAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m*
dtype0* 
_output_shapes
:
��
�
?Adam/transformer_block/multi_head_self_attention/dense_3/bias/mVarHandleOp*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_3/bias/m*
dtype0*
_output_shapes
: *
shape:�
�
SAdam/transformer_block/multi_head_self_attention/dense_3/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_3/bias/m*
dtype0*
_output_shapes	
:�
�
2Adam/transformer_block/sequential/dense_4/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*C
shared_name42Adam/transformer_block/sequential/dense_4/kernel/m
�
FAdam/transformer_block/sequential/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/sequential/dense_4/kernel/m*
dtype0* 
_output_shapes
:
��
�
0Adam/transformer_block/sequential/dense_4/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*A
shared_name20Adam/transformer_block/sequential/dense_4/bias/m
�
DAdam/transformer_block/sequential/dense_4/bias/m/Read/ReadVariableOpReadVariableOp0Adam/transformer_block/sequential/dense_4/bias/m*
dtype0*
_output_shapes	
:�
�
2Adam/transformer_block/sequential/dense_5/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*C
shared_name42Adam/transformer_block/sequential/dense_5/kernel/m
�
FAdam/transformer_block/sequential/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/sequential/dense_5/kernel/m*
dtype0* 
_output_shapes
:
��
�
0Adam/transformer_block/sequential/dense_5/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*A
shared_name20Adam/transformer_block/sequential/dense_5/bias/m
�
DAdam/transformer_block/sequential/dense_5/bias/m/Read/ReadVariableOpReadVariableOp0Adam/transformer_block/sequential/dense_5/bias/m*
dtype0*
_output_shapes	
:�
�
2Adam/transformer_block/layer_normalization/gamma/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*C
shared_name42Adam/transformer_block/layer_normalization/gamma/m
�
FAdam/transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/layer_normalization/gamma/m*
dtype0*
_output_shapes	
:�
�
1Adam/transformer_block/layer_normalization/beta/mVarHandleOp*
shape:�*B
shared_name31Adam/transformer_block/layer_normalization/beta/m*
dtype0*
_output_shapes
: 
�
EAdam/transformer_block/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp1Adam/transformer_block/layer_normalization/beta/m*
dtype0*
_output_shapes	
:�
�
4Adam/transformer_block/layer_normalization_1/gamma/mVarHandleOp*E
shared_name64Adam/transformer_block/layer_normalization_1/gamma/m*
dtype0*
_output_shapes
: *
shape:�
�
HAdam/transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/layer_normalization_1/gamma/m*
dtype0*
_output_shapes	
:�
�
3Adam/transformer_block/layer_normalization_1/beta/mVarHandleOp*
shape:�*D
shared_name53Adam/transformer_block/layer_normalization_1/beta/m*
dtype0*
_output_shapes
: 
�
GAdam/transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp3Adam/transformer_block/layer_normalization_1/beta/m*
dtype0*
_output_shapes	
:�
�
 Adam/batch_normalization/gamma/vVarHandleOp*
shape:�*1
shared_name" Adam/batch_normalization/gamma/v*
dtype0*
_output_shapes
: 
�
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
dtype0*
_output_shapes	
:�
�
Adam/batch_normalization/beta/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*0
shared_name!Adam/batch_normalization/beta/v
�
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
dtype0*
_output_shapes	
:�
�
Adam/dense_6/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*&
shared_nameAdam/dense_6/kernel/v
�
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
dtype0* 
_output_shapes
:
��

Adam/dense_6/bias/vVarHandleOp*$
shared_nameAdam/dense_6/bias/v*
dtype0*
_output_shapes
: *
shape:�
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
dtype0*
_output_shapes	
:�
�
Adam/dense_7/kernel/vVarHandleOp*&
shared_nameAdam/dense_7/kernel/v*
dtype0*
_output_shapes
: *
shape:	�
�
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
dtype0*
_output_shapes
:	�
~
Adam/dense_7/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
dtype0*
_output_shapes
:
�
8Adam/token_and_position_embedding/embedding/embeddings/vVarHandleOp*I
shared_name:8Adam/token_and_position_embedding/embedding/embeddings/v*
dtype0*
_output_shapes
: *
shape:	�
�
LAdam/token_and_position_embedding/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp8Adam/token_and_position_embedding/embedding/embeddings/v*
dtype0*
_output_shapes
:	�
�
:Adam/token_and_position_embedding/embedding_1/embeddings/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*K
shared_name<:Adam/token_and_position_embedding/embedding_1/embeddings/v
�
NAdam/token_and_position_embedding/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOp:Adam/token_and_position_embedding/embedding_1/embeddings/v*
dtype0* 
_output_shapes
:
��
�
?Adam/transformer_block/multi_head_self_attention/dense/kernel/vVarHandleOp*
shape:
��*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense/kernel/v*
dtype0*
_output_shapes
: 
�
SAdam/transformer_block/multi_head_self_attention/dense/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense/kernel/v*
dtype0* 
_output_shapes
:
��
�
=Adam/transformer_block/multi_head_self_attention/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*N
shared_name?=Adam/transformer_block/multi_head_self_attention/dense/bias/v
�
QAdam/transformer_block/multi_head_self_attention/dense/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_block/multi_head_self_attention/dense/bias/v*
dtype0*
_output_shapes	
:�
�
AAdam/transformer_block/multi_head_self_attention/dense_1/kernel/vVarHandleOp*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v*
dtype0*
_output_shapes
: *
shape:
��
�
UAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v*
dtype0* 
_output_shapes
:
��
�
?Adam/transformer_block/multi_head_self_attention/dense_1/bias/vVarHandleOp*
shape:�*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_1/bias/v*
dtype0*
_output_shapes
: 
�
SAdam/transformer_block/multi_head_self_attention/dense_1/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_1/bias/v*
dtype0*
_output_shapes	
:�
�
AAdam/transformer_block/multi_head_self_attention/dense_2/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v
�
UAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v*
dtype0* 
_output_shapes
:
��
�
?Adam/transformer_block/multi_head_self_attention/dense_2/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_2/bias/v
�
SAdam/transformer_block/multi_head_self_attention/dense_2/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_2/bias/v*
dtype0*
_output_shapes	
:�
�
AAdam/transformer_block/multi_head_self_attention/dense_3/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v
�
UAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v*
dtype0* 
_output_shapes
:
��
�
?Adam/transformer_block/multi_head_self_attention/dense_3/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_3/bias/v
�
SAdam/transformer_block/multi_head_self_attention/dense_3/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_3/bias/v*
dtype0*
_output_shapes	
:�
�
2Adam/transformer_block/sequential/dense_4/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*C
shared_name42Adam/transformer_block/sequential/dense_4/kernel/v
�
FAdam/transformer_block/sequential/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/sequential/dense_4/kernel/v*
dtype0* 
_output_shapes
:
��
�
0Adam/transformer_block/sequential/dense_4/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*A
shared_name20Adam/transformer_block/sequential/dense_4/bias/v
�
DAdam/transformer_block/sequential/dense_4/bias/v/Read/ReadVariableOpReadVariableOp0Adam/transformer_block/sequential/dense_4/bias/v*
dtype0*
_output_shapes	
:�
�
2Adam/transformer_block/sequential/dense_5/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
��*C
shared_name42Adam/transformer_block/sequential/dense_5/kernel/v
�
FAdam/transformer_block/sequential/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/sequential/dense_5/kernel/v*
dtype0* 
_output_shapes
:
��
�
0Adam/transformer_block/sequential/dense_5/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*A
shared_name20Adam/transformer_block/sequential/dense_5/bias/v
�
DAdam/transformer_block/sequential/dense_5/bias/v/Read/ReadVariableOpReadVariableOp0Adam/transformer_block/sequential/dense_5/bias/v*
dtype0*
_output_shapes	
:�
�
2Adam/transformer_block/layer_normalization/gamma/vVarHandleOp*
shape:�*C
shared_name42Adam/transformer_block/layer_normalization/gamma/v*
dtype0*
_output_shapes
: 
�
FAdam/transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/layer_normalization/gamma/v*
dtype0*
_output_shapes	
:�
�
1Adam/transformer_block/layer_normalization/beta/vVarHandleOp*
shape:�*B
shared_name31Adam/transformer_block/layer_normalization/beta/v*
dtype0*
_output_shapes
: 
�
EAdam/transformer_block/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp1Adam/transformer_block/layer_normalization/beta/v*
dtype0*
_output_shapes	
:�
�
4Adam/transformer_block/layer_normalization_1/gamma/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*E
shared_name64Adam/transformer_block/layer_normalization_1/gamma/v
�
HAdam/transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/layer_normalization_1/gamma/v*
dtype0*
_output_shapes	
:�
�
3Adam/transformer_block/layer_normalization_1/beta/vVarHandleOp*D
shared_name53Adam/transformer_block/layer_normalization_1/beta/v*
dtype0*
_output_shapes
: *
shape:�
�
GAdam/transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp3Adam/transformer_block/layer_normalization_1/beta/v*
dtype0*
_output_shapes	
:�

NoOpNoOp
��
ConstConst"/device:CPU:0*֛
value˛BǛ B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
R
trainable_variables
regularization_losses
	variables
	keras_api
n
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
�
att
ffn

layernorm1

layernorm2
dropout1
dropout2
trainable_variables
 regularization_losses
!	variables
"	keras_api
�
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
h

4kernel
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
h

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
�
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate$m�%m�4m�5m�:m�;m�Em�Fm�Gm�Hm�Im�Jm�Km�Lm�Mm�Nm�Om�Pm�Qm�Rm�Sm�Tm�Um�Vm�$v�%v�4v�5v�:v�;v�Ev�Fv�Gv�Hv�Iv�Jv�Kv�Lv�Mv�Nv�Ov�Pv�Qv�Rv�Sv�Tv�Uv�Vv�
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
$18
%19
420
521
:22
;23
 
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
$18
%19
&20
'21
422
523
:24
;25
�
Wlayer_regularization_losses

trainable_variables
regularization_losses
Xmetrics

Ylayers
	variables
Znon_trainable_variables
 
 
 
 
�
[layer_regularization_losses
trainable_variables
\metrics

]layers
regularization_losses
	variables
^non_trainable_variables
b
E
embeddings
_trainable_variables
`regularization_losses
a	variables
b	keras_api
b
F
embeddings
ctrainable_variables
dregularization_losses
e	variables
f	keras_api

E0
F1
 

E0
F1
�
glayer_regularization_losses
trainable_variables
hmetrics

ilayers
regularization_losses
	variables
jnon_trainable_variables
�
kquery_dense
l	key_dense
mvalue_dense
ncombine_heads
otrainable_variables
pregularization_losses
q	variables
r	keras_api
l
slayer-0
tlayer-1
utrainable_variables
vregularization_losses
w	variables
x	keras_api
q
yaxis
	Sgamma
Tbeta
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
t
~axis
	Ugamma
Vbeta
trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
v
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15
 
v
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15
�
 �layer_regularization_losses
trainable_variables
�metrics
�layers
 regularization_losses
!	variables
�non_trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
&2
'3
�
 �layer_regularization_losses
(trainable_variables
�metrics
�layers
)regularization_losses
*	variables
�non_trainable_variables
 
 
 
�
 �layer_regularization_losses
,trainable_variables
�metrics
�layers
-regularization_losses
.	variables
�non_trainable_variables
 
 
 
�
 �layer_regularization_losses
0trainable_variables
�metrics
�layers
1regularization_losses
2	variables
�non_trainable_variables
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
�
 �layer_regularization_losses
6trainable_variables
�metrics
�layers
7regularization_losses
8	variables
�non_trainable_variables
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
�
 �layer_regularization_losses
<trainable_variables
�metrics
�layers
=regularization_losses
>	variables
�non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1token_and_position_embedding/embedding/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE3token_and_position_embedding/embedding_1/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8transformer_block/multi_head_self_attention/dense/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE6transformer_block/multi_head_self_attention/dense/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE:transformer_block/multi_head_self_attention/dense_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8transformer_block/multi_head_self_attention/dense_1/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE:transformer_block/multi_head_self_attention/dense_2/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8transformer_block/multi_head_self_attention/dense_2/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE:transformer_block/multi_head_self_attention/dense_3/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8transformer_block/multi_head_self_attention/dense_3/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+transformer_block/sequential/dense_4/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE)transformer_block/sequential/dense_4/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+transformer_block/sequential/dense_5/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE)transformer_block/sequential/dense_5/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+transformer_block/layer_normalization/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*transformer_block/layer_normalization/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-transformer_block/layer_normalization_1/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,transformer_block/layer_normalization_1/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
 

�0
8
0
1
2
3
4
5
6
7

&0
'1
 
 
 
 

E0
 

E0
�
 �layer_regularization_losses
_trainable_variables
�metrics
�layers
`regularization_losses
a	variables
�non_trainable_variables

F0
 

F0
�
 �layer_regularization_losses
ctrainable_variables
�metrics
�layers
dregularization_losses
e	variables
�non_trainable_variables
 
 

0
1
 
l

Gkernel
Hbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
l

Ikernel
Jbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
l

Kkernel
Lbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
l

Mkernel
Nbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
8
G0
H1
I2
J3
K4
L5
M6
N7
 
8
G0
H1
I2
J3
K4
L5
M6
N7
�
 �layer_regularization_losses
otrainable_variables
�metrics
�layers
pregularization_losses
q	variables
�non_trainable_variables
l

Okernel
Pbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
l

Qkernel
Rbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api

O0
P1
Q2
R3
 

O0
P1
Q2
R3
�
 �layer_regularization_losses
utrainable_variables
vregularization_losses
�metrics
�layers
w	variables
�non_trainable_variables
 

S0
T1
 

S0
T1
�
 �layer_regularization_losses
ztrainable_variables
�metrics
�layers
{regularization_losses
|	variables
�non_trainable_variables
 

U0
V1
 

U0
V1
�
 �layer_regularization_losses
trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
 
 
 
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
 
 
 
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
 
 
*
0
1
2
3
4
5
 
 
 
 

&0
'1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


�total

�count
�
_fn_kwargs
�trainable_variables
�regularization_losses
�	variables
�	keras_api
 
 
 
 
 
 
 
 

G0
H1
 

G0
H1
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables

I0
J1
 

I0
J1
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables

K0
L1
 

K0
L1
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables

M0
N1
 

M0
N1
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
 
 

k0
l1
m2
n3
 

O0
P1
 

O0
P1
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables

Q0
R1
 

Q0
R1
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
 
 

s0
t1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

�0
�1
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1
��
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE8Adam/token_and_position_embedding/embedding/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/token_and_position_embedding/embedding_1/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE=Adam/transformer_block/multi_head_self_attention/dense/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_1/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_2/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_3/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2Adam/transformer_block/sequential/dense_4/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/transformer_block/sequential/dense_4/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2Adam/transformer_block/sequential/dense_5/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/transformer_block/sequential/dense_5/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2Adam/transformer_block/layer_normalization/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE1Adam/transformer_block/layer_normalization/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4Adam/transformer_block/layer_normalization_1/gamma/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE3Adam/transformer_block/layer_normalization_1/beta/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE8Adam/token_and_position_embedding/embedding/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE:Adam/token_and_position_embedding/embedding_1/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE=Adam/transformer_block/multi_head_self_attention/dense/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_1/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_2/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_3/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2Adam/transformer_block/sequential/dense_4/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/transformer_block/sequential/dense_4/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2Adam/transformer_block/sequential/dense_5/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/transformer_block/sequential/dense_5/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2Adam/transformer_block/layer_normalization/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE1Adam/transformer_block/layer_normalization/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4Adam/transformer_block/layer_normalization_1/gamma/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE3Adam/transformer_block/layer_normalization_1/beta/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
|
serving_default_input_1Placeholder*
dtype0*(
_output_shapes
:����������*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13token_and_position_embedding/embedding_1/embeddings1token_and_position_embedding/embedding/embeddings8transformer_block/multi_head_self_attention/dense/kernel6transformer_block/multi_head_self_attention/dense/bias:transformer_block/multi_head_self_attention/dense_1/kernel8transformer_block/multi_head_self_attention/dense_1/bias:transformer_block/multi_head_self_attention/dense_2/kernel8transformer_block/multi_head_self_attention/dense_2/bias:transformer_block/multi_head_self_attention/dense_3/kernel8transformer_block/multi_head_self_attention/dense_3/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta+transformer_block/sequential/dense_4/kernel)transformer_block/sequential/dense_4/bias+transformer_block/sequential/dense_5/kernel)transformer_block/sequential/dense_5/bias-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/beta#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*,
_gradient_op_typePartitionedCall-70211*,
f'R%
#__inference_signature_wrapper_68121*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������*&
Tin
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�+
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpEtoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpGtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/dense/kernel/Read/ReadVariableOpJtransformer_block/multi_head_self_attention/dense/bias/Read/ReadVariableOpNtransformer_block/multi_head_self_attention/dense_1/kernel/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_1/bias/Read/ReadVariableOpNtransformer_block/multi_head_self_attention/dense_2/kernel/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_2/bias/Read/ReadVariableOpNtransformer_block/multi_head_self_attention/dense_3/kernel/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_3/bias/Read/ReadVariableOp?transformer_block/sequential/dense_4/kernel/Read/ReadVariableOp=transformer_block/sequential/dense_4/bias/Read/ReadVariableOp?transformer_block/sequential/dense_5/kernel/Read/ReadVariableOp=transformer_block/sequential/dense_5/bias/Read/ReadVariableOp?transformer_block/layer_normalization/gamma/Read/ReadVariableOp>transformer_block/layer_normalization/beta/Read/ReadVariableOpAtransformer_block/layer_normalization_1/gamma/Read/ReadVariableOp@transformer_block/layer_normalization_1/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOpLAdam/token_and_position_embedding/embedding/embeddings/m/Read/ReadVariableOpNAdam/token_and_position_embedding/embedding_1/embeddings/m/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense/kernel/m/Read/ReadVariableOpQAdam/transformer_block/multi_head_self_attention/dense/bias/m/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_1/bias/m/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_2/bias/m/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_3/bias/m/Read/ReadVariableOpFAdam/transformer_block/sequential/dense_4/kernel/m/Read/ReadVariableOpDAdam/transformer_block/sequential/dense_4/bias/m/Read/ReadVariableOpFAdam/transformer_block/sequential/dense_5/kernel/m/Read/ReadVariableOpDAdam/transformer_block/sequential/dense_5/bias/m/Read/ReadVariableOpFAdam/transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpEAdam/transformer_block/layer_normalization/beta/m/Read/ReadVariableOpHAdam/transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpGAdam/transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpLAdam/token_and_position_embedding/embedding/embeddings/v/Read/ReadVariableOpNAdam/token_and_position_embedding/embedding_1/embeddings/v/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense/kernel/v/Read/ReadVariableOpQAdam/transformer_block/multi_head_self_attention/dense/bias/v/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_1/bias/v/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_2/bias/v/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_3/bias/v/Read/ReadVariableOpFAdam/transformer_block/sequential/dense_4/kernel/v/Read/ReadVariableOpDAdam/transformer_block/sequential/dense_4/bias/v/Read/ReadVariableOpFAdam/transformer_block/sequential/dense_5/kernel/v/Read/ReadVariableOpDAdam/transformer_block/sequential/dense_5/bias/v/Read/ReadVariableOpFAdam/transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpEAdam/transformer_block/layer_normalization/beta/v/Read/ReadVariableOpHAdam/transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpGAdam/transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpConst*,
_gradient_op_typePartitionedCall-70314*'
f"R 
__inference__traced_save_70313*
Tout
2*-
config_proto

GPU

CPU2*0J 8*^
TinW
U2S	*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate1token_and_position_embedding/embedding/embeddings3token_and_position_embedding/embedding_1/embeddings8transformer_block/multi_head_self_attention/dense/kernel6transformer_block/multi_head_self_attention/dense/bias:transformer_block/multi_head_self_attention/dense_1/kernel8transformer_block/multi_head_self_attention/dense_1/bias:transformer_block/multi_head_self_attention/dense_2/kernel8transformer_block/multi_head_self_attention/dense_2/bias:transformer_block/multi_head_self_attention/dense_3/kernel8transformer_block/multi_head_self_attention/dense_3/bias+transformer_block/sequential/dense_4/kernel)transformer_block/sequential/dense_4/bias+transformer_block/sequential/dense_5/kernel)transformer_block/sequential/dense_5/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betatotalcount Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/m8Adam/token_and_position_embedding/embedding/embeddings/m:Adam/token_and_position_embedding/embedding_1/embeddings/m?Adam/transformer_block/multi_head_self_attention/dense/kernel/m=Adam/transformer_block/multi_head_self_attention/dense/bias/mAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m?Adam/transformer_block/multi_head_self_attention/dense_1/bias/mAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m?Adam/transformer_block/multi_head_self_attention/dense_2/bias/mAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m?Adam/transformer_block/multi_head_self_attention/dense_3/bias/m2Adam/transformer_block/sequential/dense_4/kernel/m0Adam/transformer_block/sequential/dense_4/bias/m2Adam/transformer_block/sequential/dense_5/kernel/m0Adam/transformer_block/sequential/dense_5/bias/m2Adam/transformer_block/layer_normalization/gamma/m1Adam/transformer_block/layer_normalization/beta/m4Adam/transformer_block/layer_normalization_1/gamma/m3Adam/transformer_block/layer_normalization_1/beta/m Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v8Adam/token_and_position_embedding/embedding/embeddings/v:Adam/token_and_position_embedding/embedding_1/embeddings/v?Adam/transformer_block/multi_head_self_attention/dense/kernel/v=Adam/transformer_block/multi_head_self_attention/dense/bias/vAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v?Adam/transformer_block/multi_head_self_attention/dense_1/bias/vAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v?Adam/transformer_block/multi_head_self_attention/dense_2/bias/vAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v?Adam/transformer_block/multi_head_self_attention/dense_3/bias/v2Adam/transformer_block/sequential/dense_4/kernel/v0Adam/transformer_block/sequential/dense_4/bias/v2Adam/transformer_block/sequential/dense_5/kernel/v0Adam/transformer_block/sequential/dense_5/bias/v2Adam/transformer_block/layer_normalization/gamma/v1Adam/transformer_block/layer_normalization/beta/v4Adam/transformer_block/layer_normalization_1/gamma/v3Adam/transformer_block/layer_normalization_1/beta/v*-
config_proto

GPU

CPU2*0J 8*]
TinV
T2R*
_output_shapes
: *,
_gradient_op_typePartitionedCall-70570**
f%R#
!__inference__traced_restore_70569*
Tout
2��)
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_66965

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: �
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�T
batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
_output_shapes	
:�*
T0�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*D
_input_shapes3
1:�������������������::::28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
�4
�
@__inference_model_layer_call_and_return_conditional_losses_68012

inputs?
;token_and_position_embedding_statefulpartitionedcall_args_1?
;token_and_position_embedding_statefulpartitionedcall_args_24
0transformer_block_statefulpartitionedcall_args_14
0transformer_block_statefulpartitionedcall_args_24
0transformer_block_statefulpartitionedcall_args_34
0transformer_block_statefulpartitionedcall_args_44
0transformer_block_statefulpartitionedcall_args_54
0transformer_block_statefulpartitionedcall_args_64
0transformer_block_statefulpartitionedcall_args_74
0transformer_block_statefulpartitionedcall_args_84
0transformer_block_statefulpartitionedcall_args_95
1transformer_block_statefulpartitionedcall_args_105
1transformer_block_statefulpartitionedcall_args_115
1transformer_block_statefulpartitionedcall_args_125
1transformer_block_statefulpartitionedcall_args_135
1transformer_block_statefulpartitionedcall_args_145
1transformer_block_statefulpartitionedcall_args_155
1transformer_block_statefulpartitionedcall_args_166
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identity��+batch_normalization/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�)transformer_block/StatefulPartitionedCall�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs;token_and_position_embedding_statefulpartitionedcall_args_1;token_and_position_embedding_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-67027*`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_67021*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:������������	
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:00transformer_block_statefulpartitionedcall_args_10transformer_block_statefulpartitionedcall_args_20transformer_block_statefulpartitionedcall_args_30transformer_block_statefulpartitionedcall_args_40transformer_block_statefulpartitionedcall_args_50transformer_block_statefulpartitionedcall_args_60transformer_block_statefulpartitionedcall_args_70transformer_block_statefulpartitionedcall_args_80transformer_block_statefulpartitionedcall_args_91transformer_block_statefulpartitionedcall_args_101transformer_block_statefulpartitionedcall_args_111transformer_block_statefulpartitionedcall_args_121transformer_block_statefulpartitionedcall_args_131transformer_block_statefulpartitionedcall_args_141transformer_block_statefulpartitionedcall_args_151transformer_block_statefulpartitionedcall_args_16*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-67634*U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_67606*
Tout
2�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-67747*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67734�
$global_max_pooling1d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-66986*X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_66980*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:�����������
dropout_2/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-67800*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_67788*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:�����������
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-67822*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_67816�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-67850*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_67844*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall: : : : : : : : : : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : 
�
�
3__inference_batch_normalization_layer_call_fn_69743

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-67747*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67734*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*-
_output_shapes
:������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*<
_input_shapes+
):�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69725

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: �
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�T
batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
_output_shapes	
:�*
T0i
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:������������
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes	
:�*
T0x
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
�
�
*__inference_sequential_layer_call_fn_69953

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-66790*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_66789*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
�
P
4__inference_global_max_pooling1d_layer_call_fn_66989

inputs
identity�
PartitionedCallPartitionedCallinputs*X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_66980*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:������������������*,
_gradient_op_typePartitionedCall-66986i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:������������������*
T0"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:& "
 
_user_specified_nameinputs
�
�
3__inference_batch_normalization_layer_call_fn_69734

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-67737*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67711�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
�
�
'__inference_dense_4_layer_call_fn_70004

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-66701*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_66695*
Tout
2*-
config_proto

GPU

CPU2*0J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*4
_input_shapes#
!:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�	
�
B__inference_dense_7_layer_call_and_return_conditional_losses_67844

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:���������*
T0�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
1__inference_transformer_block_layer_call_fn_69520

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*,
_gradient_op_typePartitionedCall-67610*U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_67336*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*l
_input_shapes[
Y:�����������::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : 
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_66764
input_1*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity��dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-66701*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_66695*
Tout
2�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_66746*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-66752�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall: : : :' #
!
_user_specified_name	input_1: 
� 
�
B__inference_dense_5_layer_call_and_return_conditional_losses_70038

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:_
Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       E
Tensordot/ShapeShapeinputs*
_output_shapes
:*
T0Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0[
Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������k
Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��j
Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0\
Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*4
_input_shapes#
!:�����������::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�#
�
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_67021
x=
9embedding_1_embedding_lookup_read_readvariableop_resource;
7embedding_embedding_lookup_read_readvariableop_resource
identity��embedding/embedding_lookup�.embedding/embedding_lookup/Read/ReadVariableOp�embedding_1/embedding_lookup�0embedding_1/embedding_lookup/Read/ReadVariableOp6
ShapeShapex*
_output_shapes
:*
T0f
strided_slice/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
range/startConst*
value	B : *
dtype0*
_output_shapes
: M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: w
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:����������
0embedding_1/embedding_lookup/Read/ReadVariableOpReadVariableOp9embedding_1_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
���
%embedding_1/embedding_lookup/IdentityIdentity8embedding_1/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
embedding_1/embedding_lookupResourceGather9embedding_1_embedding_lookup_read_readvariableop_resourcerange:output:01^embedding_1/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:����������*C
_class9
75loc:@embedding_1/embedding_lookup/Read/ReadVariableOp*
Tindices0�
'embedding_1/embedding_lookup/Identity_1Identity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@embedding_1/embedding_lookup/Read/ReadVariableOp*(
_output_shapes
:�����������
'embedding_1/embedding_lookup/Identity_2Identity0embedding_1/embedding_lookup/Identity_1:output:0*
T0*(
_output_shapes
:����������[
embedding/CastCastx*

DstT0*(
_output_shapes
:����������*

SrcT0�
.embedding/embedding_lookup/Read/ReadVariableOpReadVariableOp7embedding_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
#embedding/embedding_lookup/IdentityIdentity6embedding/embedding_lookup/Read/ReadVariableOp:value:0*
_output_shapes
:	�*
T0�
embedding/embedding_lookupResourceGather7embedding_embedding_lookup_read_readvariableop_resourceembedding/Cast:y:0/^embedding/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@embedding/embedding_lookup/Read/ReadVariableOp*
Tindices0*
dtype0*-
_output_shapes
:������������
%embedding/embedding_lookup/Identity_1Identity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@embedding/embedding_lookup/Read/ReadVariableOp*-
_output_shapes
:������������
%embedding/embedding_lookup/Identity_2Identity.embedding/embedding_lookup/Identity_1:output:0*-
_output_shapes
:�����������*
T0�
addAddV2.embedding/embedding_lookup/Identity_2:output:00embedding_1/embedding_lookup/Identity_2:output:0*
T0*-
_output_shapes
:������������
IdentityIdentityadd:z:0^embedding/embedding_lookup/^embedding/embedding_lookup/Read/ReadVariableOp^embedding_1/embedding_lookup1^embedding_1/embedding_lookup/Read/ReadVariableOp*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������::2`
.embedding/embedding_lookup/Read/ReadVariableOp.embedding/embedding_lookup/Read/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2d
0embedding_1/embedding_lookup/Read/ReadVariableOp0embedding_1/embedding_lookup/Read/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup: :! 

_user_specified_namex: 
��
�
@__inference_model_layer_call_and_return_conditional_losses_68838

inputsZ
Vtoken_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resourceX
Ttoken_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resourceU
Qtransformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resourceO
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resourceK
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resourceJ
Ftransformer_block_sequential_dense_4_tensordot_readvariableop_resourceH
Dtransformer_block_sequential_dense_4_biasadd_readvariableop_resourceJ
Ftransformer_block_sequential_dense_5_tensordot_readvariableop_resourceH
Dtransformer_block_sequential_dense_5_biasadd_readvariableop_resourceQ
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resourceM
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�7token_and_position_embedding/embedding/embedding_lookup�Ktoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp�9token_and_position_embedding/embedding_1/embedding_lookup�Mtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp�>transformer_block/layer_normalization/batchnorm/ReadVariableOp�Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp�@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp�Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp�Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp�Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp�Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp�Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp�Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp�Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp�Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp�Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp�;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp�=transformer_block/sequential/dense_4/Tensordot/ReadVariableOp�;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp�=transformer_block/sequential/dense_5/Tensordot/ReadVariableOpX
"token_and_position_embedding/ShapeShapeinputs*
T0*
_output_shapes
:�
0token_and_position_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������|
2token_and_position_embedding/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:|
2token_and_position_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskj
(token_and_position_embedding/range/startConst*
dtype0*
_output_shapes
: *
value	B : j
(token_and_position_embedding/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :�
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*#
_output_shapes
:����������
Mtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOpReadVariableOpVtoken_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
���
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityUtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
9token_and_position_embedding/embedding_1/embedding_lookupResourceGatherVtoken_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resource+token_and_position_embedding/range:output:0N^token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:����������*`
_classV
TRloc:@token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp*
Tindices0�
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*(
_output_shapes
:����������*
T0*`
_classV
TRloc:@token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp�
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_2IdentityMtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*(
_output_shapes
:����������}
+token_and_position_embedding/embedding/CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:�����������
Ktoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOpReadVariableOpTtoken_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityStoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
7token_and_position_embedding/embedding/embedding_lookupResourceGatherTtoken_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resource/token_and_position_embedding/embedding/Cast:y:0L^token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*
dtype0*-
_output_shapes
:�����������*^
_classT
RPloc:@token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp�
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1Identity@token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*^
_classT
RPloc:@token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp*-
_output_shapes
:������������
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_2IdentityKtoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:������������
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_2:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_2:output:0*
T0*-
_output_shapes
:������������
1transformer_block/multi_head_self_attention/ShapeShape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:�
?transformer_block/multi_head_self_attention/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: �
Atransformer_block/multi_head_self_attention/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:�
Atransformer_block/multi_head_self_attention/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
9transformer_block/multi_head_self_attention/strided_sliceStridedSlice:transformer_block/multi_head_self_attention/Shape:output:0Htransformer_block/multi_head_self_attention/strided_slice/stack:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0�
Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
@transformer_block/multi_head_self_attention/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
@transformer_block/multi_head_self_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Atransformer_block/multi_head_self_attention/dense/Tensordot/ShapeShape$token_and_position_embedding/add:z:0*
_output_shapes
:*
T0�
Itransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Jtransformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Jtransformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Ttransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0�
Atransformer_block/multi_head_self_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
@transformer_block/multi_head_self_attention/dense/Tensordot/ProdProdMtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1ProdOtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Ltransformer_block/multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Gtransformer_block/multi_head_self_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Btransformer_block/multi_head_self_attention/dense/Tensordot/concatConcatV2Itransformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Ptransformer_block/multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Atransformer_block/multi_head_self_attention/dense/Tensordot/stackPackItransformer_block/multi_head_self_attention/dense/Tensordot/Prod:output:0Ktransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Etransformer_block/multi_head_self_attention/dense/Tensordot/transpose	Transpose$token_and_position_embedding/add:z:0Ktransformer_block/multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Ctransformer_block/multi_head_self_attention/dense/Tensordot/ReshapeReshapeItransformer_block/multi_head_self_attention/dense/Tensordot/transpose:y:0Jtransformer_block/multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ltransformer_block/multi_head_self_attention/dense/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
Gtransformer_block/multi_head_self_attention/dense/Tensordot/transpose_1	TransposeRtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0Utransformer_block/multi_head_self_attention/dense/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Ktransformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
Etransformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1ReshapeKtransformer_block/multi_head_self_attention/dense/Tensordot/transpose_1:y:0Ttransformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
Btransformer_block/multi_head_self_attention/dense/Tensordot/MatMulMatMulLtransformer_block/multi_head_self_attention/dense/Tensordot/Reshape:output:0Ntransformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Itransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Mtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense/Tensordot/Const_2:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
;transformer_block/multi_head_self_attention/dense/TensordotReshapeLtransformer_block/multi_head_self_attention/dense/Tensordot/MatMul:product:0Mtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
9transformer_block/multi_head_self_attention/dense/BiasAddBiasAddDtransformer_block/multi_head_self_attention/dense/Tensordot:output:0Ptransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/ShapeShape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0�
Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Htransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Const:output:0*
_output_shapes
: *
T0�
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Itransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose	Transpose$token_and_position_embedding/add:z:0Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
Itransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1	TransposeTtransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0Wtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1ReshapeMtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1:y:0Vtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0�
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
=transformer_block/multi_head_self_attention/dense_1/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
;transformer_block/multi_head_self_attention/dense_1/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_1/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/ShapeShape$token_and_position_embedding/add:z:0*
_output_shapes
:*
T0�
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Htransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0�
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Itransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose	Transpose$token_and_position_embedding/add:z:0Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Itransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1	TransposeTtransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0Wtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
Gtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1ReshapeMtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1:y:0Vtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
=transformer_block/multi_head_self_attention/dense_2/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
;transformer_block/multi_head_self_attention/dense_2/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_2/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
;transformer_block/multi_head_self_attention/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: }
;transformer_block/multi_head_self_attention/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :}
;transformer_block/multi_head_self_attention/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
value	B :`�
9transformer_block/multi_head_self_attention/Reshape/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/1:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/2:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:�
3transformer_block/multi_head_self_attention/ReshapeReshapeBtransformer_block/multi_head_self_attention/dense/BiasAdd:output:0Btransformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
:transformer_block/multi_head_self_attention/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
5transformer_block/multi_head_self_attention/transpose	Transpose<transformer_block/multi_head_self_attention/Reshape:output:0Ctransformer_block/multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
=transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
=transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
=transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
;transformer_block/multi_head_self_attention/Reshape_1/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
T0*
N*
_output_shapes
:�
5transformer_block/multi_head_self_attention/Reshape_1ReshapeDtransformer_block/multi_head_self_attention/dense_1/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_1/shape:output:0*8
_output_shapes&
$:"������������������`*
T0�
<transformer_block/multi_head_self_attention/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
7transformer_block/multi_head_self_attention/transpose_1	Transpose>transformer_block/multi_head_self_attention/Reshape_1:output:0Etransformer_block/multi_head_self_attention/transpose_1/perm:output:0*8
_output_shapes&
$:"������������������`*
T0�
=transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
=transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
=transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
;transformer_block/multi_head_self_attention/Reshape_2/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
T0*
N*
_output_shapes
:�
5transformer_block/multi_head_self_attention/Reshape_2ReshapeDtransformer_block/multi_head_self_attention/dense_2/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
<transformer_block/multi_head_self_attention/transpose_2/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
7transformer_block/multi_head_self_attention/transpose_2	Transpose>transformer_block/multi_head_self_attention/Reshape_2:output:0Etransformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
2transformer_block/multi_head_self_attention/MatMulBatchMatMulV29transformer_block/multi_head_self_attention/transpose:y:0;transformer_block/multi_head_self_attention/transpose_1:y:0*A
_output_shapes/
-:+���������������������������*
adj_y(*
T0�
3transformer_block/multi_head_self_attention/Shape_1Shape;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:�
Atransformer_block/multi_head_self_attention/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
����������
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: �
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
;transformer_block/multi_head_self_attention/strided_slice_1StridedSlice<transformer_block/multi_head_self_attention/Shape_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice_1/stack:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
0transformer_block/multi_head_self_attention/CastCastDtransformer_block/multi_head_self_attention/strided_slice_1:output:0*

SrcT0*

DstT0*
_output_shapes
: �
0transformer_block/multi_head_self_attention/SqrtSqrt4transformer_block/multi_head_self_attention/Cast:y:0*
_output_shapes
: *
T0�
3transformer_block/multi_head_self_attention/truedivRealDiv;transformer_block/multi_head_self_attention/MatMul:output:04transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
3transformer_block/multi_head_self_attention/SoftmaxSoftmax7transformer_block/multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
4transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2=transformer_block/multi_head_self_attention/Softmax:softmax:0;transformer_block/multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"������������������`�
<transformer_block/multi_head_self_attention/transpose_3/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
7transformer_block/multi_head_self_attention/transpose_3	Transpose=transformer_block/multi_head_self_attention/MatMul_1:output:0Etransformer_block/multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
=transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: �
=transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: �
;transformer_block/multi_head_self_attention/Reshape_3/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
T0*
N*
_output_shapes
:�
5transformer_block/multi_head_self_attention/Reshape_3Reshape;transformer_block/multi_head_self_attention/transpose_3:y:0Dtransformer_block/multi_head_self_attention/Reshape_3/shape:output:0*5
_output_shapes#
!:�������������������*
T0�
Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/ShapeShape>transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Htransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
_output_shapes
: *
T0�
Itransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0�
Gtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose	Transpose>transformer_block/multi_head_self_attention/Reshape_3:output:0Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Itransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1	TransposeTtransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0Wtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1ReshapeMtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1:y:0Vtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
=transformer_block/multi_head_self_attention/dense_3/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
;transformer_block/multi_head_self_attention/dense_3/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_3/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*5
_output_shapes#
!:�������������������*
T0�
"transformer_block/dropout/IdentityIdentityDtransformer_block/multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
transformer_block/addAddV2$token_and_position_embedding/add:z:0+transformer_block/dropout/Identity:output:0*
T0*-
_output_shapes
:������������
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*,
_output_shapes
:�����������
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*,
_output_shapes
:����������*
T0�
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:�
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(*
T0z
5transformer_block/layer_normalization/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:�����������
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*-
_output_shapes
:�����������*
T0�
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
=transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��}
3transformer_block/sequential/dense_4/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
3transformer_block/sequential/dense_4/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
4transformer_block/sequential/dense_4/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
7transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_4/Tensordot/Shape:output:0<transformer_block/sequential/dense_4/Tensordot/free:output:0Etransformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
>transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
9transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_4/Tensordot/Shape:output:0<transformer_block/sequential/dense_4/Tensordot/axes:output:0Gtransformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0~
4transformer_block/sequential/dense_4/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
3transformer_block/sequential/dense_4/Tensordot/ProdProd@transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_4/Tensordot/Const:output:0*
_output_shapes
: *
T0�
6transformer_block/sequential/dense_4/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
5transformer_block/sequential/dense_4/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
_output_shapes
: *
T0|
:transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
5transformer_block/sequential/dense_4/Tensordot/concatConcatV2<transformer_block/sequential/dense_4/Tensordot/free:output:0<transformer_block/sequential/dense_4/Tensordot/axes:output:0Ctransformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
4transformer_block/sequential/dense_4/Tensordot/stackPack<transformer_block/sequential/dense_4/Tensordot/Prod:output:0>transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
8transformer_block/sequential/dense_4/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0>transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
6transformer_block/sequential/dense_4/Tensordot/ReshapeReshape<transformer_block/sequential/dense_4/Tensordot/transpose:y:0=transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
?transformer_block/sequential/dense_4/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
:transformer_block/sequential/dense_4/Tensordot/transpose_1	TransposeEtransformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0Htransformer_block/sequential/dense_4/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
>transformer_block/sequential/dense_4/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
8transformer_block/sequential/dense_4/Tensordot/Reshape_1Reshape>transformer_block/sequential/dense_4/Tensordot/transpose_1:y:0Gtransformer_block/sequential/dense_4/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
5transformer_block/sequential/dense_4/Tensordot/MatMulMatMul?transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Atransformer_block/sequential/dense_4/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0�
6transformer_block/sequential/dense_4/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:~
<transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
7transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
.transformer_block/sequential/dense_4/TensordotReshape?transformer_block/sequential/dense_4/Tensordot/MatMul:product:0@transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
,transformer_block/sequential/dense_4/BiasAddBiasAdd7transformer_block/sequential/dense_4/Tensordot:output:0Ctransformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
)transformer_block/sequential/dense_4/ReluRelu5transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*-
_output_shapes
:������������
=transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��}
3transformer_block/sequential/dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
3transformer_block/sequential/dense_5/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
4transformer_block/sequential/dense_5/Tensordot/ShapeShape7transformer_block/sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
7transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_5/Tensordot/Shape:output:0<transformer_block/sequential/dense_5/Tensordot/free:output:0Etransformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0�
>transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
9transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_5/Tensordot/Shape:output:0<transformer_block/sequential/dense_5/Tensordot/axes:output:0Gtransformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense_5/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: �
3transformer_block/sequential/dense_5/Tensordot/ProdProd@transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: �
6transformer_block/sequential/dense_5/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
5transformer_block/sequential/dense_5/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
5transformer_block/sequential/dense_5/Tensordot/concatConcatV2<transformer_block/sequential/dense_5/Tensordot/free:output:0<transformer_block/sequential/dense_5/Tensordot/axes:output:0Ctransformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
4transformer_block/sequential/dense_5/Tensordot/stackPack<transformer_block/sequential/dense_5/Tensordot/Prod:output:0>transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
8transformer_block/sequential/dense_5/Tensordot/transpose	Transpose7transformer_block/sequential/dense_4/Relu:activations:0>transformer_block/sequential/dense_5/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
6transformer_block/sequential/dense_5/Tensordot/ReshapeReshape<transformer_block/sequential/dense_5/Tensordot/transpose:y:0=transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
?transformer_block/sequential/dense_5/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
:transformer_block/sequential/dense_5/Tensordot/transpose_1	TransposeEtransformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0Htransformer_block/sequential/dense_5/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0�
>transformer_block/sequential/dense_5/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
8transformer_block/sequential/dense_5/Tensordot/Reshape_1Reshape>transformer_block/sequential/dense_5/Tensordot/transpose_1:y:0Gtransformer_block/sequential/dense_5/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
5transformer_block/sequential/dense_5/Tensordot/MatMulMatMul?transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Atransformer_block/sequential/dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense_5/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�~
<transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
7transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
.transformer_block/sequential/dense_5/TensordotReshape?transformer_block/sequential/dense_5/Tensordot/MatMul:product:0@transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
,transformer_block/sequential/dense_5/BiasAddBiasAdd7transformer_block/sequential/dense_5/Tensordot:output:0Ctransformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
$transformer_block/dropout_1/IdentityIdentity5transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*-
_output_shapes
:������������
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_1/Identity:output:0*-
_output_shapes
:�����������*
T0�
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:�����������
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*-
_output_shapes
:�����������*
T0�
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*,
_output_shapes
:����������|
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5�
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*,
_output_shapes
:����������*
T0�
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:�����������b
 batch_normalization/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: b
 batch_normalization/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: �
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: �
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�h
#batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_1Mul;transformer_block/layer_normalization_1/batchnorm/add_1:z:0%batch_normalization/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
_output_shapes	
:�*
T0�
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*-
_output_shapes
:�����������*
T0l
*global_max_pooling1d/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: �
global_max_pooling1d/MaxMax'batch_normalization/batchnorm/add_1:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������t
dropout_2/IdentityIdentity!global_max_pooling1d/Max:output:0*
T0*(
_output_shapes
:�����������
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
dense_6/MatMulMatMuldropout_2/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0a
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	��
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������*
T0�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*'
_output_shapes
:���������*
T0�
IdentityIdentitydense_7/Sigmoid:y:0-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookupL^token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp:^token_and_position_embedding/embedding_1/embedding_lookupN^token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpI^transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpK^transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpK^transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpM^transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpK^transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpM^transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpK^transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpM^transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp<^transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp>^transformer_block/sequential/dense_4/Tensordot/ReadVariableOp<^transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp>^transformer_block/sequential/dense_5/Tensordot/ReadVariableOp*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::2�
Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpJtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp2�
Mtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOpMtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp2�
Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2�
Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpJtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2�
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2~
=transformer_block/sequential/dense_5/Tensordot/ReadVariableOp=transformer_block/sequential/dense_5/Tensordot/ReadVariableOp2z
;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp2�
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2�
Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpJtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2�
Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpHtransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp2�
Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2�
Ktoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOpKtoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp2z
;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp2�
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2�
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2~
=transformer_block/sequential/dense_4/Tensordot/ReadVariableOp=transformer_block/sequential/dense_4/Tensordot/ReadVariableOp2�
Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2�
Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpJtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_2: : : : : : : : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : 
��
�
L__inference_transformer_block_layer_call_and_return_conditional_losses_69499

inputsE
Amulti_head_self_attention_dense_tensordot_readvariableop_resourceC
?multi_head_self_attention_dense_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_1_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_1_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_2_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_2_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_3_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_3_biasadd_readvariableop_resource=
9layer_normalization_batchnorm_mul_readvariableop_resource9
5layer_normalization_batchnorm_readvariableop_resource8
4sequential_dense_4_tensordot_readvariableop_resource6
2sequential_dense_4_biasadd_readvariableop_resource8
4sequential_dense_5_tensordot_readvariableop_resource6
2sequential_dense_5_biasadd_readvariableop_resource?
;layer_normalization_1_batchnorm_mul_readvariableop_resource;
7layer_normalization_1_batchnorm_readvariableop_resource
identity��,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�6multi_head_self_attention/dense/BiasAdd/ReadVariableOp�8multi_head_self_attention/dense/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp�)sequential/dense_4/BiasAdd/ReadVariableOp�+sequential/dense_4/Tensordot/ReadVariableOp�)sequential/dense_5/BiasAdd/ReadVariableOp�+sequential/dense_5/Tensordot/ReadVariableOpU
multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:w
-multi_head_self_attention/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/multi_head_self_attention/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/multi_head_self_attention/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
8multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��x
.multi_head_self_attention/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
.multi_head_self_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:e
/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense/Tensordot/GatherV2GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/free:output:0@multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
4multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/axes:output:0Bmulti_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
.multi_head_self_attention/dense/Tensordot/ProdProd;multi_head_self_attention/dense/Tensordot/GatherV2:output:08multi_head_self_attention/dense/Tensordot/Const:output:0*
_output_shapes
: *
T0{
1multi_head_self_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense/Tensordot/Prod_1Prod=multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0:multi_head_self_attention/dense/Tensordot/Const_1:output:0*
_output_shapes
: *
T0w
5multi_head_self_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
0multi_head_self_attention/dense/Tensordot/concatConcatV27multi_head_self_attention/dense/Tensordot/free:output:07multi_head_self_attention/dense/Tensordot/axes:output:0>multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
/multi_head_self_attention/dense/Tensordot/stackPack7multi_head_self_attention/dense/Tensordot/Prod:output:09multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0�
3multi_head_self_attention/dense/Tensordot/transpose	Transposeinputs9multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
1multi_head_self_attention/dense/Tensordot/ReshapeReshape7multi_head_self_attention/dense/Tensordot/transpose:y:08multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
:multi_head_self_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense/Tensordot/transpose_1	Transpose@multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0Cmulti_head_self_attention/dense/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0�
9multi_head_self_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
3multi_head_self_attention/dense/Tensordot/Reshape_1Reshape9multi_head_self_attention/dense/Tensordot/transpose_1:y:0Bmulti_head_self_attention/dense/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
0multi_head_self_attention/dense/Tensordot/MatMulMatMul:multi_head_self_attention/dense/Tensordot/Reshape:output:0<multi_head_self_attention/dense/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������|
1multi_head_self_attention/dense/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:y
7multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense/Tensordot/concat_1ConcatV2;multi_head_self_attention/dense/Tensordot/GatherV2:output:0:multi_head_self_attention/dense/Tensordot/Const_2:output:0@multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
)multi_head_self_attention/dense/TensordotReshape:multi_head_self_attention/dense/Tensordot/MatMul:product:0;multi_head_self_attention/dense/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
6multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
'multi_head_self_attention/dense/BiasAddBiasAdd2multi_head_self_attention/dense/Tensordot:output:0>multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:g
1multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : �
4multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/free:output:0Bmulti_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0Dmulti_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0{
1multi_head_self_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_1/Tensordot/ProdProd=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_1/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
2multi_head_self_attention/dense_1/Tensordot/Prod_1Prod?multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
_output_shapes
: *
T0y
7multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense_1/Tensordot/concatConcatV29multi_head_self_attention/dense_1/Tensordot/free:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0@multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
1multi_head_self_attention/dense_1/Tensordot/stackPack9multi_head_self_attention/dense_1/Tensordot/Prod:output:0;multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
3multi_head_self_attention/dense_1/Tensordot/ReshapeReshape9multi_head_self_attention/dense_1/Tensordot/transpose:y:0:multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<multi_head_self_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_1/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_1/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_1/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_1/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_1/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_1/Tensordot/MatMulMatMul<multi_head_self_attention/dense_1/Tensordot/Reshape:output:0>multi_head_self_attention/dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������~
3multi_head_self_attention/dense_1/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_1/TensordotReshape<multi_head_self_attention/dense_1/Tensordot/MatMul:product:0=multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_1/BiasAddBiasAdd4multi_head_self_attention/dense_1/Tensordot:output:0@multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_2/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       g
1multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/free:output:0Bmulti_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0Dmulti_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention/dense_2/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: �
0multi_head_self_attention/dense_2/Tensordot/ProdProd=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_2/Tensordot/Prod_1Prod?multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
_output_shapes
: *
T0y
7multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense_2/Tensordot/concatConcatV29multi_head_self_attention/dense_2/Tensordot/free:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0@multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
1multi_head_self_attention/dense_2/Tensordot/stackPack9multi_head_self_attention/dense_2/Tensordot/Prod:output:0;multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
3multi_head_self_attention/dense_2/Tensordot/ReshapeReshape9multi_head_self_attention/dense_2/Tensordot/transpose:y:0:multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<multi_head_self_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_2/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_2/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_2/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_2/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_2/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
2multi_head_self_attention/dense_2/Tensordot/MatMulMatMul<multi_head_self_attention/dense_2/Tensordot/Reshape:output:0>multi_head_self_attention/dense_2/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0~
3multi_head_self_attention/dense_2/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_2/TensordotReshape<multi_head_self_attention/dense_2/Tensordot/MatMul:product:0=multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_2/BiasAddBiasAdd4multi_head_self_attention/dense_2/Tensordot:output:0@multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0t
)multi_head_self_attention/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: k
)multi_head_self_attention/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: k
)multi_head_self_attention/Reshape/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:�
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/dense/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
(multi_head_self_attention/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������`v
+multi_head_self_attention/Reshape_1/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_1/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
_output_shapes
:*
T0�
#multi_head_self_attention/Reshape_1Reshape2multi_head_self_attention/dense_1/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"������������������`v
+multi_head_self_attention/Reshape_2/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_2/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_2Reshape2multi_head_self_attention/dense_2/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_2/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
adj_y(*
T0*A
_output_shapes/
-:+���������������������������z
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:�
/multi_head_self_attention/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������{
1multi_head_self_attention/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: {
1multi_head_self_attention/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0�
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

SrcT0*

DstT0*
_output_shapes
: k
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: �
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*A
_output_shapes/
-:+���������������������������*
T0�
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*8
_output_shapes&
$:"������������������`*
T0v
+multi_head_self_attention/Reshape_3/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: n
+multi_head_self_attention/Reshape_3/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_3/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
1multi_head_self_attention/dense_3/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:{
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : �
4multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/free:output:0Bmulti_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0}
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0Dmulti_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0{
1multi_head_self_attention/dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_3/Tensordot/ProdProd=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_3/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
2multi_head_self_attention/dense_3/Tensordot/Prod_1Prod?multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense_3/Tensordot/concatConcatV29multi_head_self_attention/dense_3/Tensordot/free:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0@multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_3/Tensordot/stackPack9multi_head_self_attention/dense_3/Tensordot/Prod:output:0;multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_3/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:0;multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
3multi_head_self_attention/dense_3/Tensordot/ReshapeReshape9multi_head_self_attention/dense_3/Tensordot/transpose:y:0:multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<multi_head_self_attention/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_3/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_3/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0�
;multi_head_self_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_3/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_3/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_3/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_3/Tensordot/MatMulMatMul<multi_head_self_attention/dense_3/Tensordot/Reshape:output:0>multi_head_self_attention/dense_3/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������~
3multi_head_self_attention/dense_3/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_3/TensordotReshape<multi_head_self_attention/dense_3/Tensordot/MatMul:product:0=multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_3/BiasAddBiasAdd4multi_head_self_attention/dense_3/Tensordot:output:0@multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*5
_output_shapes#
!:�������������������*
T0�
dropout/IdentityIdentity2multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������g
addAddV2inputsdropout/Identity:output:0*-
_output_shapes
:�����������*
T0|
2layer_normalization/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*,
_output_shapes
:����������*
T0�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
6layer_normalization/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:�����������
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*-
_output_shapes
:�����������*
T0�
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��k
!sequential/dense_4/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:r
!sequential/dense_4/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:y
"sequential/dense_4/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0l
"sequential/dense_4/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
&sequential/dense_4/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������~
-sequential/dense_4/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
(sequential/dense_4/Tensordot/transpose_1	Transpose3sequential/dense_4/Tensordot/ReadVariableOp:value:06sequential/dense_4/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��}
,sequential/dense_4/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
&sequential/dense_4/Tensordot/Reshape_1Reshape,sequential/dense_4/Tensordot/transpose_1:y:05sequential/dense_4/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:0/sequential/dense_4/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������o
$sequential/dense_4/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:l
*sequential/dense_4/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������|
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*-
_output_shapes
:������������
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��k
!sequential/dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:r
!sequential/dense_5/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:w
"sequential/dense_5/Tensordot/ShapeShape%sequential/dense_4/Relu:activations:0*
_output_shapes
:*
T0l
*sequential/dense_5/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0n
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0l
"sequential/dense_5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_5/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
&sequential/dense_5/Tensordot/transpose	Transpose%sequential/dense_4/Relu:activations:0,sequential/dense_5/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������~
-sequential/dense_5/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
(sequential/dense_5/Tensordot/transpose_1	Transpose3sequential/dense_5/Tensordot/ReadVariableOp:value:06sequential/dense_5/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��}
,sequential/dense_5/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
&sequential/dense_5/Tensordot/Reshape_1Reshape,sequential/dense_5/Tensordot/transpose_1:y:05sequential/dense_5/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:0/sequential/dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������o
$sequential/dense_5/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:l
*sequential/dense_5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0{
dropout_1/IdentityIdentity#sequential/dense_5/BiasAdd:output:0*
T0*-
_output_shapes
:������������
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*-
_output_shapes
:�����������*
T0~
4layer_normalization_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(*
T0�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:�����������
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
8layer_normalization_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:�����������
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*-
_output_shapes
:�����������*
T0�
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp7^multi_head_self_attention/dense/BiasAdd/ReadVariableOp9^multi_head_self_attention/dense/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_1/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_2/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp,^sequential/dense_5/Tensordot/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*l
_input_shapes[
Y:�����������::::::::::::::::2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2t
8multi_head_self_attention/dense/Tensordot/ReadVariableOp8multi_head_self_attention/dense/Tensordot/ReadVariableOp2x
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2t
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2x
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp2t
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_5/Tensordot/ReadVariableOp+sequential/dense_5/Tensordot/ReadVariableOp2p
6multi_head_self_attention/dense/BiasAdd/ReadVariableOp6multi_head_self_attention/dense/BiasAdd/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2x
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : 
�
�
L__inference_transformer_block_layer_call_and_return_conditional_losses_69231

inputsE
Amulti_head_self_attention_dense_tensordot_readvariableop_resourceC
?multi_head_self_attention_dense_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_1_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_1_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_2_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_2_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_3_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_3_biasadd_readvariableop_resource=
9layer_normalization_batchnorm_mul_readvariableop_resource9
5layer_normalization_batchnorm_readvariableop_resource8
4sequential_dense_4_tensordot_readvariableop_resource6
2sequential_dense_4_biasadd_readvariableop_resource8
4sequential_dense_5_tensordot_readvariableop_resource6
2sequential_dense_5_biasadd_readvariableop_resource?
;layer_normalization_1_batchnorm_mul_readvariableop_resource;
7layer_normalization_1_batchnorm_readvariableop_resource
identity��,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�6multi_head_self_attention/dense/BiasAdd/ReadVariableOp�8multi_head_self_attention/dense/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp�)sequential/dense_4/BiasAdd/ReadVariableOp�+sequential/dense_4/Tensordot/ReadVariableOp�)sequential/dense_5/BiasAdd/ReadVariableOp�+sequential/dense_5/Tensordot/ReadVariableOpU
multi_head_self_attention/ShapeShapeinputs*
_output_shapes
:*
T0w
-multi_head_self_attention/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/multi_head_self_attention/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/multi_head_self_attention/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: �
8multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��x
.multi_head_self_attention/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
.multi_head_self_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:e
/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense/Tensordot/GatherV2GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/free:output:0@multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0{
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/axes:output:0Bmulti_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0y
/multi_head_self_attention/dense/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: �
.multi_head_self_attention/dense/Tensordot/ProdProd;multi_head_self_attention/dense/Tensordot/GatherV2:output:08multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense/Tensordot/Prod_1Prod=multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0:multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/dense/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
0multi_head_self_attention/dense/Tensordot/concatConcatV27multi_head_self_attention/dense/Tensordot/free:output:07multi_head_self_attention/dense/Tensordot/axes:output:0>multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
/multi_head_self_attention/dense/Tensordot/stackPack7multi_head_self_attention/dense/Tensordot/Prod:output:09multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
3multi_head_self_attention/dense/Tensordot/transpose	Transposeinputs9multi_head_self_attention/dense/Tensordot/concat:output:0*-
_output_shapes
:�����������*
T0�
1multi_head_self_attention/dense/Tensordot/ReshapeReshape7multi_head_self_attention/dense/Tensordot/transpose:y:08multi_head_self_attention/dense/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0�
:multi_head_self_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense/Tensordot/transpose_1	Transpose@multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0Cmulti_head_self_attention/dense/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
9multi_head_self_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
3multi_head_self_attention/dense/Tensordot/Reshape_1Reshape9multi_head_self_attention/dense/Tensordot/transpose_1:y:0Bmulti_head_self_attention/dense/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
0multi_head_self_attention/dense/Tensordot/MatMulMatMul:multi_head_self_attention/dense/Tensordot/Reshape:output:0<multi_head_self_attention/dense/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������|
1multi_head_self_attention/dense/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:y
7multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense/Tensordot/concat_1ConcatV2;multi_head_self_attention/dense/Tensordot/GatherV2:output:0:multi_head_self_attention/dense/Tensordot/Const_2:output:0@multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
)multi_head_self_attention/dense/TensordotReshape:multi_head_self_attention/dense/Tensordot/MatMul:product:0;multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
6multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
'multi_head_self_attention/dense/BiasAddBiasAdd2multi_head_self_attention/dense/Tensordot:output:0>multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:g
1multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/free:output:0Bmulti_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0}
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0Dmulti_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0{
1multi_head_self_attention/dense_1/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: �
0multi_head_self_attention/dense_1/Tensordot/ProdProd=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_1/Tensordot/Const:output:0*
_output_shapes
: *
T0}
3multi_head_self_attention/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_1/Tensordot/Prod_1Prod?multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense_1/Tensordot/concatConcatV29multi_head_self_attention/dense_1/Tensordot/free:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0@multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_1/Tensordot/stackPack9multi_head_self_attention/dense_1/Tensordot/Prod:output:0;multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
3multi_head_self_attention/dense_1/Tensordot/ReshapeReshape9multi_head_self_attention/dense_1/Tensordot/transpose:y:0:multi_head_self_attention/dense_1/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0�
<multi_head_self_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_1/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_1/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0�
;multi_head_self_attention/dense_1/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
5multi_head_self_attention/dense_1/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_1/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_1/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_1/Tensordot/MatMulMatMul<multi_head_self_attention/dense_1/Tensordot/Reshape:output:0>multi_head_self_attention/dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������~
3multi_head_self_attention/dense_1/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
4multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_1/TensordotReshape<multi_head_self_attention/dense_1/Tensordot/MatMul:product:0=multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_1/BiasAddBiasAdd4multi_head_self_attention/dense_1/Tensordot:output:0@multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:g
1multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/free:output:0Bmulti_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0}
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0Dmulti_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0{
1multi_head_self_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_2/Tensordot/ProdProd=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_2/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
2multi_head_self_attention/dense_2/Tensordot/Prod_1Prod?multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense_2/Tensordot/concatConcatV29multi_head_self_attention/dense_2/Tensordot/free:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0@multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_2/Tensordot/stackPack9multi_head_self_attention/dense_2/Tensordot/Prod:output:0;multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
3multi_head_self_attention/dense_2/Tensordot/ReshapeReshape9multi_head_self_attention/dense_2/Tensordot/transpose:y:0:multi_head_self_attention/dense_2/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0�
<multi_head_self_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_2/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_2/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_2/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_2/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_2/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_2/Tensordot/MatMulMatMul<multi_head_self_attention/dense_2/Tensordot/Reshape:output:0>multi_head_self_attention/dense_2/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������~
3multi_head_self_attention/dense_2/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_2/TensordotReshape<multi_head_self_attention/dense_2/Tensordot/MatMul:product:0=multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_2/BiasAddBiasAdd4multi_head_self_attention/dense_2/Tensordot:output:0@multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������t
)multi_head_self_attention/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: k
)multi_head_self_attention/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: k
)multi_head_self_attention/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
value	B :`�
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:�
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/dense/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*8
_output_shapes&
$:"������������������`*
T0�
(multi_head_self_attention/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*8
_output_shapes&
$:"������������������`*
T0v
+multi_head_self_attention/Reshape_1/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :m
+multi_head_self_attention/Reshape_1/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_1Reshape2multi_head_self_attention/dense_1/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"������������������`v
+multi_head_self_attention/Reshape_2/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������m
+multi_head_self_attention/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_2/shape/3Const*
dtype0*
_output_shapes
: *
value	B :`�
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
_output_shapes
:*
T0�
#multi_head_self_attention/Reshape_2Reshape2multi_head_self_attention/dense_2/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
adj_y(*
T0*A
_output_shapes/
-:+���������������������������z
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:�
/multi_head_self_attention/strided_slice_1/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:{
1multi_head_self_attention/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:{
1multi_head_self_attention/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask�
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

SrcT0*

DstT0*
_output_shapes
: k
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: �
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_3/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������`v
+multi_head_self_attention/Reshape_3/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: n
+multi_head_self_attention/Reshape_3/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
_output_shapes
:*
T0�
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_3/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:�
0multi_head_self_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
1multi_head_self_attention/dense_3/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
_output_shapes
:*
T0{
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/free:output:0Bmulti_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0}
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0Dmulti_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0{
1multi_head_self_attention/dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_3/Tensordot/ProdProd=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_3/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_3/Tensordot/Prod_1Prod?multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense_3/Tensordot/concatConcatV29multi_head_self_attention/dense_3/Tensordot/free:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0@multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_3/Tensordot/stackPack9multi_head_self_attention/dense_3/Tensordot/Prod:output:0;multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_3/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:0;multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
3multi_head_self_attention/dense_3/Tensordot/ReshapeReshape9multi_head_self_attention/dense_3/Tensordot/transpose:y:0:multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<multi_head_self_attention/dense_3/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
7multi_head_self_attention/dense_3/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_3/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_3/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
5multi_head_self_attention/dense_3/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_3/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_3/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_3/Tensordot/MatMulMatMul<multi_head_self_attention/dense_3/Tensordot/Reshape:output:0>multi_head_self_attention/dense_3/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0~
3multi_head_self_attention/dense_3/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_3/TensordotReshape<multi_head_self_attention/dense_3/Tensordot/MatMul:product:0=multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_3/BiasAddBiasAdd4multi_head_self_attention/dense_3/Tensordot:output:0@multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������Y
dropout/dropout/rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: w
dropout/dropout/ShapeShape2multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
dtype0*5
_output_shapes#
!:�������������������*
T0�
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*5
_output_shapes#
!:�������������������*
T0�
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*5
_output_shapes#
!:�������������������Z
dropout/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*5
_output_shapes#
!:��������������������
dropout/dropout/mulMul2multi_head_self_attention/dense_3/BiasAdd:output:0dropout/dropout/truediv:z:0*
T0*5
_output_shapes#
!:��������������������
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*5
_output_shapes#
!:�������������������*

SrcT0
�
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*5
_output_shapes#
!:�������������������*
T0g
addAddV2inputsdropout/dropout/mul_1:z:0*
T0*-
_output_shapes
:�����������|
2layer_normalization/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:�����������
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
6layer_normalization/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*,
_output_shapes
:����������h
#layer_normalization/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:�����������
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*-
_output_shapes
:�����������*
T0�
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��k
!sequential/dense_4/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:r
!sequential/dense_4/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:y
"sequential/dense_4/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0l
"sequential/dense_4/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
&sequential/dense_4/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0~
-sequential/dense_4/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
(sequential/dense_4/Tensordot/transpose_1	Transpose3sequential/dense_4/Tensordot/ReadVariableOp:value:06sequential/dense_4/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��}
,sequential/dense_4/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
&sequential/dense_4/Tensordot/Reshape_1Reshape,sequential/dense_4/Tensordot/transpose_1:y:05sequential/dense_4/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:0/sequential/dense_4/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������o
$sequential/dense_4/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:l
*sequential/dense_4/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������|
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*-
_output_shapes
:������������
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��k
!sequential/dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:r
!sequential/dense_5/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:w
"sequential/dense_5/Tensordot/ShapeShape%sequential/dense_4/Relu:activations:0*
_output_shapes
:*
T0l
*sequential/dense_5/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0n
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0l
"sequential/dense_5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_5/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0�
&sequential/dense_5/Tensordot/transpose	Transpose%sequential/dense_4/Relu:activations:0,sequential/dense_5/Tensordot/concat:output:0*-
_output_shapes
:�����������*
T0�
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������~
-sequential/dense_5/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
(sequential/dense_5/Tensordot/transpose_1	Transpose3sequential/dense_5/Tensordot/ReadVariableOp:value:06sequential/dense_5/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0}
,sequential/dense_5/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
&sequential/dense_5/Tensordot/Reshape_1Reshape,sequential/dense_5/Tensordot/transpose_1:y:05sequential/dense_5/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:0/sequential/dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������o
$sequential/dense_5/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:l
*sequential/dense_5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������[
dropout_1/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *���=j
dropout_1/dropout/ShapeShape#sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
dtype0*-
_output_shapes
:������������
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*-
_output_shapes
:������������
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*-
_output_shapes
:�����������\
dropout_1/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_1/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*-
_output_shapes
:������������
dropout_1/dropout/mulMul#sequential/dense_5/BiasAdd:output:0dropout_1/dropout/truediv:z:0*
T0*-
_output_shapes
:������������
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*-
_output_shapes
:�����������*

SrcT0
�
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:������������
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/mul_1:z:0*
T0*-
_output_shapes
:�����������~
4layer_normalization_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*,
_output_shapes
:�����������
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*,
_output_shapes
:����������*
T0�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
8layer_normalization_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:�����������
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp7^multi_head_self_attention/dense/BiasAdd/ReadVariableOp9^multi_head_self_attention/dense/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_1/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_2/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp,^sequential/dense_5/Tensordot/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*l
_input_shapes[
Y:�����������::::::::::::::::2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp2t
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_5/Tensordot/ReadVariableOp+sequential/dense_5/Tensordot/ReadVariableOp2p
6multi_head_self_attention/dense/BiasAdd/ReadVariableOp6multi_head_self_attention/dense/BiasAdd/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2x
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2t
8multi_head_self_attention/dense/Tensordot/ReadVariableOp8multi_head_self_attention/dense/Tensordot/ReadVariableOp2x
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2t
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2x
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp: : : : : : : :	 :
 : : : : : : :& "
 
_user_specified_nameinputs: 
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_66811

inputs*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity��dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*-
_output_shapes
:�����������*
Tin
2*,
_gradient_op_typePartitionedCall-66701*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_66695�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_66746*
Tout
2*-
config_proto

GPU

CPU2*0J 8*-
_output_shapes
:�����������*
Tin
2*,
_gradient_op_typePartitionedCall-66752�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*<
_input_shapes+
):�����������::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67734

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: �
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�T
batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�i
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:������������
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes	
:�*
T0x
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*-
_output_shapes
:�����������*
T0�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
�
E
)__inference_dropout_2_layer_call_fn_69778

inputs
identity�
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-67800*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_67788*
Tout
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_68900

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*&
Tin
2*'
_output_shapes
:���������*,
_gradient_op_typePartitionedCall-68013*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_68012*
Tout
2*-
config_proto

GPU

CPU2*0J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : 
�
�
3__inference_batch_normalization_layer_call_fn_69661

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-66966*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_66965*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*5
_output_shapes#
!:��������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*D
_input_shapes3
1:�������������������::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
�
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_67781

inputs
identity�Q
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *���=C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:�����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:����������*
T0b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:����������*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:����������j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�5
�
@__inference_model_layer_call_and_return_conditional_losses_67862
input_1?
;token_and_position_embedding_statefulpartitionedcall_args_1?
;token_and_position_embedding_statefulpartitionedcall_args_24
0transformer_block_statefulpartitionedcall_args_14
0transformer_block_statefulpartitionedcall_args_24
0transformer_block_statefulpartitionedcall_args_34
0transformer_block_statefulpartitionedcall_args_44
0transformer_block_statefulpartitionedcall_args_54
0transformer_block_statefulpartitionedcall_args_64
0transformer_block_statefulpartitionedcall_args_74
0transformer_block_statefulpartitionedcall_args_84
0transformer_block_statefulpartitionedcall_args_95
1transformer_block_statefulpartitionedcall_args_105
1transformer_block_statefulpartitionedcall_args_115
1transformer_block_statefulpartitionedcall_args_125
1transformer_block_statefulpartitionedcall_args_135
1transformer_block_statefulpartitionedcall_args_145
1transformer_block_statefulpartitionedcall_args_155
1transformer_block_statefulpartitionedcall_args_166
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identity��+batch_normalization/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�)transformer_block/StatefulPartitionedCall�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1;token_and_position_embedding_statefulpartitionedcall_args_1;token_and_position_embedding_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-67027*`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_67021*
Tout
2�	
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:00transformer_block_statefulpartitionedcall_args_10transformer_block_statefulpartitionedcall_args_20transformer_block_statefulpartitionedcall_args_30transformer_block_statefulpartitionedcall_args_40transformer_block_statefulpartitionedcall_args_50transformer_block_statefulpartitionedcall_args_60transformer_block_statefulpartitionedcall_args_70transformer_block_statefulpartitionedcall_args_80transformer_block_statefulpartitionedcall_args_91transformer_block_statefulpartitionedcall_args_101transformer_block_statefulpartitionedcall_args_111transformer_block_statefulpartitionedcall_args_121transformer_block_statefulpartitionedcall_args_131transformer_block_statefulpartitionedcall_args_141transformer_block_statefulpartitionedcall_args_151transformer_block_statefulpartitionedcall_args_16*
Tout
2*-
config_proto

GPU

CPU2*0J 8*-
_output_shapes
:�����������*
Tin
2*,
_gradient_op_typePartitionedCall-67610*U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_67336�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tout
2*-
config_proto

GPU

CPU2*0J 8*-
_output_shapes
:�����������*
Tin	
2*,
_gradient_op_typePartitionedCall-67737*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67711�
$global_max_pooling1d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-66986*X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_66980*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:����������*
Tin
2�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-67792*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_67781*
Tout
2*-
config_proto

GPU

CPU2*0J 8�
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:����������*
Tin
2*,
_gradient_op_typePartitionedCall-67822*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_67816*
Tout
2�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������*
Tin
2*,
_gradient_op_typePartitionedCall-67850*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_67844*
Tout
2�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall: : : : : : : : : : : : : :' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : 
�
b
)__inference_dropout_2_layer_call_fn_69773

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-67792*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_67781�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_68869

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*-
config_proto

GPU

CPU2*0J 8*&
Tin
2*'
_output_shapes
:���������*,
_gradient_op_typePartitionedCall-67942*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_67941*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : 
�
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_67788

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:����������*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�8
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69620

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�#AssignMovingAvg/Read/ReadVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp�%AssignMovingAvg_1/Read/ReadVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 ZN
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: o
moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*#
_output_shapes
:�*
T0�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:�������������������s
"moments/variance/reduction_indicesConst*
valueB"       *
dtype0*
_output_shapes
:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
squeeze_dims
 *
T0*
_output_shapes	
:�u
moments/Squeeze_1Squeezemoments/variance:output:0*
_output_shapes	
:�*
squeeze_dims
 *
T0�
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:�*
T0�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *
�#<*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: �
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:��
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:��
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 �
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *
�#<*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:��
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:��
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 T
batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:�������������������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
_output_shapes	
:�*
T0�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*D
_input_shapes3
1:�������������������::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
�#
�
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_68926
x=
9embedding_1_embedding_lookup_read_readvariableop_resource;
7embedding_embedding_lookup_read_readvariableop_resource
identity��embedding/embedding_lookup�.embedding/embedding_lookup/Read/ReadVariableOp�embedding_1/embedding_lookup�0embedding_1/embedding_lookup/Read/ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
range/startConst*
value	B : *
dtype0*
_output_shapes
: M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: w
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:����������
0embedding_1/embedding_lookup/Read/ReadVariableOpReadVariableOp9embedding_1_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
���
%embedding_1/embedding_lookup/IdentityIdentity8embedding_1/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
embedding_1/embedding_lookupResourceGather9embedding_1_embedding_lookup_read_readvariableop_resourcerange:output:01^embedding_1/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@embedding_1/embedding_lookup/Read/ReadVariableOp*
Tindices0*
dtype0*(
_output_shapes
:�����������
'embedding_1/embedding_lookup/Identity_1Identity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@embedding_1/embedding_lookup/Read/ReadVariableOp*(
_output_shapes
:�����������
'embedding_1/embedding_lookup/Identity_2Identity0embedding_1/embedding_lookup/Identity_1:output:0*
T0*(
_output_shapes
:����������[
embedding/CastCastx*

SrcT0*

DstT0*(
_output_shapes
:�����������
.embedding/embedding_lookup/Read/ReadVariableOpReadVariableOp7embedding_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
#embedding/embedding_lookup/IdentityIdentity6embedding/embedding_lookup/Read/ReadVariableOp:value:0*
_output_shapes
:	�*
T0�
embedding/embedding_lookupResourceGather7embedding_embedding_lookup_read_readvariableop_resourceembedding/Cast:y:0/^embedding/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*
dtype0*-
_output_shapes
:�����������*A
_class7
53loc:@embedding/embedding_lookup/Read/ReadVariableOp�
%embedding/embedding_lookup/Identity_1Identity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@embedding/embedding_lookup/Read/ReadVariableOp*-
_output_shapes
:������������
%embedding/embedding_lookup/Identity_2Identity.embedding/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:������������
addAddV2.embedding/embedding_lookup/Identity_2:output:00embedding_1/embedding_lookup/Identity_2:output:0*
T0*-
_output_shapes
:������������
IdentityIdentityadd:z:0^embedding/embedding_lookup/^embedding/embedding_lookup/Read/ReadVariableOp^embedding_1/embedding_lookup1^embedding_1/embedding_lookup/Read/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*/
_input_shapes
:����������::2d
0embedding_1/embedding_lookup/Read/ReadVariableOp0embedding_1/embedding_lookup/Read/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2`
.embedding/embedding_lookup/Read/ReadVariableOp.embedding/embedding_lookup/Read/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup: :! 

_user_specified_namex: 
�
�
*__inference_sequential_layer_call_fn_66819
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-66812*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_66811*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*-
_output_shapes
:������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*<
_input_shapes+
):�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : 
�5
�
@__inference_model_layer_call_and_return_conditional_losses_67941

inputs?
;token_and_position_embedding_statefulpartitionedcall_args_1?
;token_and_position_embedding_statefulpartitionedcall_args_24
0transformer_block_statefulpartitionedcall_args_14
0transformer_block_statefulpartitionedcall_args_24
0transformer_block_statefulpartitionedcall_args_34
0transformer_block_statefulpartitionedcall_args_44
0transformer_block_statefulpartitionedcall_args_54
0transformer_block_statefulpartitionedcall_args_64
0transformer_block_statefulpartitionedcall_args_74
0transformer_block_statefulpartitionedcall_args_84
0transformer_block_statefulpartitionedcall_args_95
1transformer_block_statefulpartitionedcall_args_105
1transformer_block_statefulpartitionedcall_args_115
1transformer_block_statefulpartitionedcall_args_125
1transformer_block_statefulpartitionedcall_args_135
1transformer_block_statefulpartitionedcall_args_145
1transformer_block_statefulpartitionedcall_args_155
1transformer_block_statefulpartitionedcall_args_166
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identity��+batch_normalization/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�)transformer_block/StatefulPartitionedCall�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs;token_and_position_embedding_statefulpartitionedcall_args_1;token_and_position_embedding_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-67027*`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_67021*
Tout
2�	
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:00transformer_block_statefulpartitionedcall_args_10transformer_block_statefulpartitionedcall_args_20transformer_block_statefulpartitionedcall_args_30transformer_block_statefulpartitionedcall_args_40transformer_block_statefulpartitionedcall_args_50transformer_block_statefulpartitionedcall_args_60transformer_block_statefulpartitionedcall_args_70transformer_block_statefulpartitionedcall_args_80transformer_block_statefulpartitionedcall_args_91transformer_block_statefulpartitionedcall_args_101transformer_block_statefulpartitionedcall_args_111transformer_block_statefulpartitionedcall_args_121transformer_block_statefulpartitionedcall_args_131transformer_block_statefulpartitionedcall_args_141transformer_block_statefulpartitionedcall_args_151transformer_block_statefulpartitionedcall_args_16*,
_gradient_op_typePartitionedCall-67610*U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_67336*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:������������
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-67737*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67711*
Tout
2�
$global_max_pooling1d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_66980*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-66986�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-67792*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_67781*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:����������*
Tin
2�
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-67822*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_67816*
Tout
2�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������*,
_gradient_op_typePartitionedCall-67850*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_67844*
Tout
2�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
� 
�
B__inference_dense_4_layer_call_and_return_conditional_losses_66695

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:_
Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0[
Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������k
Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��j
Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0\
Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������V
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:������������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*4
_input_shapes#
!:�����������::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
1__inference_transformer_block_layer_call_fn_69541

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16*,
_gradient_op_typePartitionedCall-67634*U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_67606*
Tout
2*-
config_proto

GPU

CPU2*0J 8*-
_output_shapes
:�����������*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*l
_input_shapes[
Y:�����������::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 :
 : : : : : : :& "
 
_user_specified_nameinputs: : : 
�	
�
B__inference_dense_6_layer_call_and_return_conditional_losses_67816

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
'__inference_dense_7_layer_call_fn_69814

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-67850*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_67844*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
� 
�
B__inference_dense_4_layer_call_and_return_conditional_losses_69997

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:_
Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0[
Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
_output_shapes
: *
T0W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������k
Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0j
Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0V
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:������������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*4
_input_shapes#
!:�����������::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
��
�9
!__inference__traced_restore_70569
file_prefix.
*assignvariableop_batch_normalization_gamma/
+assignvariableop_1_batch_normalization_beta6
2assignvariableop_2_batch_normalization_moving_mean:
6assignvariableop_3_batch_normalization_moving_variance%
!assignvariableop_4_dense_6_kernel#
assignvariableop_5_dense_6_bias%
!assignvariableop_6_dense_7_kernel#
assignvariableop_7_dense_7_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rateI
Eassignvariableop_13_token_and_position_embedding_embedding_embeddingsK
Gassignvariableop_14_token_and_position_embedding_embedding_1_embeddingsP
Lassignvariableop_15_transformer_block_multi_head_self_attention_dense_kernelN
Jassignvariableop_16_transformer_block_multi_head_self_attention_dense_biasR
Nassignvariableop_17_transformer_block_multi_head_self_attention_dense_1_kernelP
Lassignvariableop_18_transformer_block_multi_head_self_attention_dense_1_biasR
Nassignvariableop_19_transformer_block_multi_head_self_attention_dense_2_kernelP
Lassignvariableop_20_transformer_block_multi_head_self_attention_dense_2_biasR
Nassignvariableop_21_transformer_block_multi_head_self_attention_dense_3_kernelP
Lassignvariableop_22_transformer_block_multi_head_self_attention_dense_3_biasC
?assignvariableop_23_transformer_block_sequential_dense_4_kernelA
=assignvariableop_24_transformer_block_sequential_dense_4_biasC
?assignvariableop_25_transformer_block_sequential_dense_5_kernelA
=assignvariableop_26_transformer_block_sequential_dense_5_biasC
?assignvariableop_27_transformer_block_layer_normalization_gammaB
>assignvariableop_28_transformer_block_layer_normalization_betaE
Aassignvariableop_29_transformer_block_layer_normalization_1_gammaD
@assignvariableop_30_transformer_block_layer_normalization_1_beta
assignvariableop_31_total
assignvariableop_32_count8
4assignvariableop_33_adam_batch_normalization_gamma_m7
3assignvariableop_34_adam_batch_normalization_beta_m-
)assignvariableop_35_adam_dense_6_kernel_m+
'assignvariableop_36_adam_dense_6_bias_m-
)assignvariableop_37_adam_dense_7_kernel_m+
'assignvariableop_38_adam_dense_7_bias_mP
Lassignvariableop_39_adam_token_and_position_embedding_embedding_embeddings_mR
Nassignvariableop_40_adam_token_and_position_embedding_embedding_1_embeddings_mW
Sassignvariableop_41_adam_transformer_block_multi_head_self_attention_dense_kernel_mU
Qassignvariableop_42_adam_transformer_block_multi_head_self_attention_dense_bias_mY
Uassignvariableop_43_adam_transformer_block_multi_head_self_attention_dense_1_kernel_mW
Sassignvariableop_44_adam_transformer_block_multi_head_self_attention_dense_1_bias_mY
Uassignvariableop_45_adam_transformer_block_multi_head_self_attention_dense_2_kernel_mW
Sassignvariableop_46_adam_transformer_block_multi_head_self_attention_dense_2_bias_mY
Uassignvariableop_47_adam_transformer_block_multi_head_self_attention_dense_3_kernel_mW
Sassignvariableop_48_adam_transformer_block_multi_head_self_attention_dense_3_bias_mJ
Fassignvariableop_49_adam_transformer_block_sequential_dense_4_kernel_mH
Dassignvariableop_50_adam_transformer_block_sequential_dense_4_bias_mJ
Fassignvariableop_51_adam_transformer_block_sequential_dense_5_kernel_mH
Dassignvariableop_52_adam_transformer_block_sequential_dense_5_bias_mJ
Fassignvariableop_53_adam_transformer_block_layer_normalization_gamma_mI
Eassignvariableop_54_adam_transformer_block_layer_normalization_beta_mL
Hassignvariableop_55_adam_transformer_block_layer_normalization_1_gamma_mK
Gassignvariableop_56_adam_transformer_block_layer_normalization_1_beta_m8
4assignvariableop_57_adam_batch_normalization_gamma_v7
3assignvariableop_58_adam_batch_normalization_beta_v-
)assignvariableop_59_adam_dense_6_kernel_v+
'assignvariableop_60_adam_dense_6_bias_v-
)assignvariableop_61_adam_dense_7_kernel_v+
'assignvariableop_62_adam_dense_7_bias_vP
Lassignvariableop_63_adam_token_and_position_embedding_embedding_embeddings_vR
Nassignvariableop_64_adam_token_and_position_embedding_embedding_1_embeddings_vW
Sassignvariableop_65_adam_transformer_block_multi_head_self_attention_dense_kernel_vU
Qassignvariableop_66_adam_transformer_block_multi_head_self_attention_dense_bias_vY
Uassignvariableop_67_adam_transformer_block_multi_head_self_attention_dense_1_kernel_vW
Sassignvariableop_68_adam_transformer_block_multi_head_self_attention_dense_1_bias_vY
Uassignvariableop_69_adam_transformer_block_multi_head_self_attention_dense_2_kernel_vW
Sassignvariableop_70_adam_transformer_block_multi_head_self_attention_dense_2_bias_vY
Uassignvariableop_71_adam_transformer_block_multi_head_self_attention_dense_3_kernel_vW
Sassignvariableop_72_adam_transformer_block_multi_head_self_attention_dense_3_bias_vJ
Fassignvariableop_73_adam_transformer_block_sequential_dense_4_kernel_vH
Dassignvariableop_74_adam_transformer_block_sequential_dense_4_bias_vJ
Fassignvariableop_75_adam_transformer_block_sequential_dense_5_kernel_vH
Dassignvariableop_76_adam_transformer_block_sequential_dense_5_bias_vJ
Fassignvariableop_77_adam_transformer_block_layer_normalization_gamma_vI
Eassignvariableop_78_adam_transformer_block_layer_normalization_beta_vL
Hassignvariableop_79_adam_transformer_block_layer_normalization_1_gamma_vK
Gassignvariableop_80_adam_transformer_block_layer_normalization_1_beta_v
identity_82��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_9�	RestoreV2�RestoreV2_1�+
RestoreV2/tensor_namesConst"/device:CPU:0*�+
value�+B�+QB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:Q�
RestoreV2/shape_and_slicesConst"/device:CPU:0*�
value�B�QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:Q�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_7_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_7_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:|
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0*
dtype0	*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:~
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpEassignvariableop_13_token_and_position_embedding_embedding_embeddingsIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0�
AssignVariableOp_14AssignVariableOpGassignvariableop_14_token_and_position_embedding_embedding_1_embeddingsIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0�
AssignVariableOp_15AssignVariableOpLassignvariableop_15_transformer_block_multi_head_self_attention_dense_kernelIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpJassignvariableop_16_transformer_block_multi_head_self_attention_dense_biasIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0�
AssignVariableOp_17AssignVariableOpNassignvariableop_17_transformer_block_multi_head_self_attention_dense_1_kernelIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0�
AssignVariableOp_18AssignVariableOpLassignvariableop_18_transformer_block_multi_head_self_attention_dense_1_biasIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpNassignvariableop_19_transformer_block_multi_head_self_attention_dense_2_kernelIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpLassignvariableop_20_transformer_block_multi_head_self_attention_dense_2_biasIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpNassignvariableop_21_transformer_block_multi_head_self_attention_dense_3_kernelIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpLassignvariableop_22_transformer_block_multi_head_self_attention_dense_3_biasIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp?assignvariableop_23_transformer_block_sequential_dense_4_kernelIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp=assignvariableop_24_transformer_block_sequential_dense_4_biasIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp?assignvariableop_25_transformer_block_sequential_dense_5_kernelIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp=assignvariableop_26_transformer_block_sequential_dense_5_biasIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp?assignvariableop_27_transformer_block_layer_normalization_gammaIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp>assignvariableop_28_transformer_block_layer_normalization_betaIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0�
AssignVariableOp_29AssignVariableOpAassignvariableop_29_transformer_block_layer_normalization_1_gammaIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp@assignvariableop_30_transformer_block_layer_normalization_1_betaIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:{
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:{
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_batch_normalization_gamma_mIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_batch_normalization_beta_mIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_6_kernel_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_6_bias_mIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_7_kernel_mIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
_output_shapes
:*
T0�
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_7_bias_mIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpLassignvariableop_39_adam_token_and_position_embedding_embedding_embeddings_mIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpNassignvariableop_40_adam_token_and_position_embedding_embedding_1_embeddings_mIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpSassignvariableop_41_adam_transformer_block_multi_head_self_attention_dense_kernel_mIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpQassignvariableop_42_adam_transformer_block_multi_head_self_attention_dense_bias_mIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpUassignvariableop_43_adam_transformer_block_multi_head_self_attention_dense_1_kernel_mIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
_output_shapes
:*
T0�
AssignVariableOp_44AssignVariableOpSassignvariableop_44_adam_transformer_block_multi_head_self_attention_dense_1_bias_mIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T0�
AssignVariableOp_45AssignVariableOpUassignvariableop_45_adam_transformer_block_multi_head_self_attention_dense_2_kernel_mIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
_output_shapes
:*
T0�
AssignVariableOp_46AssignVariableOpSassignvariableop_46_adam_transformer_block_multi_head_self_attention_dense_2_bias_mIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpUassignvariableop_47_adam_transformer_block_multi_head_self_attention_dense_3_kernel_mIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpSassignvariableop_48_adam_transformer_block_multi_head_self_attention_dense_3_bias_mIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpFassignvariableop_49_adam_transformer_block_sequential_dense_4_kernel_mIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
_output_shapes
:*
T0�
AssignVariableOp_50AssignVariableOpDassignvariableop_50_adam_transformer_block_sequential_dense_4_bias_mIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpFassignvariableop_51_adam_transformer_block_sequential_dense_5_kernel_mIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpDassignvariableop_52_adam_transformer_block_sequential_dense_5_bias_mIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
_output_shapes
:*
T0�
AssignVariableOp_53AssignVariableOpFassignvariableop_53_adam_transformer_block_layer_normalization_gamma_mIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
_output_shapes
:*
T0�
AssignVariableOp_54AssignVariableOpEassignvariableop_54_adam_transformer_block_layer_normalization_beta_mIdentity_54:output:0*
dtype0*
_output_shapes
 P
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpHassignvariableop_55_adam_transformer_block_layer_normalization_1_gamma_mIdentity_55:output:0*
dtype0*
_output_shapes
 P
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpGassignvariableop_56_adam_transformer_block_layer_normalization_1_beta_mIdentity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp4assignvariableop_57_adam_batch_normalization_gamma_vIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
_output_shapes
:*
T0�
AssignVariableOp_58AssignVariableOp3assignvariableop_58_adam_batch_normalization_beta_vIdentity_58:output:0*
dtype0*
_output_shapes
 P
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_6_kernel_vIdentity_59:output:0*
dtype0*
_output_shapes
 P
Identity_60IdentityRestoreV2:tensors:60*
_output_shapes
:*
T0�
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_6_bias_vIdentity_60:output:0*
dtype0*
_output_shapes
 P
Identity_61IdentityRestoreV2:tensors:61*
_output_shapes
:*
T0�
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_7_kernel_vIdentity_61:output:0*
dtype0*
_output_shapes
 P
Identity_62IdentityRestoreV2:tensors:62*
_output_shapes
:*
T0�
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_7_bias_vIdentity_62:output:0*
dtype0*
_output_shapes
 P
Identity_63IdentityRestoreV2:tensors:63*
_output_shapes
:*
T0�
AssignVariableOp_63AssignVariableOpLassignvariableop_63_adam_token_and_position_embedding_embedding_embeddings_vIdentity_63:output:0*
dtype0*
_output_shapes
 P
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpNassignvariableop_64_adam_token_and_position_embedding_embedding_1_embeddings_vIdentity_64:output:0*
dtype0*
_output_shapes
 P
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpSassignvariableop_65_adam_transformer_block_multi_head_self_attention_dense_kernel_vIdentity_65:output:0*
dtype0*
_output_shapes
 P
Identity_66IdentityRestoreV2:tensors:66*
_output_shapes
:*
T0�
AssignVariableOp_66AssignVariableOpQassignvariableop_66_adam_transformer_block_multi_head_self_attention_dense_bias_vIdentity_66:output:0*
dtype0*
_output_shapes
 P
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpUassignvariableop_67_adam_transformer_block_multi_head_self_attention_dense_1_kernel_vIdentity_67:output:0*
dtype0*
_output_shapes
 P
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpSassignvariableop_68_adam_transformer_block_multi_head_self_attention_dense_1_bias_vIdentity_68:output:0*
dtype0*
_output_shapes
 P
Identity_69IdentityRestoreV2:tensors:69*
_output_shapes
:*
T0�
AssignVariableOp_69AssignVariableOpUassignvariableop_69_adam_transformer_block_multi_head_self_attention_dense_2_kernel_vIdentity_69:output:0*
dtype0*
_output_shapes
 P
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOpSassignvariableop_70_adam_transformer_block_multi_head_self_attention_dense_2_bias_vIdentity_70:output:0*
dtype0*
_output_shapes
 P
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOpUassignvariableop_71_adam_transformer_block_multi_head_self_attention_dense_3_kernel_vIdentity_71:output:0*
dtype0*
_output_shapes
 P
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOpSassignvariableop_72_adam_transformer_block_multi_head_self_attention_dense_3_bias_vIdentity_72:output:0*
dtype0*
_output_shapes
 P
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOpFassignvariableop_73_adam_transformer_block_sequential_dense_4_kernel_vIdentity_73:output:0*
dtype0*
_output_shapes
 P
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpDassignvariableop_74_adam_transformer_block_sequential_dense_4_bias_vIdentity_74:output:0*
dtype0*
_output_shapes
 P
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpFassignvariableop_75_adam_transformer_block_sequential_dense_5_kernel_vIdentity_75:output:0*
dtype0*
_output_shapes
 P
Identity_76IdentityRestoreV2:tensors:76*
_output_shapes
:*
T0�
AssignVariableOp_76AssignVariableOpDassignvariableop_76_adam_transformer_block_sequential_dense_5_bias_vIdentity_76:output:0*
dtype0*
_output_shapes
 P
Identity_77IdentityRestoreV2:tensors:77*
_output_shapes
:*
T0�
AssignVariableOp_77AssignVariableOpFassignvariableop_77_adam_transformer_block_layer_normalization_gamma_vIdentity_77:output:0*
dtype0*
_output_shapes
 P
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpEassignvariableop_78_adam_transformer_block_layer_normalization_beta_vIdentity_78:output:0*
dtype0*
_output_shapes
 P
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOpHassignvariableop_79_adam_transformer_block_layer_normalization_1_gamma_vIdentity_79:output:0*
dtype0*
_output_shapes
 P
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpGassignvariableop_80_adam_transformer_block_layer_normalization_1_beta_vIdentity_80:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_81Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_82IdentityIdentity_81:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_82Identity_82:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_59AssignVariableOp_592*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_11:
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :M :N :O :P :Q :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 
�
�
<__inference_token_and_position_embedding_layer_call_fn_68933
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-67027*`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_67021*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex: : 
�
�
'__inference_dense_6_layer_call_fn_69796

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-67822*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_67816*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:�����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
'__inference_dense_5_layer_call_fn_70045

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66752*K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_66746*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*4
_input_shapes#
!:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
��
�
 __inference__wrapped_model_66654
input_1`
\model_token_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resource^
Zmodel_token_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resource]
Ymodel_transformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resource[
Wmodel_transformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resource_
[model_transformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resource]
Ymodel_transformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resource_
[model_transformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resource]
Ymodel_transformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resource_
[model_transformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resource]
Ymodel_transformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resourceU
Qmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resourceQ
Mmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resourceP
Lmodel_transformer_block_sequential_dense_4_tensordot_readvariableop_resourceN
Jmodel_transformer_block_sequential_dense_4_biasadd_readvariableop_resourceP
Lmodel_transformer_block_sequential_dense_5_tensordot_readvariableop_resourceN
Jmodel_transformer_block_sequential_dense_5_biasadd_readvariableop_resourceW
Smodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resourceS
Omodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource?
;model_batch_normalization_batchnorm_readvariableop_resourceC
?model_batch_normalization_batchnorm_mul_readvariableop_resourceA
=model_batch_normalization_batchnorm_readvariableop_1_resourceA
=model_batch_normalization_batchnorm_readvariableop_2_resource0
,model_dense_6_matmul_readvariableop_resource1
-model_dense_6_biasadd_readvariableop_resource0
,model_dense_7_matmul_readvariableop_resource1
-model_dense_7_biasadd_readvariableop_resource
identity��2model/batch_normalization/batchnorm/ReadVariableOp�4model/batch_normalization/batchnorm/ReadVariableOp_1�4model/batch_normalization/batchnorm/ReadVariableOp_2�6model/batch_normalization/batchnorm/mul/ReadVariableOp�$model/dense_6/BiasAdd/ReadVariableOp�#model/dense_6/MatMul/ReadVariableOp�$model/dense_7/BiasAdd/ReadVariableOp�#model/dense_7/MatMul/ReadVariableOp�=model/token_and_position_embedding/embedding/embedding_lookup�Qmodel/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp�?model/token_and_position_embedding/embedding_1/embedding_lookup�Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp�Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp�Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp�Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp�Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp�Nmodel/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp�Pmodel/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp�Pmodel/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp�Rmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp�Pmodel/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp�Rmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp�Pmodel/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp�Rmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp�Amodel/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp�Cmodel/transformer_block/sequential/dense_4/Tensordot/ReadVariableOp�Amodel/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp�Cmodel/transformer_block/sequential/dense_5/Tensordot/ReadVariableOp_
(model/token_and_position_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:�
6model/token_and_position_embedding/strided_slice/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:�
8model/token_and_position_embedding/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:�
8model/token_and_position_embedding/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
0model/token_and_position_embedding/strided_sliceStridedSlice1model/token_and_position_embedding/Shape:output:0?model/token_and_position_embedding/strided_slice/stack:output:0Amodel/token_and_position_embedding/strided_slice/stack_1:output:0Amodel/token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: p
.model/token_and_position_embedding/range/startConst*
dtype0*
_output_shapes
: *
value	B : p
.model/token_and_position_embedding/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: �
(model/token_and_position_embedding/rangeRange7model/token_and_position_embedding/range/start:output:09model/token_and_position_embedding/strided_slice:output:07model/token_and_position_embedding/range/delta:output:0*#
_output_shapes
:����������
Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOpReadVariableOp\model_token_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
���
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentity[model/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
���
?model/token_and_position_embedding/embedding_1/embedding_lookupResourceGather\model_token_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resource1model/token_and_position_embedding/range:output:0T^model/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:����������*f
_class\
ZXloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp*
Tindices0�
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityHmodel/token_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*f
_class\
ZXloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp*(
_output_shapes
:�����������
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_2IdentitySmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*(
_output_shapes
:����������*
T0�
1model/token_and_position_embedding/embedding/CastCastinput_1*

SrcT0*

DstT0*(
_output_shapes
:�����������
Qmodel/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOpReadVariableOpZmodel_token_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
Fmodel/token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityYmodel/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
=model/token_and_position_embedding/embedding/embedding_lookupResourceGatherZmodel_token_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resource5model/token_and_position_embedding/embedding/Cast:y:0R^model/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*d
_classZ
XVloc:@model/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp*
Tindices0*
dtype0*-
_output_shapes
:������������
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityFmodel/token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*-
_output_shapes
:�����������*
T0*d
_classZ
XVloc:@model/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp�
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_2IdentityQmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:������������
&model/token_and_position_embedding/addAddV2Qmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_2:output:0Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_2:output:0*
T0*-
_output_shapes
:������������
7model/transformer_block/multi_head_self_attention/ShapeShape*model/token_and_position_embedding/add:z:0*
T0*
_output_shapes
:�
Emodel/transformer_block/multi_head_self_attention/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:�
Gmodel/transformer_block/multi_head_self_attention/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:�
Gmodel/transformer_block/multi_head_self_attention/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
?model/transformer_block/multi_head_self_attention/strided_sliceStridedSlice@model/transformer_block/multi_head_self_attention/Shape:output:0Nmodel/transformer_block/multi_head_self_attention/strided_slice/stack:output:0Pmodel/transformer_block/multi_head_self_attention/strided_slice/stack_1:output:0Pmodel/transformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0�
Pmodel/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Fmodel/transformer_block/multi_head_self_attention/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
Fmodel/transformer_block/multi_head_self_attention/dense/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       �
Gmodel/transformer_block/multi_head_self_attention/dense/Tensordot/ShapeShape*model/token_and_position_embedding/add:z:0*
T0*
_output_shapes
:�
Omodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Jmodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Pmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Omodel/transformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Xmodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0�
Qmodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Lmodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Pmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Omodel/transformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Zmodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Gmodel/transformer_block/multi_head_self_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Fmodel/transformer_block/multi_head_self_attention/dense/Tensordot/ProdProdSmodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Pmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Const:output:0*
_output_shapes
: *
T0�
Imodel/transformer_block/multi_head_self_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Hmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Prod_1ProdUmodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Rmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Mmodel/transformer_block/multi_head_self_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Hmodel/transformer_block/multi_head_self_attention/dense/Tensordot/concatConcatV2Omodel/transformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Omodel/transformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Vmodel/transformer_block/multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Gmodel/transformer_block/multi_head_self_attention/dense/Tensordot/stackPackOmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Prod:output:0Qmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Kmodel/transformer_block/multi_head_self_attention/dense/Tensordot/transpose	Transpose*model/token_and_position_embedding/add:z:0Qmodel/transformer_block/multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Imodel/transformer_block/multi_head_self_attention/dense/Tensordot/ReshapeReshapeOmodel/transformer_block/multi_head_self_attention/dense/Tensordot/transpose:y:0Pmodel/transformer_block/multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Rmodel/transformer_block/multi_head_self_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Mmodel/transformer_block/multi_head_self_attention/dense/Tensordot/transpose_1	TransposeXmodel/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0[model/transformer_block/multi_head_self_attention/dense/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Qmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Kmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1ReshapeQmodel/transformer_block/multi_head_self_attention/dense/Tensordot/transpose_1:y:0Zmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
Hmodel/transformer_block/multi_head_self_attention/dense/Tensordot/MatMulMatMulRmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Reshape:output:0Tmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Imodel/transformer_block/multi_head_self_attention/dense/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Omodel/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Jmodel/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Smodel/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Rmodel/transformer_block/multi_head_self_attention/dense/Tensordot/Const_2:output:0Xmodel/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
Amodel/transformer_block/multi_head_self_attention/dense/TensordotReshapeRmodel/transformer_block/multi_head_self_attention/dense/Tensordot/MatMul:product:0Smodel/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
Nmodel/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpWmodel_transformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
?model/transformer_block/multi_head_self_attention/dense/BiasAddBiasAddJmodel/transformer_block/multi_head_self_attention/dense/Tensordot:output:0Vmodel/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
Rmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp[model_transformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Hmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
Hmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       �
Imodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ShapeShape*model/token_and_position_embedding/add:z:0*
_output_shapes
:*
T0�
Qmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Lmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Rmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Qmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Zmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Smodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Nmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Rmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Qmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0\model/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Imodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: �
Hmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ProdProdUmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Rmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const:output:0*
_output_shapes
: *
T0�
Kmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Jmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1ProdWmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Tmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
_output_shapes
: *
T0�
Omodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Jmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/concatConcatV2Qmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Qmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Xmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Imodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/stackPackQmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Prod:output:0Smodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Mmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/transpose	Transpose*model/token_and_position_embedding/add:z:0Smodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Kmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeQmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/transpose:y:0Rmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0�
Tmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Omodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1	TransposeZmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0]model/transformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Smodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Mmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1ReshapeSmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1:y:0\model/transformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Jmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulMatMulTmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Vmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0�
Kmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Qmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Lmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Umodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Tmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Zmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
Cmodel/transformer_block/multi_head_self_attention/dense_1/TensordotReshapeTmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Umodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
Pmodel/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
Amodel/transformer_block/multi_head_self_attention/dense_1/BiasAddBiasAddLmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot:output:0Xmodel/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
Rmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOp[model_transformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Hmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:�
Hmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       �
Imodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ShapeShape*model/token_and_position_embedding/add:z:0*
T0*
_output_shapes
:�
Qmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Lmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Rmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Qmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Zmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Smodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Nmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Rmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Qmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0\model/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0�
Imodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Hmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ProdProdUmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Rmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const:output:0*
_output_shapes
: *
T0�
Kmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Jmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1ProdWmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Tmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
_output_shapes
: *
T0�
Omodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Jmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/concatConcatV2Qmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Qmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Xmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Imodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/stackPackQmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Prod:output:0Smodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Mmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/transpose	Transpose*model/token_and_position_embedding/add:z:0Smodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Kmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeQmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/transpose:y:0Rmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Omodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1	TransposeZmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0]model/transformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Smodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Mmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1ReshapeSmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1:y:0\model/transformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Jmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulMatMulTmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Vmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Kmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Qmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Lmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Umodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Tmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Zmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
Cmodel/transformer_block/multi_head_self_attention/dense_2/TensordotReshapeTmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Umodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
Pmodel/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
Amodel/transformer_block/multi_head_self_attention/dense_2/BiasAddBiasAddLmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot:output:0Xmodel/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
Amodel/transformer_block/multi_head_self_attention/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: �
Amodel/transformer_block/multi_head_self_attention/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :�
Amodel/transformer_block/multi_head_self_attention/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
value	B :`�
?model/transformer_block/multi_head_self_attention/Reshape/shapePackHmodel/transformer_block/multi_head_self_attention/strided_slice:output:0Jmodel/transformer_block/multi_head_self_attention/Reshape/shape/1:output:0Jmodel/transformer_block/multi_head_self_attention/Reshape/shape/2:output:0Jmodel/transformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:�
9model/transformer_block/multi_head_self_attention/ReshapeReshapeHmodel/transformer_block/multi_head_self_attention/dense/BiasAdd:output:0Hmodel/transformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
@model/transformer_block/multi_head_self_attention/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
;model/transformer_block/multi_head_self_attention/transpose	TransposeBmodel/transformer_block/multi_head_self_attention/Reshape:output:0Imodel/transformer_block/multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
Cmodel/transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: �
Cmodel/transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :�
Cmodel/transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
dtype0*
_output_shapes
: *
value	B :`�
Amodel/transformer_block/multi_head_self_attention/Reshape_1/shapePackHmodel/transformer_block/multi_head_self_attention/strided_slice:output:0Lmodel/transformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Lmodel/transformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Lmodel/transformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
T0*
N*
_output_shapes
:�
;model/transformer_block/multi_head_self_attention/Reshape_1ReshapeJmodel/transformer_block/multi_head_self_attention/dense_1/BiasAdd:output:0Jmodel/transformer_block/multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
Bmodel/transformer_block/multi_head_self_attention/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
=model/transformer_block/multi_head_self_attention/transpose_1	TransposeDmodel/transformer_block/multi_head_self_attention/Reshape_1:output:0Kmodel/transformer_block/multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
Cmodel/transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
����������
Cmodel/transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: �
Cmodel/transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
Amodel/transformer_block/multi_head_self_attention/Reshape_2/shapePackHmodel/transformer_block/multi_head_self_attention/strided_slice:output:0Lmodel/transformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Lmodel/transformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Lmodel/transformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
T0*
N*
_output_shapes
:�
;model/transformer_block/multi_head_self_attention/Reshape_2ReshapeJmodel/transformer_block/multi_head_self_attention/dense_2/BiasAdd:output:0Jmodel/transformer_block/multi_head_self_attention/Reshape_2/shape:output:0*8
_output_shapes&
$:"������������������`*
T0�
Bmodel/transformer_block/multi_head_self_attention/transpose_2/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
=model/transformer_block/multi_head_self_attention/transpose_2	TransposeDmodel/transformer_block/multi_head_self_attention/Reshape_2:output:0Kmodel/transformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
8model/transformer_block/multi_head_self_attention/MatMulBatchMatMulV2?model/transformer_block/multi_head_self_attention/transpose:y:0Amodel/transformer_block/multi_head_self_attention/transpose_1:y:0*
adj_y(*
T0*A
_output_shapes/
-:+����������������������������
9model/transformer_block/multi_head_self_attention/Shape_1ShapeAmodel/transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:�
Gmodel/transformer_block/multi_head_self_attention/strided_slice_1/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:�
Imodel/transformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:�
Imodel/transformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
Amodel/transformer_block/multi_head_self_attention/strided_slice_1StridedSliceBmodel/transformer_block/multi_head_self_attention/Shape_1:output:0Pmodel/transformer_block/multi_head_self_attention/strided_slice_1/stack:output:0Rmodel/transformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0Rmodel/transformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0�
6model/transformer_block/multi_head_self_attention/CastCastJmodel/transformer_block/multi_head_self_attention/strided_slice_1:output:0*

SrcT0*

DstT0*
_output_shapes
: �
6model/transformer_block/multi_head_self_attention/SqrtSqrt:model/transformer_block/multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: �
9model/transformer_block/multi_head_self_attention/truedivRealDivAmodel/transformer_block/multi_head_self_attention/MatMul:output:0:model/transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
9model/transformer_block/multi_head_self_attention/SoftmaxSoftmax=model/transformer_block/multi_head_self_attention/truediv:z:0*A
_output_shapes/
-:+���������������������������*
T0�
:model/transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2Cmodel/transformer_block/multi_head_self_attention/Softmax:softmax:0Amodel/transformer_block/multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"������������������`�
Bmodel/transformer_block/multi_head_self_attention/transpose_3/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
=model/transformer_block/multi_head_self_attention/transpose_3	TransposeCmodel/transformer_block/multi_head_self_attention/MatMul_1:output:0Kmodel/transformer_block/multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
Cmodel/transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: �
Cmodel/transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: �
Amodel/transformer_block/multi_head_self_attention/Reshape_3/shapePackHmodel/transformer_block/multi_head_self_attention/strided_slice:output:0Lmodel/transformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Lmodel/transformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
T0*
N*
_output_shapes
:�
;model/transformer_block/multi_head_self_attention/Reshape_3ReshapeAmodel/transformer_block/multi_head_self_attention/transpose_3:y:0Jmodel/transformer_block/multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
Rmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOp[model_transformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Hmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:�
Hmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Imodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ShapeShapeDmodel/transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:�
Qmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Lmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Rmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Qmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Zmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Smodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Nmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Rmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Qmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0\model/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0�
Imodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Hmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ProdProdUmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Rmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Kmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Jmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1ProdWmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Tmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
_output_shapes
: *
T0�
Omodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Jmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/concatConcatV2Qmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Qmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Xmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Imodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/stackPackQmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Prod:output:0Smodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0�
Mmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/transpose	TransposeDmodel/transformer_block/multi_head_self_attention/Reshape_3:output:0Smodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Kmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeQmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/transpose:y:0Rmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Omodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1	TransposeZmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0]model/transformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Smodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Mmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1ReshapeSmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1:y:0\model/transformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
Jmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulMatMulTmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Vmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0�
Kmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Qmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Lmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Umodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Tmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Zmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
Cmodel/transformer_block/multi_head_self_attention/dense_3/TensordotReshapeTmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Umodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Pmodel/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpYmodel_transformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
Amodel/transformer_block/multi_head_self_attention/dense_3/BiasAddBiasAddLmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot:output:0Xmodel/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
(model/transformer_block/dropout/IdentityIdentityJmodel/transformer_block/multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
model/transformer_block/addAddV2*model/token_and_position_embedding/add:z:01model/transformer_block/dropout/Identity:output:0*
T0*-
_output_shapes
:������������
Jmodel/transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
8model/transformer_block/layer_normalization/moments/meanMeanmodel/transformer_block/add:z:0Smodel/transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
@model/transformer_block/layer_normalization/moments/StopGradientStopGradientAmodel/transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:�����������
Emodel/transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencemodel/transformer_block/add:z:0Imodel/transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
Nmodel/transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
<model/transformer_block/layer_normalization/moments/varianceMeanImodel/transformer_block/layer_normalization/moments/SquaredDifference:z:0Wmodel/transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
;model/transformer_block/layer_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5�
9model/transformer_block/layer_normalization/batchnorm/addAddV2Emodel/transformer_block/layer_normalization/moments/variance:output:0Dmodel/transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
;model/transformer_block/layer_normalization/batchnorm/RsqrtRsqrt=model/transformer_block/layer_normalization/batchnorm/add:z:0*,
_output_shapes
:����������*
T0�
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
9model/transformer_block/layer_normalization/batchnorm/mulMul?model/transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Pmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
;model/transformer_block/layer_normalization/batchnorm/mul_1Mulmodel/transformer_block/add:z:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
;model/transformer_block/layer_normalization/batchnorm/mul_2MulAmodel/transformer_block/layer_normalization/moments/mean:output:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
9model/transformer_block/layer_normalization/batchnorm/subSubLmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0?model/transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
;model/transformer_block/layer_normalization/batchnorm/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/mul_1:z:0=model/transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
Cmodel/transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpLmodel_transformer_block_sequential_dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
9model/transformer_block/sequential/dense_4/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:�
9model/transformer_block/sequential/dense_4/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
:model/transformer_block/sequential/dense_4/Tensordot/ShapeShape?model/transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
=model/transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2Cmodel/transformer_block/sequential/dense_4/Tensordot/Shape:output:0Bmodel/transformer_block/sequential/dense_4/Tensordot/free:output:0Kmodel/transformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0�
Dmodel/transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
?model/transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2Cmodel/transformer_block/sequential/dense_4/Tensordot/Shape:output:0Bmodel/transformer_block/sequential/dense_4/Tensordot/axes:output:0Mmodel/transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0�
:model/transformer_block/sequential/dense_4/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
9model/transformer_block/sequential/dense_4/Tensordot/ProdProdFmodel/transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Cmodel/transformer_block/sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: �
<model/transformer_block/sequential/dense_4/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: �
;model/transformer_block/sequential/dense_4/Tensordot/Prod_1ProdHmodel/transformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0Emodel/transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
@model/transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
;model/transformer_block/sequential/dense_4/Tensordot/concatConcatV2Bmodel/transformer_block/sequential/dense_4/Tensordot/free:output:0Bmodel/transformer_block/sequential/dense_4/Tensordot/axes:output:0Imodel/transformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
:model/transformer_block/sequential/dense_4/Tensordot/stackPackBmodel/transformer_block/sequential/dense_4/Tensordot/Prod:output:0Dmodel/transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
>model/transformer_block/sequential/dense_4/Tensordot/transpose	Transpose?model/transformer_block/layer_normalization/batchnorm/add_1:z:0Dmodel/transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
<model/transformer_block/sequential/dense_4/Tensordot/ReshapeReshapeBmodel/transformer_block/sequential/dense_4/Tensordot/transpose:y:0Cmodel/transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Emodel/transformer_block/sequential/dense_4/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
@model/transformer_block/sequential/dense_4/Tensordot/transpose_1	TransposeKmodel/transformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0Nmodel/transformer_block/sequential/dense_4/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Dmodel/transformer_block/sequential/dense_4/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
>model/transformer_block/sequential/dense_4/Tensordot/Reshape_1ReshapeDmodel/transformer_block/sequential/dense_4/Tensordot/transpose_1:y:0Mmodel/transformer_block/sequential/dense_4/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
;model/transformer_block/sequential/dense_4/Tensordot/MatMulMatMulEmodel/transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Gmodel/transformer_block/sequential/dense_4/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0�
<model/transformer_block/sequential/dense_4/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
=model/transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2Fmodel/transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Emodel/transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Kmodel/transformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
4model/transformer_block/sequential/dense_4/TensordotReshapeEmodel/transformer_block/sequential/dense_4/Tensordot/MatMul:product:0Fmodel/transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
Amodel/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
2model/transformer_block/sequential/dense_4/BiasAddBiasAdd=model/transformer_block/sequential/dense_4/Tensordot:output:0Imodel/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
/model/transformer_block/sequential/dense_4/ReluRelu;model/transformer_block/sequential/dense_4/BiasAdd:output:0*-
_output_shapes
:�����������*
T0�
Cmodel/transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpLmodel_transformer_block_sequential_dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
9model/transformer_block/sequential/dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
9model/transformer_block/sequential/dense_5/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
:model/transformer_block/sequential/dense_5/Tensordot/ShapeShape=model/transformer_block/sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : �
=model/transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2Cmodel/transformer_block/sequential/dense_5/Tensordot/Shape:output:0Bmodel/transformer_block/sequential/dense_5/Tensordot/free:output:0Kmodel/transformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Dmodel/transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
?model/transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2Cmodel/transformer_block/sequential/dense_5/Tensordot/Shape:output:0Bmodel/transformer_block/sequential/dense_5/Tensordot/axes:output:0Mmodel/transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
:model/transformer_block/sequential/dense_5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
9model/transformer_block/sequential/dense_5/Tensordot/ProdProdFmodel/transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Cmodel/transformer_block/sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: �
<model/transformer_block/sequential/dense_5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
;model/transformer_block/sequential/dense_5/Tensordot/Prod_1ProdHmodel/transformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0Emodel/transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
@model/transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
;model/transformer_block/sequential/dense_5/Tensordot/concatConcatV2Bmodel/transformer_block/sequential/dense_5/Tensordot/free:output:0Bmodel/transformer_block/sequential/dense_5/Tensordot/axes:output:0Imodel/transformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
:model/transformer_block/sequential/dense_5/Tensordot/stackPackBmodel/transformer_block/sequential/dense_5/Tensordot/Prod:output:0Dmodel/transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0�
>model/transformer_block/sequential/dense_5/Tensordot/transpose	Transpose=model/transformer_block/sequential/dense_4/Relu:activations:0Dmodel/transformer_block/sequential/dense_5/Tensordot/concat:output:0*-
_output_shapes
:�����������*
T0�
<model/transformer_block/sequential/dense_5/Tensordot/ReshapeReshapeBmodel/transformer_block/sequential/dense_5/Tensordot/transpose:y:0Cmodel/transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Emodel/transformer_block/sequential/dense_5/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
@model/transformer_block/sequential/dense_5/Tensordot/transpose_1	TransposeKmodel/transformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0Nmodel/transformer_block/sequential/dense_5/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Dmodel/transformer_block/sequential/dense_5/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
>model/transformer_block/sequential/dense_5/Tensordot/Reshape_1ReshapeDmodel/transformer_block/sequential/dense_5/Tensordot/transpose_1:y:0Mmodel/transformer_block/sequential/dense_5/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
;model/transformer_block/sequential/dense_5/Tensordot/MatMulMatMulEmodel/transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Gmodel/transformer_block/sequential/dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
<model/transformer_block/sequential/dense_5/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Bmodel/transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
=model/transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2Fmodel/transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Emodel/transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Kmodel/transformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
4model/transformer_block/sequential/dense_5/TensordotReshapeEmodel/transformer_block/sequential/dense_5/Tensordot/MatMul:product:0Fmodel/transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
Amodel/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpJmodel_transformer_block_sequential_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
2model/transformer_block/sequential/dense_5/BiasAddBiasAdd=model/transformer_block/sequential/dense_5/Tensordot:output:0Imodel/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
*model/transformer_block/dropout_1/IdentityIdentity;model/transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*-
_output_shapes
:������������
model/transformer_block/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/add_1:z:03model/transformer_block/dropout_1/Identity:output:0*
T0*-
_output_shapes
:������������
Lmodel/transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
:model/transformer_block/layer_normalization_1/moments/meanMean!model/transformer_block/add_1:z:0Umodel/transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
Bmodel/transformer_block/layer_normalization_1/moments/StopGradientStopGradientCmodel/transformer_block/layer_normalization_1/moments/mean:output:0*,
_output_shapes
:����������*
T0�
Gmodel/transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!model/transformer_block/add_1:z:0Kmodel/transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
Pmodel/transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
>model/transformer_block/layer_normalization_1/moments/varianceMeanKmodel/transformer_block/layer_normalization_1/moments/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
=model/transformer_block/layer_normalization_1/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
;model/transformer_block/layer_normalization_1/batchnorm/addAddV2Gmodel/transformer_block/layer_normalization_1/moments/variance:output:0Fmodel/transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
=model/transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt?model/transformer_block/layer_normalization_1/batchnorm/add:z:0*,
_output_shapes
:����������*
T0�
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
;model/transformer_block/layer_normalization_1/batchnorm/mulMulAmodel/transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Rmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
=model/transformer_block/layer_normalization_1/batchnorm/mul_1Mul!model/transformer_block/add_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
=model/transformer_block/layer_normalization_1/batchnorm/mul_2MulCmodel/transformer_block/layer_normalization_1/moments/mean:output:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
;model/transformer_block/layer_normalization_1/batchnorm/subSubNmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
=model/transformer_block/layer_normalization_1/batchnorm/add_1AddV2Amodel/transformer_block/layer_normalization_1/batchnorm/mul_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:�����������h
&model/batch_normalization/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z h
&model/batch_normalization/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: �
$model/batch_normalization/LogicalAnd
LogicalAnd/model/batch_normalization/LogicalAnd/x:output:0/model/batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: �
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�n
)model/batch_normalization/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: �
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
_output_shapes	
:�*
T0�
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
_output_shapes	
:�*
T0�
)model/batch_normalization/batchnorm/mul_1MulAmodel/transformer_block/layer_normalization_1/batchnorm/add_1:z:0+model/batch_normalization/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
_output_shapes	
:�*
T0�
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*-
_output_shapes
:�����������r
0model/global_max_pooling1d/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: �
model/global_max_pooling1d/MaxMax-model/batch_normalization/batchnorm/add_1:z:09model/global_max_pooling1d/Max/reduction_indices:output:0*(
_output_shapes
:����������*
T0�
model/dropout_2/IdentityIdentity'model/global_max_pooling1d/Max:output:0*
T0*(
_output_shapes
:�����������
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
model/dense_6/MatMulMatMul!model/dropout_2/Identity:output:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_6/ReluRelumodel/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
#model/dense_7/MatMul/ReadVariableOpReadVariableOp,model_dense_7_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	��
model/dense_7/MatMulMatMul model/dense_6/Relu:activations:0+model/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_7/BiasAdd/ReadVariableOpReadVariableOp-model_dense_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model/dense_7/BiasAddBiasAddmodel/dense_7/MatMul:product:0,model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/dense_7/SigmoidSigmoidmodel/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitymodel/dense_7/Sigmoid:y:03^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp$^model/dense_6/MatMul/ReadVariableOp%^model/dense_7/BiasAdd/ReadVariableOp$^model/dense_7/MatMul/ReadVariableOp>^model/token_and_position_embedding/embedding/embedding_lookupR^model/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp@^model/token_and_position_embedding/embedding_1/embedding_lookupT^model/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOpE^model/transformer_block/layer_normalization/batchnorm/ReadVariableOpI^model/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpG^model/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpK^model/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpO^model/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpQ^model/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpQ^model/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpS^model/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpQ^model/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpS^model/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpQ^model/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpS^model/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpB^model/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpD^model/transformer_block/sequential/dense_4/Tensordot/ReadVariableOpB^model/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpD^model/transformer_block/sequential/dense_5/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::2l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22�
Nmodel/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpNmodel/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp2J
#model/dense_7/MatMul/ReadVariableOp#model/dense_7/MatMul/ReadVariableOp2�
Amodel/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpAmodel/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp2�
Rmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpRmodel/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2~
=model/token_and_position_embedding/embedding/embedding_lookup=model/token_and_position_embedding/embedding/embedding_lookup2�
Pmodel/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpPmodel/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp2�
?model/token_and_position_embedding/embedding_1/embedding_lookup?model/token_and_position_embedding/embedding_1/embedding_lookup2�
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpFmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2�
Pmodel/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpPmodel/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2�
Rmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpRmodel/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2�
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpHmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2�
Qmodel/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOpQmodel/token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp2p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2�
Amodel/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpAmodel/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp2�
Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOpSmodel/token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp2�
Pmodel/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpPmodel/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense_5/Tensordot/ReadVariableOpCmodel/transformer_block/sequential/dense_5/Tensordot/ReadVariableOp2�
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpDmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp2�
Rmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpRmodel/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2�
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpJmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2�
Pmodel/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpPmodel/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_6/MatMul/ReadVariableOp#model/dense_6/MatMul/ReadVariableOp2L
$model/dense_7/BiasAdd/ReadVariableOp$model/dense_7/BiasAdd/ReadVariableOp2�
Cmodel/transformer_block/sequential/dense_4/Tensordot/ReadVariableOpCmodel/transformer_block/sequential/dense_4/Tensordot/ReadVariableOp2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_1:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
�8
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_66930

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�#AssignMovingAvg/Read/ReadVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp�%AssignMovingAvg_1/Read/ReadVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: o
moments/mean/reduction_indicesConst*
valueB"       *
dtype0*
_output_shapes
:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*#
_output_shapes
:�*
	keep_dims(*
T0i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*5
_output_shapes#
!:�������������������*
T0s
"moments/variance/reduction_indicesConst*
valueB"       *
dtype0*
_output_shapes
:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
squeeze_dims
 *
T0*
_output_shapes	
:�u
moments/Squeeze_1Squeezemoments/variance:output:0*
squeeze_dims
 *
T0*
_output_shapes	
:��
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *
�#<*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: �
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:��
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:�*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 �
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:�*
T0�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *
�#<*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: �
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:��
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:��
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 T
batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes	
:�*
T0�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
_output_shapes	
:�*
T0q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*5
_output_shapes#
!:�������������������*
T0i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*D
_input_shapes3
1:�������������������::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
�
�
*__inference_sequential_layer_call_fn_66797
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-66790*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_66789*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::22
StatefulPartitionedCallStatefulPartitionedCall: :' #
!
_user_specified_name	input_1: : : 
� 
�
B__inference_dense_5_layer_call_and_return_conditional_losses_66746

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��X
Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:_
Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       E
Tensordot/ShapeShapeinputs*
_output_shapes
:*
T0Y
Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
_output_shapes
: *
T0[
Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0k
Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��j
Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0\
Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�Y
Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*4
_input_shapes#
!:�����������::24
Tensordot/ReadVariableOpTensordot/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ì
�
@__inference_model_layer_call_and_return_conditional_losses_68512

inputsZ
Vtoken_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resourceX
Ttoken_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resourceU
Qtransformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resourceO
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resourceK
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resourceJ
Ftransformer_block_sequential_dense_4_tensordot_readvariableop_resourceH
Dtransformer_block_sequential_dense_4_biasadd_readvariableop_resourceJ
Ftransformer_block_sequential_dense_5_tensordot_readvariableop_resourceH
Dtransformer_block_sequential_dense_5_biasadd_readvariableop_resourceQ
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resourceM
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resourceD
@batch_normalization_assignmovingavg_read_readvariableop_resourceF
Bbatch_normalization_assignmovingavg_1_read_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity��7batch_normalization/AssignMovingAvg/AssignSubVariableOp�7batch_normalization/AssignMovingAvg/Read/ReadVariableOp�2batch_normalization/AssignMovingAvg/ReadVariableOp�9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp�9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�7token_and_position_embedding/embedding/embedding_lookup�Ktoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp�9token_and_position_embedding/embedding_1/embedding_lookup�Mtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp�>transformer_block/layer_normalization/batchnorm/ReadVariableOp�Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp�@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp�Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp�Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp�Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp�Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp�Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp�Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp�Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp�Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp�Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp�;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp�=transformer_block/sequential/dense_4/Tensordot/ReadVariableOp�;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp�=transformer_block/sequential/dense_5/Tensordot/ReadVariableOpX
"token_and_position_embedding/ShapeShapeinputs*
_output_shapes
:*
T0�
0token_and_position_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������|
2token_and_position_embedding/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:|
2token_and_position_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskj
(token_and_position_embedding/range/startConst*
value	B : *
dtype0*
_output_shapes
: j
(token_and_position_embedding/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :�
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*#
_output_shapes
:����������
Mtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOpReadVariableOpVtoken_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
���
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityUtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp:value:0* 
_output_shapes
:
��*
T0�
9token_and_position_embedding/embedding_1/embedding_lookupResourceGatherVtoken_and_position_embedding_embedding_1_embedding_lookup_read_readvariableop_resource+token_and_position_embedding/range:output:0N^token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*`
_classV
TRloc:@token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp*
Tindices0*
dtype0*(
_output_shapes
:�����������
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*`
_classV
TRloc:@token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp*(
_output_shapes
:�����������
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_2IdentityMtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*(
_output_shapes
:����������}
+token_and_position_embedding/embedding/CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:�����������
Ktoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOpReadVariableOpTtoken_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	��
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityStoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
7token_and_position_embedding/embedding/embedding_lookupResourceGatherTtoken_and_position_embedding_embedding_embedding_lookup_read_readvariableop_resource/token_and_position_embedding/embedding/Cast:y:0L^token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*-
_output_shapes
:�����������*^
_classT
RPloc:@token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp*
Tindices0�
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1Identity@token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*^
_classT
RPloc:@token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp*-
_output_shapes
:������������
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_2IdentityKtoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:������������
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_2:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_2:output:0*
T0*-
_output_shapes
:������������
1transformer_block/multi_head_self_attention/ShapeShape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:�
?transformer_block/multi_head_self_attention/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:�
Atransformer_block/multi_head_self_attention/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:�
Atransformer_block/multi_head_self_attention/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
9transformer_block/multi_head_self_attention/strided_sliceStridedSlice:transformer_block/multi_head_self_attention/Shape:output:0Htransformer_block/multi_head_self_attention/strided_slice/stack:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask�
Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
@transformer_block/multi_head_self_attention/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
@transformer_block/multi_head_self_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Atransformer_block/multi_head_self_attention/dense/Tensordot/ShapeShape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:�
Itransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Jtransformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Ktransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Jtransformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Ttransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Atransformer_block/multi_head_self_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
@transformer_block/multi_head_self_attention/dense/Tensordot/ProdProdMtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/dense/Tensordot/Const:output:0*
_output_shapes
: *
T0�
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1ProdOtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Ltransformer_block/multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Gtransformer_block/multi_head_self_attention/dense/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Btransformer_block/multi_head_self_attention/dense/Tensordot/concatConcatV2Itransformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Ptransformer_block/multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
Atransformer_block/multi_head_self_attention/dense/Tensordot/stackPackItransformer_block/multi_head_self_attention/dense/Tensordot/Prod:output:0Ktransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Etransformer_block/multi_head_self_attention/dense/Tensordot/transpose	Transpose$token_and_position_embedding/add:z:0Ktransformer_block/multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Ctransformer_block/multi_head_self_attention/dense/Tensordot/ReshapeReshapeItransformer_block/multi_head_self_attention/dense/Tensordot/transpose:y:0Jtransformer_block/multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ltransformer_block/multi_head_self_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense/Tensordot/transpose_1	TransposeRtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0Utransformer_block/multi_head_self_attention/dense/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Ktransformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Etransformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1ReshapeKtransformer_block/multi_head_self_attention/dense/Tensordot/transpose_1:y:0Ttransformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Btransformer_block/multi_head_self_attention/dense/Tensordot/MatMulMatMulLtransformer_block/multi_head_self_attention/dense/Tensordot/Reshape:output:0Ntransformer_block/multi_head_self_attention/dense/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:��
Itransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Mtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense/Tensordot/Const_2:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
;transformer_block/multi_head_self_attention/dense/TensordotReshapeLtransformer_block/multi_head_self_attention/dense/Tensordot/MatMul:product:0Mtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
9transformer_block/multi_head_self_attention/dense/BiasAddBiasAddDtransformer_block/multi_head_self_attention/dense/Tensordot:output:0Ptransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/ShapeShape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Htransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Itransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose	Transpose$token_and_position_embedding/add:z:0Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0�
Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
Itransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1	TransposeTtransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0Wtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0�
Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1ReshapeMtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose_1:y:0Vtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
=transformer_block/multi_head_self_attention/dense_1/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
;transformer_block/multi_head_self_attention/dense_1/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_1/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/ShapeShape$token_and_position_embedding/add:z:0*
_output_shapes
:*
T0�
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Htransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
_output_shapes
: *
T0�
Itransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose	Transpose$token_and_position_embedding/add:z:0Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
Itransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1	TransposeTtransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0Wtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1ReshapeMtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose_1:y:0Vtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
=transformer_block/multi_head_self_attention/dense_2/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
;transformer_block/multi_head_self_attention/dense_2/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_2/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
;transformer_block/multi_head_self_attention/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: }
;transformer_block/multi_head_self_attention/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: }
;transformer_block/multi_head_self_attention/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
value	B :`�
9transformer_block/multi_head_self_attention/Reshape/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/1:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/2:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:�
3transformer_block/multi_head_self_attention/ReshapeReshapeBtransformer_block/multi_head_self_attention/dense/BiasAdd:output:0Btransformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
:transformer_block/multi_head_self_attention/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
5transformer_block/multi_head_self_attention/transpose	Transpose<transformer_block/multi_head_self_attention/Reshape:output:0Ctransformer_block/multi_head_self_attention/transpose/perm:output:0*8
_output_shapes&
$:"������������������`*
T0�
=transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
=transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
=transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
;transformer_block/multi_head_self_attention/Reshape_1/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
T0*
N*
_output_shapes
:�
5transformer_block/multi_head_self_attention/Reshape_1ReshapeDtransformer_block/multi_head_self_attention/dense_1/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
<transformer_block/multi_head_self_attention/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
7transformer_block/multi_head_self_attention/transpose_1	Transpose>transformer_block/multi_head_self_attention/Reshape_1:output:0Etransformer_block/multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
=transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
=transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
=transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
;transformer_block/multi_head_self_attention/Reshape_2/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
T0*
N*
_output_shapes
:�
5transformer_block/multi_head_self_attention/Reshape_2ReshapeDtransformer_block/multi_head_self_attention/dense_2/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_2/shape:output:0*8
_output_shapes&
$:"������������������`*
T0�
<transformer_block/multi_head_self_attention/transpose_2/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
7transformer_block/multi_head_self_attention/transpose_2	Transpose>transformer_block/multi_head_self_attention/Reshape_2:output:0Etransformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
2transformer_block/multi_head_self_attention/MatMulBatchMatMulV29transformer_block/multi_head_self_attention/transpose:y:0;transformer_block/multi_head_self_attention/transpose_1:y:0*
adj_y(*
T0*A
_output_shapes/
-:+����������������������������
3transformer_block/multi_head_self_attention/Shape_1Shape;transformer_block/multi_head_self_attention/transpose_1:y:0*
_output_shapes
:*
T0�
Atransformer_block/multi_head_self_attention/strided_slice_1/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
;transformer_block/multi_head_self_attention/strided_slice_1StridedSlice<transformer_block/multi_head_self_attention/Shape_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice_1/stack:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
0transformer_block/multi_head_self_attention/CastCastDtransformer_block/multi_head_self_attention/strided_slice_1:output:0*

SrcT0*

DstT0*
_output_shapes
: �
0transformer_block/multi_head_self_attention/SqrtSqrt4transformer_block/multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: �
3transformer_block/multi_head_self_attention/truedivRealDiv;transformer_block/multi_head_self_attention/MatMul:output:04transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
3transformer_block/multi_head_self_attention/SoftmaxSoftmax7transformer_block/multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
4transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2=transformer_block/multi_head_self_attention/Softmax:softmax:0;transformer_block/multi_head_self_attention/transpose_2:y:0*8
_output_shapes&
$:"������������������`*
T0�
<transformer_block/multi_head_self_attention/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
7transformer_block/multi_head_self_attention/transpose_3	Transpose=transformer_block/multi_head_self_attention/MatMul_1:output:0Etransformer_block/multi_head_self_attention/transpose_3/perm:output:0*8
_output_shapes&
$:"������������������`*
T0�
=transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: �
=transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: �
;transformer_block/multi_head_self_attention/Reshape_3/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
_output_shapes
:*
T0�
5transformer_block/multi_head_self_attention/Reshape_3Reshape;transformer_block/multi_head_self_attention/transpose_3:y:0Dtransformer_block/multi_head_self_attention/Reshape_3/shape:output:0*5
_output_shapes#
!:�������������������*
T0�
Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/ShapeShape>transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Htransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0�
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Const:output:0*
_output_shapes
: *
T0�
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Itransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose	Transpose>transformer_block/multi_head_self_attention/Reshape_3:output:0Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/concat:output:0*5
_output_shapes#
!:�������������������*
T0�
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
Itransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1	TransposeTtransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0Wtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
Gtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1ReshapeMtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose_1:y:0Vtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:�
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
=transformer_block/multi_head_self_attention/dense_3/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*5
_output_shapes#
!:�������������������*
T0�
Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
;transformer_block/multi_head_self_attention/dense_3/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_3/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*5
_output_shapes#
!:�������������������*
T0k
&transformer_block/dropout/dropout/rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: �
'transformer_block/dropout/dropout/ShapeShapeDtransformer_block/multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:y
4transformer_block/dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: y
4transformer_block/dropout/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
>transformer_block/dropout/dropout/random_uniform/RandomUniformRandomUniform0transformer_block/dropout/dropout/Shape:output:0*
T0*
dtype0*5
_output_shapes#
!:��������������������
4transformer_block/dropout/dropout/random_uniform/subSub=transformer_block/dropout/dropout/random_uniform/max:output:0=transformer_block/dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
4transformer_block/dropout/dropout/random_uniform/mulMulGtransformer_block/dropout/dropout/random_uniform/RandomUniform:output:08transformer_block/dropout/dropout/random_uniform/sub:z:0*
T0*5
_output_shapes#
!:��������������������
0transformer_block/dropout/dropout/random_uniformAdd8transformer_block/dropout/dropout/random_uniform/mul:z:0=transformer_block/dropout/dropout/random_uniform/min:output:0*
T0*5
_output_shapes#
!:�������������������l
'transformer_block/dropout/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
%transformer_block/dropout/dropout/subSub0transformer_block/dropout/dropout/sub/x:output:0/transformer_block/dropout/dropout/rate:output:0*
T0*
_output_shapes
: p
+transformer_block/dropout/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
)transformer_block/dropout/dropout/truedivRealDiv4transformer_block/dropout/dropout/truediv/x:output:0)transformer_block/dropout/dropout/sub:z:0*
T0*
_output_shapes
: �
.transformer_block/dropout/dropout/GreaterEqualGreaterEqual4transformer_block/dropout/dropout/random_uniform:z:0/transformer_block/dropout/dropout/rate:output:0*
T0*5
_output_shapes#
!:��������������������
%transformer_block/dropout/dropout/mulMulDtransformer_block/multi_head_self_attention/dense_3/BiasAdd:output:0-transformer_block/dropout/dropout/truediv:z:0*5
_output_shapes#
!:�������������������*
T0�
&transformer_block/dropout/dropout/CastCast2transformer_block/dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*5
_output_shapes#
!:��������������������
'transformer_block/dropout/dropout/mul_1Mul)transformer_block/dropout/dropout/mul:z:0*transformer_block/dropout/dropout/Cast:y:0*
T0*5
_output_shapes#
!:��������������������
transformer_block/addAddV2$token_and_position_embedding/add:z:0+transformer_block/dropout/dropout/mul_1:z:0*
T0*-
_output_shapes
:������������
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:�����������
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(z
5transformer_block/layer_normalization/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*,
_output_shapes
:����������*
T0�
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:�����������
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*-
_output_shapes
:�����������*
T0�
=transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��}
3transformer_block/sequential/dense_4/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
3transformer_block/sequential/dense_4/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
4transformer_block/sequential/dense_4/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
7transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_4/Tensordot/Shape:output:0<transformer_block/sequential/dense_4/Tensordot/free:output:0Etransformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0�
>transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
9transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_4/Tensordot/Shape:output:0<transformer_block/sequential/dense_4/Tensordot/axes:output:0Gtransformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0~
4transformer_block/sequential/dense_4/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
3transformer_block/sequential/dense_4/Tensordot/ProdProd@transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_4/Tensordot/Const:output:0*
_output_shapes
: *
T0�
6transformer_block/sequential/dense_4/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
5transformer_block/sequential/dense_4/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
_output_shapes
: *
T0|
:transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
5transformer_block/sequential/dense_4/Tensordot/concatConcatV2<transformer_block/sequential/dense_4/Tensordot/free:output:0<transformer_block/sequential/dense_4/Tensordot/axes:output:0Ctransformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
4transformer_block/sequential/dense_4/Tensordot/stackPack<transformer_block/sequential/dense_4/Tensordot/Prod:output:0>transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
8transformer_block/sequential/dense_4/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0>transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
6transformer_block/sequential/dense_4/Tensordot/ReshapeReshape<transformer_block/sequential/dense_4/Tensordot/transpose:y:0=transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
?transformer_block/sequential/dense_4/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
:transformer_block/sequential/dense_4/Tensordot/transpose_1	TransposeEtransformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0Htransformer_block/sequential/dense_4/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
>transformer_block/sequential/dense_4/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
8transformer_block/sequential/dense_4/Tensordot/Reshape_1Reshape>transformer_block/sequential/dense_4/Tensordot/transpose_1:y:0Gtransformer_block/sequential/dense_4/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
5transformer_block/sequential/dense_4/Tensordot/MatMulMatMul?transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Atransformer_block/sequential/dense_4/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense_4/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:~
<transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
7transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
.transformer_block/sequential/dense_4/TensordotReshape?transformer_block/sequential/dense_4/Tensordot/MatMul:product:0@transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
,transformer_block/sequential/dense_4/BiasAddBiasAdd7transformer_block/sequential/dense_4/Tensordot:output:0Ctransformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
)transformer_block/sequential/dense_4/ReluRelu5transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*-
_output_shapes
:������������
=transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��}
3transformer_block/sequential/dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
3transformer_block/sequential/dense_5/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       �
4transformer_block/sequential/dense_5/Tensordot/ShapeShape7transformer_block/sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:~
<transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
7transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_5/Tensordot/Shape:output:0<transformer_block/sequential/dense_5/Tensordot/free:output:0Etransformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
9transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_5/Tensordot/Shape:output:0<transformer_block/sequential/dense_5/Tensordot/axes:output:0Gtransformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4transformer_block/sequential/dense_5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
3transformer_block/sequential/dense_5/Tensordot/ProdProd@transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_5/Tensordot/Const:output:0*
_output_shapes
: *
T0�
6transformer_block/sequential/dense_5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
5transformer_block/sequential/dense_5/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
5transformer_block/sequential/dense_5/Tensordot/concatConcatV2<transformer_block/sequential/dense_5/Tensordot/free:output:0<transformer_block/sequential/dense_5/Tensordot/axes:output:0Ctransformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
4transformer_block/sequential/dense_5/Tensordot/stackPack<transformer_block/sequential/dense_5/Tensordot/Prod:output:0>transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
8transformer_block/sequential/dense_5/Tensordot/transpose	Transpose7transformer_block/sequential/dense_4/Relu:activations:0>transformer_block/sequential/dense_5/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
6transformer_block/sequential/dense_5/Tensordot/ReshapeReshape<transformer_block/sequential/dense_5/Tensordot/transpose:y:0=transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
?transformer_block/sequential/dense_5/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
:transformer_block/sequential/dense_5/Tensordot/transpose_1	TransposeEtransformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0Htransformer_block/sequential/dense_5/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
>transformer_block/sequential/dense_5/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
8transformer_block/sequential/dense_5/Tensordot/Reshape_1Reshape>transformer_block/sequential/dense_5/Tensordot/transpose_1:y:0Gtransformer_block/sequential/dense_5/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
5transformer_block/sequential/dense_5/Tensordot/MatMulMatMul?transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Atransformer_block/sequential/dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:�����������
6transformer_block/sequential/dense_5/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�~
<transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
7transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
.transformer_block/sequential/dense_5/TensordotReshape?transformer_block/sequential/dense_5/Tensordot/MatMul:product:0@transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
,transformer_block/sequential/dense_5/BiasAddBiasAdd7transformer_block/sequential/dense_5/Tensordot:output:0Ctransformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������m
(transformer_block/dropout_1/dropout/rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: �
)transformer_block/dropout_1/dropout/ShapeShape5transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:{
6transformer_block/dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: {
6transformer_block/dropout_1/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
@transformer_block/dropout_1/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_1/dropout/Shape:output:0*
T0*
dtype0*-
_output_shapes
:������������
6transformer_block/dropout_1/dropout/random_uniform/subSub?transformer_block/dropout_1/dropout/random_uniform/max:output:0?transformer_block/dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
6transformer_block/dropout_1/dropout/random_uniform/mulMulItransformer_block/dropout_1/dropout/random_uniform/RandomUniform:output:0:transformer_block/dropout_1/dropout/random_uniform/sub:z:0*
T0*-
_output_shapes
:������������
2transformer_block/dropout_1/dropout/random_uniformAdd:transformer_block/dropout_1/dropout/random_uniform/mul:z:0?transformer_block/dropout_1/dropout/random_uniform/min:output:0*-
_output_shapes
:�����������*
T0n
)transformer_block/dropout_1/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
'transformer_block/dropout_1/dropout/subSub2transformer_block/dropout_1/dropout/sub/x:output:01transformer_block/dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: r
-transformer_block/dropout_1/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
+transformer_block/dropout_1/dropout/truedivRealDiv6transformer_block/dropout_1/dropout/truediv/x:output:0+transformer_block/dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: �
0transformer_block/dropout_1/dropout/GreaterEqualGreaterEqual6transformer_block/dropout_1/dropout/random_uniform:z:01transformer_block/dropout_1/dropout/rate:output:0*
T0*-
_output_shapes
:������������
'transformer_block/dropout_1/dropout/mulMul5transformer_block/sequential/dense_5/BiasAdd:output:0/transformer_block/dropout_1/dropout/truediv:z:0*-
_output_shapes
:�����������*
T0�
(transformer_block/dropout_1/dropout/CastCast4transformer_block/dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*-
_output_shapes
:������������
)transformer_block/dropout_1/dropout/mul_1Mul+transformer_block/dropout_1/dropout/mul:z:0,transformer_block/dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:������������
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_1/dropout/mul_1:z:0*
T0*-
_output_shapes
:������������
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*,
_output_shapes
:����������*
T0�
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:�
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(|
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:�����������
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0�
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:�����������b
 batch_normalization/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: b
 batch_normalization/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: �
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: �
2batch_normalization/moments/mean/reduction_indicesConst*
valueB"       *
dtype0*
_output_shapes
:�
 batch_normalization/moments/meanMean;transformer_block/layer_normalization_1/batchnorm/add_1:z:0;batch_normalization/moments/mean/reduction_indices:output:0*#
_output_shapes
:�*
	keep_dims(*
T0�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*#
_output_shapes
:��
-batch_normalization/moments/SquaredDifferenceSquaredDifference;transformer_block/layer_normalization_1/batchnorm/add_1:z:01batch_normalization/moments/StopGradient:output:0*-
_output_shapes
:�����������*
T0�
6batch_normalization/moments/variance/reduction_indicesConst*
valueB"       *
dtype0*
_output_shapes
:�
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*#
_output_shapes
:��
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
_output_shapes	
:�*
squeeze_dims
 *
T0�
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
squeeze_dims
 *
T0*
_output_shapes	
:��
7batch_normalization/AssignMovingAvg/Read/ReadVariableOpReadVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
,batch_normalization/AssignMovingAvg/IdentityIdentity?batch_normalization/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:�*
T0�
)batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *
�#<*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: �
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource8^batch_normalization/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:�*
T0*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:��
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp@batch_normalization_assignmovingavg_read_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@batch_normalization/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 �
9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
.batch_normalization/AssignMovingAvg_1/IdentityIdentityAbatch_normalization/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:�*
T0�
+batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *
�#<*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource:^batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:��
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:�*
T0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp�
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_assignmovingavg_1_read_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*L
_classB
@>loc:@batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 h
#batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_1Mul;transformer_block/layer_normalization_1/batchnorm/add_1:z:0%batch_normalization/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*-
_output_shapes
:�����������l
*global_max_pooling1d/Max/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :�
global_max_pooling1d/MaxMax'batch_normalization/batchnorm/add_1:z:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������[
dropout_2/dropout/rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: h
dropout_2/dropout/ShapeShape!global_max_pooling1d/Max:output:0*
T0*
_output_shapes
:i
$dropout_2/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    i
$dropout_2/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
dtype0*(
_output_shapes
:����������*
T0�
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*(
_output_shapes
:����������*
T0�
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������\
dropout_2/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_2/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*(
_output_shapes
:�����������
dropout_2/dropout/mulMul!global_max_pooling1d/Max:output:0dropout_2/dropout/truediv:z:0*
T0*(
_output_shapes
:�����������
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:�����������
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*(
_output_shapes
:����������*
T0�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
���
dense_6/MatMulMatMuldropout_2/dropout/mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	��
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitydense_7/Sigmoid:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp8^batch_normalization/AssignMovingAvg/Read/ReadVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookupL^token_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp:^token_and_position_embedding/embedding_1/embedding_lookupN^token_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpI^transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpK^transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpK^transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpM^transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpK^transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpM^transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpK^transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpM^transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp<^transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp>^transformer_block/sequential/dense_4/Tensordot/ReadVariableOp<^transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp>^transformer_block/sequential/dense_5/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp9batch_normalization/AssignMovingAvg_1/Read/ReadVariableOp2�
Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpJtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2r
7batch_normalization/AssignMovingAvg/Read/ReadVariableOp7batch_normalization/AssignMovingAvg/Read/ReadVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2�
Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpJtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp2�
Mtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOpMtoken_and_position_embedding/embedding_1/embedding_lookup/Read/ReadVariableOp2�
Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2�
Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpJtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2�
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2~
=transformer_block/sequential/dense_5/Tensordot/ReadVariableOp=transformer_block/sequential/dense_5/Tensordot/ReadVariableOp2z
;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp2�
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2�
Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpJtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2�
Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpHtransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp2�
Ktoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOpKtoken_and_position_embedding/embedding/embedding_lookup/Read/ReadVariableOp2�
Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2z
;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp2�
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2�
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2~
=transformer_block/sequential/dense_4/Tensordot/ReadVariableOp=transformer_block/sequential/dense_4/Tensordot/ReadVariableOp2�
Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp: : : : : : : :	 :
 : : : : : : : : : : : : : : : : :& "
 
_user_specified_nameinputs: 
�
�
%__inference_model_layer_call_fn_67971
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*,
_gradient_op_typePartitionedCall-67942*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_67941*
Tout
2*-
config_proto

GPU

CPU2*0J 8*&
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : : : : : : : :' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : 
��
�
L__inference_transformer_block_layer_call_and_return_conditional_losses_67606

inputsE
Amulti_head_self_attention_dense_tensordot_readvariableop_resourceC
?multi_head_self_attention_dense_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_1_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_1_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_2_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_2_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_3_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_3_biasadd_readvariableop_resource=
9layer_normalization_batchnorm_mul_readvariableop_resource9
5layer_normalization_batchnorm_readvariableop_resource8
4sequential_dense_4_tensordot_readvariableop_resource6
2sequential_dense_4_biasadd_readvariableop_resource8
4sequential_dense_5_tensordot_readvariableop_resource6
2sequential_dense_5_biasadd_readvariableop_resource?
;layer_normalization_1_batchnorm_mul_readvariableop_resource;
7layer_normalization_1_batchnorm_readvariableop_resource
identity��,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�6multi_head_self_attention/dense/BiasAdd/ReadVariableOp�8multi_head_self_attention/dense/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp�)sequential/dense_4/BiasAdd/ReadVariableOp�+sequential/dense_4/Tensordot/ReadVariableOp�)sequential/dense_5/BiasAdd/ReadVariableOp�+sequential/dense_5/Tensordot/ReadVariableOpU
multi_head_self_attention/ShapeShapeinputs*
_output_shapes
:*
T0w
-multi_head_self_attention/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/multi_head_self_attention/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:y
/multi_head_self_attention/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:�
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask�
8multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��x
.multi_head_self_attention/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
.multi_head_self_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:e
/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:y
7multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense/Tensordot/GatherV2GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/free:output:0@multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0{
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/axes:output:0Bmulti_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
.multi_head_self_attention/dense/Tensordot/ProdProd;multi_head_self_attention/dense/Tensordot/GatherV2:output:08multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense/Tensordot/Prod_1Prod=multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0:multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
0multi_head_self_attention/dense/Tensordot/concatConcatV27multi_head_self_attention/dense/Tensordot/free:output:07multi_head_self_attention/dense/Tensordot/axes:output:0>multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
/multi_head_self_attention/dense/Tensordot/stackPack7multi_head_self_attention/dense/Tensordot/Prod:output:09multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
3multi_head_self_attention/dense/Tensordot/transpose	Transposeinputs9multi_head_self_attention/dense/Tensordot/concat:output:0*-
_output_shapes
:�����������*
T0�
1multi_head_self_attention/dense/Tensordot/ReshapeReshape7multi_head_self_attention/dense/Tensordot/transpose:y:08multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
:multi_head_self_attention/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense/Tensordot/transpose_1	Transpose@multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0Cmulti_head_self_attention/dense/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
9multi_head_self_attention/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
3multi_head_self_attention/dense/Tensordot/Reshape_1Reshape9multi_head_self_attention/dense/Tensordot/transpose_1:y:0Bmulti_head_self_attention/dense/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
0multi_head_self_attention/dense/Tensordot/MatMulMatMul:multi_head_self_attention/dense/Tensordot/Reshape:output:0<multi_head_self_attention/dense/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������|
1multi_head_self_attention/dense/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�y
7multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense/Tensordot/concat_1ConcatV2;multi_head_self_attention/dense/Tensordot/GatherV2:output:0:multi_head_self_attention/dense/Tensordot/Const_2:output:0@multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
)multi_head_self_attention/dense/TensordotReshape:multi_head_self_attention/dense/Tensordot/MatMul:product:0;multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
6multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
'multi_head_self_attention/dense/BiasAddBiasAdd2multi_head_self_attention/dense/Tensordot:output:0>multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:g
1multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : �
4multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/free:output:0Bmulti_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0}
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0Dmulti_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0{
1multi_head_self_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_1/Tensordot/ProdProd=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_1/Tensordot/Const:output:0*
_output_shapes
: *
T0}
3multi_head_self_attention/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_1/Tensordot/Prod_1Prod?multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
2multi_head_self_attention/dense_1/Tensordot/concatConcatV29multi_head_self_attention/dense_1/Tensordot/free:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0@multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_1/Tensordot/stackPack9multi_head_self_attention/dense_1/Tensordot/Prod:output:0;multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
3multi_head_self_attention/dense_1/Tensordot/ReshapeReshape9multi_head_self_attention/dense_1/Tensordot/transpose:y:0:multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<multi_head_self_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_1/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_1/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_1/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_1/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_1/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
2multi_head_self_attention/dense_1/Tensordot/MatMulMatMul<multi_head_self_attention/dense_1/Tensordot/Reshape:output:0>multi_head_self_attention/dense_1/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0~
3multi_head_self_attention/dense_1/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_1/TensordotReshape<multi_head_self_attention/dense_1/Tensordot/MatMul:product:0=multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_1/BiasAddBiasAdd4multi_head_self_attention/dense_1/Tensordot:output:0@multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:g
1multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
_output_shapes
:*
T0{
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/free:output:0Bmulti_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0}
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0Dmulti_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0{
1multi_head_self_attention/dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_2/Tensordot/ProdProd=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_2/Tensordot/Prod_1Prod?multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
2multi_head_self_attention/dense_2/Tensordot/concatConcatV29multi_head_self_attention/dense_2/Tensordot/free:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0@multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
1multi_head_self_attention/dense_2/Tensordot/stackPack9multi_head_self_attention/dense_2/Tensordot/Prod:output:0;multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
_output_shapes
:*
T0�
5multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_2/Tensordot/concat:output:0*-
_output_shapes
:�����������*
T0�
3multi_head_self_attention/dense_2/Tensordot/ReshapeReshape9multi_head_self_attention/dense_2/Tensordot/transpose:y:0:multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<multi_head_self_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_2/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_2/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_2/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_2/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_2/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_2/Tensordot/MatMulMatMul<multi_head_self_attention/dense_2/Tensordot/Reshape:output:0>multi_head_self_attention/dense_2/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0~
3multi_head_self_attention/dense_2/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_2/TensordotReshape<multi_head_self_attention/dense_2/Tensordot/MatMul:product:0=multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_2/BiasAddBiasAdd4multi_head_self_attention/dense_2/Tensordot:output:0@multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������t
)multi_head_self_attention/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: k
)multi_head_self_attention/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: k
)multi_head_self_attention/Reshape/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:�
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/dense/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
(multi_head_self_attention/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*8
_output_shapes&
$:"������������������`*
T0v
+multi_head_self_attention/Reshape_1/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_1/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_1Reshape2multi_head_self_attention/dense_1/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*8
_output_shapes&
$:"������������������`*
T0v
+multi_head_self_attention/Reshape_2/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_2/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_2Reshape2multi_head_self_attention/dense_2/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+���������������������������*
adj_y(z
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
_output_shapes
:*
T0�
/multi_head_self_attention/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
���������{
1multi_head_self_attention/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:{
1multi_head_self_attention/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0�
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

SrcT0*

DstT0*
_output_shapes
: k
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: �
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*8
_output_shapes&
$:"������������������`*
T0�
*multi_head_self_attention/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������`v
+multi_head_self_attention/Reshape_3/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: n
+multi_head_self_attention/Reshape_3/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_3/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:�
0multi_head_self_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
1multi_head_self_attention/dense_3/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
_output_shapes
:*
T0{
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/free:output:0Bmulti_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0}
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0Dmulti_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0{
1multi_head_self_attention/dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_3/Tensordot/ProdProd=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_3/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_3/Tensordot/Prod_1Prod?multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense_3/Tensordot/concatConcatV29multi_head_self_attention/dense_3/Tensordot/free:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0@multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_3/Tensordot/stackPack9multi_head_self_attention/dense_3/Tensordot/Prod:output:0;multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_3/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:0;multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
3multi_head_self_attention/dense_3/Tensordot/ReshapeReshape9multi_head_self_attention/dense_3/Tensordot/transpose:y:0:multi_head_self_attention/dense_3/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0�
<multi_head_self_attention/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_3/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_3/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_3/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_3/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_3/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_3/Tensordot/MatMulMatMul<multi_head_self_attention/dense_3/Tensordot/Reshape:output:0>multi_head_self_attention/dense_3/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0~
3multi_head_self_attention/dense_3/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
+multi_head_self_attention/dense_3/TensordotReshape<multi_head_self_attention/dense_3/Tensordot/MatMul:product:0=multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_3/BiasAddBiasAdd4multi_head_self_attention/dense_3/Tensordot:output:0@multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
dropout/IdentityIdentity2multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������g
addAddV2inputsdropout/Identity:output:0*
T0*-
_output_shapes
:�����������|
2layer_normalization/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(*
T0�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*,
_output_shapes
:����������*
T0�
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
6layer_normalization/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(*
T0h
#layer_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*,
_output_shapes
:����������*
T0�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��k
!sequential/dense_4/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:r
!sequential/dense_4/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:y
"sequential/dense_4/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
_output_shapes
: *
T0n
$sequential/dense_4/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
&sequential/dense_4/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������~
-sequential/dense_4/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
(sequential/dense_4/Tensordot/transpose_1	Transpose3sequential/dense_4/Tensordot/ReadVariableOp:value:06sequential/dense_4/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��}
,sequential/dense_4/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
&sequential/dense_4/Tensordot/Reshape_1Reshape,sequential/dense_4/Tensordot/transpose_1:y:05sequential/dense_4/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:0/sequential/dense_4/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������o
$sequential/dense_4/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:l
*sequential/dense_4/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������|
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*-
_output_shapes
:������������
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��k
!sequential/dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:r
!sequential/dense_5/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:w
"sequential/dense_5/Tensordot/ShapeShape%sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_5/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0n
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
_output_shapes
: *
T0n
$sequential/dense_5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_5/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
&sequential/dense_5/Tensordot/transpose	Transpose%sequential/dense_4/Relu:activations:0,sequential/dense_5/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0~
-sequential/dense_5/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
(sequential/dense_5/Tensordot/transpose_1	Transpose3sequential/dense_5/Tensordot/ReadVariableOp:value:06sequential/dense_5/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��}
,sequential/dense_5/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
&sequential/dense_5/Tensordot/Reshape_1Reshape,sequential/dense_5/Tensordot/transpose_1:y:05sequential/dense_5/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:0/sequential/dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������o
$sequential/dense_5/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�l
*sequential/dense_5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������{
dropout_1/IdentityIdentity#sequential/dense_5/BiasAdd:output:0*
T0*-
_output_shapes
:������������
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*-
_output_shapes
:�����������~
4layer_normalization_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*,
_output_shapes
:�����������
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:�����������
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
8layer_normalization_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*,
_output_shapes
:����������j
%layer_normalization_1/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*,
_output_shapes
:����������*
T0�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*-
_output_shapes
:�����������*
T0�
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*-
_output_shapes
:�����������*
T0�
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp7^multi_head_self_attention/dense/BiasAdd/ReadVariableOp9^multi_head_self_attention/dense/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_1/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_2/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp,^sequential/dense_5/Tensordot/ReadVariableOp*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*l
_input_shapes[
Y:�����������::::::::::::::::2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp2t
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_5/Tensordot/ReadVariableOp+sequential/dense_5/Tensordot/ReadVariableOp2p
6multi_head_self_attention/dense/BiasAdd/ReadVariableOp6multi_head_self_attention/dense/BiasAdd/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2x
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2t
8multi_head_self_attention/dense/Tensordot/ReadVariableOp8multi_head_self_attention/dense/Tensordot/ReadVariableOp2x
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2t
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2x
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : 
�8
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67711

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�#AssignMovingAvg/Read/ReadVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp�%AssignMovingAvg_1/Read/ReadVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 ZN
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: o
moments/mean/reduction_indicesConst*
valueB"       *
dtype0*
_output_shapes
:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*#
_output_shapes
:�*
	keep_dims(*
T0i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:�����������s
"moments/variance/reduction_indicesConst*
valueB"       *
dtype0*
_output_shapes
:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*#
_output_shapes
:�o
moments/SqueezeSqueezemoments/mean:output:0*
_output_shapes	
:�*
squeeze_dims
 *
T0u
moments/Squeeze_1Squeezemoments/variance:output:0*
_output_shapes	
:�*
squeeze_dims
 *
T0�
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:�*
T0�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *
�#<*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: �
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:��
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:��
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 �
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *
�#<*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:�*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:��
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 T
batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes	
:�*
T0�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�i
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*-
_output_shapes
:�����������*
T0i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�x
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
�
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_69768

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:����������*
T0\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:����������*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_69763

inputs
identity�Q
dropout/rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:����������*
T0�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:����������*
T0R
dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:����������*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�8
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69702

inputs0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�#AssignMovingAvg/Read/ReadVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp�%AssignMovingAvg_1/Read/ReadVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: o
moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:�*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:�����������s
"moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*#
_output_shapes
:�o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
_output_shapes	
:�*
squeeze_dims
 *
T0�
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *
�#<*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: �
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:��
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:�*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 �
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *
�#<*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: �
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:��
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:�*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 T
batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
_output_shapes	
:�*
T0�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�i
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*-
_output_shapes
:�����������*
T0i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:��
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�x
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
�G
�
E__inference_sequential_layer_call_and_return_conditional_losses_69944

inputs-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity��dense_4/BiasAdd/ReadVariableOp� dense_4/Tensordot/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��`
dense_4/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:g
dense_4/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:M
dense_4/Tensordot/ShapeShapeinputs*
_output_shapes
:*
T0a
dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0c
!dense_4/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
_output_shapes
: *
T0_
dense_4/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
dense_4/Tensordot/transpose	Transposeinputs!dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������s
"dense_4/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
dense_4/Tensordot/transpose_1	Transpose(dense_4/Tensordot/ReadVariableOp:value:0+dense_4/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��r
!dense_4/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"�   �   �
dense_4/Tensordot/Reshape_1Reshape!dense_4/Tensordot/transpose_1:y:0*dense_4/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0$dense_4/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������d
dense_4/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:a
dense_4/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������f
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*-
_output_shapes
:������������
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��`
dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:g
dense_5/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:a
dense_5/Tensordot/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0a
dense_5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
_output_shapes
: *
T0c
dense_5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
dense_5/Tensordot/transpose	Transposedense_4/Relu:activations:0!dense_5/Tensordot/concat:output:0*-
_output_shapes
:�����������*
T0�
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������s
"dense_5/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
dense_5/Tensordot/transpose_1	Transpose(dense_5/Tensordot/ReadVariableOp:value:0+dense_5/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0r
!dense_5/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
dense_5/Tensordot/Reshape_1Reshape!dense_5/Tensordot/transpose_1:y:0*dense_5/Tensordot/Reshape_1/shape:output:0* 
_output_shapes
:
��*
T0�
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0$dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������d
dense_5/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:a
dense_5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
IdentityIdentitydense_5/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
�G
�
E__inference_sequential_layer_call_and_return_conditional_losses_69879

inputs-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity��dense_4/BiasAdd/ReadVariableOp� dense_4/Tensordot/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��`
dense_4/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:g
dense_4/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:M
dense_4/Tensordot/ShapeShapeinputs*
_output_shapes
:*
T0a
dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0c
!dense_4/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0a
dense_4/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: �
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
dense_4/Tensordot/transpose	Transposeinputs!dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������s
"dense_4/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
dense_4/Tensordot/transpose_1	Transpose(dense_4/Tensordot/ReadVariableOp:value:0+dense_4/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��r
!dense_4/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
dense_4/Tensordot/Reshape_1Reshape!dense_4/Tensordot/transpose_1:y:0*dense_4/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0$dense_4/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������d
dense_4/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�a
dense_4/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*-
_output_shapes
:�����������*
T0f
dense_4/ReluReludense_4/BiasAdd:output:0*-
_output_shapes
:�����������*
T0�
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��`
dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:g
dense_5/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:a
dense_5/Tensordot/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0c
!dense_5/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
_output_shapes
: *
T0_
dense_5/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
dense_5/Tensordot/transpose	Transposedense_4/Relu:activations:0!dense_5/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������s
"dense_5/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
dense_5/Tensordot/transpose_1	Transpose(dense_5/Tensordot/ReadVariableOp:value:0+dense_5/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��r
!dense_5/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
dense_5/Tensordot/Reshape_1Reshape!dense_5/Tensordot/transpose_1:y:0*dense_5/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0$dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������d
dense_5/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:a
dense_5/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
IdentityIdentitydense_5/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp*-
_output_shapes
:�����������*
T0"
identityIdentity:output:0*<
_input_shapes+
):�����������::::2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
�
�
#__inference_signature_wrapper_68121
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*
Tout
2*-
config_proto

GPU

CPU2*0J 8*&
Tin
2*'
_output_shapes
:���������*,
_gradient_op_typePartitionedCall-68092*)
f$R"
 __inference__wrapped_model_66654�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69643

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpN
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: �
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�T
batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:��
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*5
_output_shapes#
!:�������������������*
T0�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
_output_shapes	
:�*
T0�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
_output_shapes	
:�*
T0�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*D
_input_shapes3
1:�������������������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_2: : : :& "
 
_user_specified_nameinputs: 
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_66789

inputs*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity��dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_66695*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-66701�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66752*K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_66746*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:������������
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_66776
input_1*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity��dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-66701*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_66695*
Tout
2�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-66752*K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_66746*
Tout
2�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall: : : :' #
!
_user_specified_name	input_1: 
�	
�
B__inference_dense_6_layer_call_and_return_conditional_losses_69789

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
k
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_66980

inputs
identityW
Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:������������������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:& "
 
_user_specified_nameinputs
�4
�
@__inference_model_layer_call_and_return_conditional_losses_67901
input_1?
;token_and_position_embedding_statefulpartitionedcall_args_1?
;token_and_position_embedding_statefulpartitionedcall_args_24
0transformer_block_statefulpartitionedcall_args_14
0transformer_block_statefulpartitionedcall_args_24
0transformer_block_statefulpartitionedcall_args_34
0transformer_block_statefulpartitionedcall_args_44
0transformer_block_statefulpartitionedcall_args_54
0transformer_block_statefulpartitionedcall_args_64
0transformer_block_statefulpartitionedcall_args_74
0transformer_block_statefulpartitionedcall_args_84
0transformer_block_statefulpartitionedcall_args_95
1transformer_block_statefulpartitionedcall_args_105
1transformer_block_statefulpartitionedcall_args_115
1transformer_block_statefulpartitionedcall_args_125
1transformer_block_statefulpartitionedcall_args_135
1transformer_block_statefulpartitionedcall_args_145
1transformer_block_statefulpartitionedcall_args_155
1transformer_block_statefulpartitionedcall_args_166
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_4*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identity��+batch_normalization/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�)transformer_block/StatefulPartitionedCall�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1;token_and_position_embedding_statefulpartitionedcall_args_1;token_and_position_embedding_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-67027*`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_67021*
Tout
2*-
config_proto

GPU

CPU2*0J 8*-
_output_shapes
:�����������*
Tin
2�	
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:00transformer_block_statefulpartitionedcall_args_10transformer_block_statefulpartitionedcall_args_20transformer_block_statefulpartitionedcall_args_30transformer_block_statefulpartitionedcall_args_40transformer_block_statefulpartitionedcall_args_50transformer_block_statefulpartitionedcall_args_60transformer_block_statefulpartitionedcall_args_70transformer_block_statefulpartitionedcall_args_80transformer_block_statefulpartitionedcall_args_91transformer_block_statefulpartitionedcall_args_101transformer_block_statefulpartitionedcall_args_111transformer_block_statefulpartitionedcall_args_121transformer_block_statefulpartitionedcall_args_131transformer_block_statefulpartitionedcall_args_141transformer_block_statefulpartitionedcall_args_151transformer_block_statefulpartitionedcall_args_16*,
_gradient_op_typePartitionedCall-67634*U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_67606*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*-
_output_shapes
:������������
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall2transformer_block/StatefulPartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*-
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-67747*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67734*
Tout
2�
$global_max_pooling1d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-66986*X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_66980�
dropout_2/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-67800*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_67788*
Tout
2�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_67816*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:����������*,
_gradient_op_typePartitionedCall-67822�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:���������*,
_gradient_op_typePartitionedCall-67850*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_67844*
Tout
2�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*'
_output_shapes
:���������*
T0"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
�
�
L__inference_transformer_block_layer_call_and_return_conditional_losses_67336

inputsE
Amulti_head_self_attention_dense_tensordot_readvariableop_resourceC
?multi_head_self_attention_dense_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_1_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_1_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_2_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_2_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_3_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_3_biasadd_readvariableop_resource=
9layer_normalization_batchnorm_mul_readvariableop_resource9
5layer_normalization_batchnorm_readvariableop_resource8
4sequential_dense_4_tensordot_readvariableop_resource6
2sequential_dense_4_biasadd_readvariableop_resource8
4sequential_dense_5_tensordot_readvariableop_resource6
2sequential_dense_5_biasadd_readvariableop_resource?
;layer_normalization_1_batchnorm_mul_readvariableop_resource;
7layer_normalization_1_batchnorm_readvariableop_resource
identity��,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�6multi_head_self_attention/dense/BiasAdd/ReadVariableOp�8multi_head_self_attention/dense/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp�8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp�:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp�)sequential/dense_4/BiasAdd/ReadVariableOp�+sequential/dense_4/Tensordot/ReadVariableOp�)sequential/dense_5/BiasAdd/ReadVariableOp�+sequential/dense_5/Tensordot/ReadVariableOpU
multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:w
-multi_head_self_attention/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/multi_head_self_attention/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/multi_head_self_attention/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0�
8multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��x
.multi_head_self_attention/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
.multi_head_self_attention/dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:e
/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
_output_shapes
:*
T0y
7multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense/Tensordot/GatherV2GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/free:output:0@multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0{
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/axes:output:0Bmulti_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0y
/multi_head_self_attention/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
.multi_head_self_attention/dense/Tensordot/ProdProd;multi_head_self_attention/dense/Tensordot/GatherV2:output:08multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense/Tensordot/Prod_1Prod=multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0:multi_head_self_attention/dense/Tensordot/Const_1:output:0*
_output_shapes
: *
T0w
5multi_head_self_attention/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
0multi_head_self_attention/dense/Tensordot/concatConcatV27multi_head_self_attention/dense/Tensordot/free:output:07multi_head_self_attention/dense/Tensordot/axes:output:0>multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
/multi_head_self_attention/dense/Tensordot/stackPack7multi_head_self_attention/dense/Tensordot/Prod:output:09multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
3multi_head_self_attention/dense/Tensordot/transpose	Transposeinputs9multi_head_self_attention/dense/Tensordot/concat:output:0*-
_output_shapes
:�����������*
T0�
1multi_head_self_attention/dense/Tensordot/ReshapeReshape7multi_head_self_attention/dense/Tensordot/transpose:y:08multi_head_self_attention/dense/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0�
:multi_head_self_attention/dense/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       �
5multi_head_self_attention/dense/Tensordot/transpose_1	Transpose@multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0Cmulti_head_self_attention/dense/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0�
9multi_head_self_attention/dense/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
3multi_head_self_attention/dense/Tensordot/Reshape_1Reshape9multi_head_self_attention/dense/Tensordot/transpose_1:y:0Bmulti_head_self_attention/dense/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
0multi_head_self_attention/dense/Tensordot/MatMulMatMul:multi_head_self_attention/dense/Tensordot/Reshape:output:0<multi_head_self_attention/dense/Tensordot/Reshape_1:output:0*(
_output_shapes
:����������*
T0|
1multi_head_self_attention/dense/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:y
7multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense/Tensordot/concat_1ConcatV2;multi_head_self_attention/dense/Tensordot/GatherV2:output:0:multi_head_self_attention/dense/Tensordot/Const_2:output:0@multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
)multi_head_self_attention/dense/TensordotReshape:multi_head_self_attention/dense/Tensordot/MatMul:product:0;multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
6multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
'multi_head_self_attention/dense/BiasAddBiasAdd2multi_head_self_attention/dense/Tensordot:output:0>multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_1_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_1/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:�
0multi_head_self_attention/dense_1/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:g
1multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
_output_shapes
:*
T0{
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : �
4multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/free:output:0Bmulti_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0}
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0Dmulti_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0{
1multi_head_self_attention/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_1/Tensordot/ProdProd=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_1/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_1/Tensordot/Prod_1Prod?multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
2multi_head_self_attention/dense_1/Tensordot/concatConcatV29multi_head_self_attention/dense_1/Tensordot/free:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0@multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_1/Tensordot/stackPack9multi_head_self_attention/dense_1/Tensordot/Prod:output:0;multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
3multi_head_self_attention/dense_1/Tensordot/ReshapeReshape9multi_head_self_attention/dense_1/Tensordot/transpose:y:0:multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<multi_head_self_attention/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_1/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_1/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0�
;multi_head_self_attention/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_1/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_1/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_1/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_1/Tensordot/MatMulMatMul<multi_head_self_attention/dense_1/Tensordot/Reshape:output:0>multi_head_self_attention/dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������~
3multi_head_self_attention/dense_1/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
_output_shapes
:*
T0�
+multi_head_self_attention/dense_1/TensordotReshape<multi_head_self_attention/dense_1/Tensordot/MatMul:product:0=multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_1/BiasAddBiasAdd4multi_head_self_attention/dense_1/Tensordot:output:0@multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_2_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_2/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:g
1multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:{
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/free:output:0Bmulti_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0}
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : �
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0Dmulti_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0{
1multi_head_self_attention/dense_2/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: �
0multi_head_self_attention/dense_2/Tensordot/ProdProd=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_2/Tensordot/Const:output:0*
_output_shapes
: *
T0}
3multi_head_self_attention/dense_2/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_2/Tensordot/Prod_1Prod?multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
_output_shapes
: *
T0y
7multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
2multi_head_self_attention/dense_2/Tensordot/concatConcatV29multi_head_self_attention/dense_2/Tensordot/free:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0@multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_2/Tensordot/stackPack9multi_head_self_attention/dense_2/Tensordot/Prod:output:0;multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_2/Tensordot/concat:output:0*-
_output_shapes
:�����������*
T0�
3multi_head_self_attention/dense_2/Tensordot/ReshapeReshape9multi_head_self_attention/dense_2/Tensordot/transpose:y:0:multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<multi_head_self_attention/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_2/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_2/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_2/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_2/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_2/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_2/Tensordot/MatMulMatMul<multi_head_self_attention/dense_2/Tensordot/Reshape:output:0>multi_head_self_attention/dense_2/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������~
3multi_head_self_attention/dense_2/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_2/TensordotReshape<multi_head_self_attention/dense_2/Tensordot/MatMul:product:0=multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_2/BiasAddBiasAdd4multi_head_self_attention/dense_2/Tensordot:output:0@multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������t
)multi_head_self_attention/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: k
)multi_head_self_attention/Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: k
)multi_head_self_attention/Reshape/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
T0*
N*
_output_shapes
:�
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/dense/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*8
_output_shapes&
$:"������������������`*
T0�
(multi_head_self_attention/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������`v
+multi_head_self_attention/Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
���������m
+multi_head_self_attention/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :m
+multi_head_self_attention/Reshape_1/shape/3Const*
dtype0*
_output_shapes
: *
value	B :`�
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_1Reshape2multi_head_self_attention/dense_1/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"������������������`v
+multi_head_self_attention/Reshape_2/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: m
+multi_head_self_attention/Reshape_2/shape/2Const*
dtype0*
_output_shapes
: *
value	B :m
+multi_head_self_attention/Reshape_2/shape/3Const*
value	B :`*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_2Reshape2multi_head_self_attention/dense_2/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_2/permConst*
dtype0*
_output_shapes
:*%
valueB"             �
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"������������������`�
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*A
_output_shapes/
-:+���������������������������*
adj_y(*
T0z
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
_output_shapes
:*
T0�
/multi_head_self_attention/strided_slice_1/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:{
1multi_head_self_attention/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:{
1multi_head_self_attention/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

SrcT0*

DstT0*
_output_shapes
: k
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: �
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"������������������`�
*multi_head_self_attention/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:�
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������`v
+multi_head_self_attention/Reshape_3/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: n
+multi_head_self_attention/Reshape_3/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: �
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
T0*
N*
_output_shapes
:�
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*5
_output_shapes#
!:�������������������*
T0�
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_3_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��z
0multi_head_self_attention/dense_3/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:�
1multi_head_self_attention/dense_3/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:{
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/free:output:0Bmulti_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0}
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0Dmulti_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1multi_head_self_attention/dense_3/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
0multi_head_self_attention/dense_3/Tensordot/ProdProd=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3multi_head_self_attention/dense_3/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
2multi_head_self_attention/dense_3/Tensordot/Prod_1Prod?multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
_output_shapes
: *
T0y
7multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : �
2multi_head_self_attention/dense_3/Tensordot/concatConcatV29multi_head_self_attention/dense_3/Tensordot/free:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0@multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
1multi_head_self_attention/dense_3/Tensordot/stackPack9multi_head_self_attention/dense_3/Tensordot/Prod:output:0;multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
5multi_head_self_attention/dense_3/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:0;multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
3multi_head_self_attention/dense_3/Tensordot/ReshapeReshape9multi_head_self_attention/dense_3/Tensordot/transpose:y:0:multi_head_self_attention/dense_3/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0�
<multi_head_self_attention/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
7multi_head_self_attention/dense_3/Tensordot/transpose_1	TransposeBmulti_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0Emulti_head_self_attention/dense_3/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
���
;multi_head_self_attention/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
5multi_head_self_attention/dense_3/Tensordot/Reshape_1Reshape;multi_head_self_attention/dense_3/Tensordot/transpose_1:y:0Dmulti_head_self_attention/dense_3/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2multi_head_self_attention/dense_3/Tensordot/MatMulMatMul<multi_head_self_attention/dense_3/Tensordot/Reshape:output:0>multi_head_self_attention/dense_3/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������~
3multi_head_self_attention/dense_3/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:{
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
4multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
+multi_head_self_attention/dense_3/TensordotReshape<multi_head_self_attention/dense_3/Tensordot/MatMul:product:0=multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*5
_output_shapes#
!:�������������������*
T0�
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
)multi_head_self_attention/dense_3/BiasAddBiasAdd4multi_head_self_attention/dense_3/Tensordot:output:0@multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������Y
dropout/dropout/rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: w
dropout/dropout/ShapeShape2multi_head_self_attention/dense_3/BiasAdd:output:0*
_output_shapes
:*
T0g
"dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
dtype0*5
_output_shapes#
!:�������������������*
T0�
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*5
_output_shapes#
!:��������������������
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*5
_output_shapes#
!:�������������������*
T0Z
dropout/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
_output_shapes
: *
T0�
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*5
_output_shapes#
!:��������������������
dropout/dropout/mulMul2multi_head_self_attention/dense_3/BiasAdd:output:0dropout/dropout/truediv:z:0*
T0*5
_output_shapes#
!:��������������������
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*5
_output_shapes#
!:��������������������
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*5
_output_shapes#
!:�������������������*
T0g
addAddV2inputsdropout/dropout/mul_1:z:0*
T0*-
_output_shapes
:�����������|
2layer_normalization/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:�
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:�����������
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*-
_output_shapes
:�����������*
T0�
6layer_normalization/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(*
T0h
#layer_normalization/batchnorm/add/yConst*
valueB
 *�7�5*
dtype0*
_output_shapes
: �
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*,
_output_shapes
:����������*
T0�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*,
_output_shapes
:����������*
T0�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��k
!sequential/dense_4/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:r
!sequential/dense_4/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       y
"sequential/dense_4/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0l
"sequential/dense_4/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
T0*
N*
_output_shapes
:�
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
&sequential/dense_4/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0~
-sequential/dense_4/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
(sequential/dense_4/Tensordot/transpose_1	Transpose3sequential/dense_4/Tensordot/ReadVariableOp:value:06sequential/dense_4/Tensordot/transpose_1/perm:output:0* 
_output_shapes
:
��*
T0}
,sequential/dense_4/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
&sequential/dense_4/Tensordot/Reshape_1Reshape,sequential/dense_4/Tensordot/transpose_1:y:05sequential/dense_4/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:0/sequential/dense_4/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������o
$sequential/dense_4/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:l
*sequential/dense_4/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*-
_output_shapes
:�����������*
T0�
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������|
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*-
_output_shapes
:������������
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
��k
!sequential/dense_5/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:r
!sequential/dense_5/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       w
"sequential/dense_5/Tensordot/ShapeShape%sequential/dense_4/Relu:activations:0*
T0*
_output_shapes
:l
*sequential/dense_5/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0l
"sequential/dense_5/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
_output_shapes
: *
T0n
$sequential/dense_5/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:�
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_5/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: �
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
_output_shapes
:*
T0�
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
T0*
N*
_output_shapes
:�
&sequential/dense_5/Tensordot/transpose	Transpose%sequential/dense_4/Relu:activations:0,sequential/dense_5/Tensordot/concat:output:0*
T0*-
_output_shapes
:������������
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*0
_output_shapes
:������������������*
T0~
-sequential/dense_5/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:�
(sequential/dense_5/Tensordot/transpose_1	Transpose3sequential/dense_5/Tensordot/ReadVariableOp:value:06sequential/dense_5/Tensordot/transpose_1/perm:output:0*
T0* 
_output_shapes
:
��}
,sequential/dense_5/Tensordot/Reshape_1/shapeConst*
valueB"�   �   *
dtype0*
_output_shapes
:�
&sequential/dense_5/Tensordot/Reshape_1Reshape,sequential/dense_5/Tensordot/transpose_1:y:05sequential/dense_5/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:0/sequential/dense_5/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������o
$sequential/dense_5/Tensordot/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:l
*sequential/dense_5/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: �
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
T0*
N*
_output_shapes
:�
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:������������
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������[
dropout_1/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *���=j
dropout_1/dropout/ShapeShape#sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_1/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
dtype0*-
_output_shapes
:������������
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*-
_output_shapes
:�����������*
T0�
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*-
_output_shapes
:�����������\
dropout_1/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_1/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*-
_output_shapes
:������������
dropout_1/dropout/mulMul#sequential/dense_5/BiasAdd:output:0dropout_1/dropout/truediv:z:0*
T0*-
_output_shapes
:������������
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*-
_output_shapes
:������������
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:������������
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/mul_1:z:0*
T0*-
_output_shapes
:�����������~
4layer_normalization_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:�
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(*
T0�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:�����������
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*-
_output_shapes
:������������
8layer_normalization_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *�7�5�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:�����������
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:�����������
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*-
_output_shapes
:������������
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:��
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:������������
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*-
_output_shapes
:������������
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp7^multi_head_self_attention/dense/BiasAdd/ReadVariableOp9^multi_head_self_attention/dense/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_1/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_2/Tensordot/ReadVariableOp9^multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp;^multi_head_self_attention/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp,^sequential/dense_5/Tensordot/ReadVariableOp*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*l
_input_shapes[
Y:�����������::::::::::::::::2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2t
8multi_head_self_attention/dense/Tensordot/ReadVariableOp8multi_head_self_attention/dense/Tensordot/ReadVariableOp2x
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:multi_head_self_attention/dense_2/Tensordot/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2t
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp2x
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:multi_head_self_attention/dense_3/Tensordot/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp2t
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_5/Tensordot/ReadVariableOp+sequential/dense_5/Tensordot/ReadVariableOp2p
6multi_head_self_attention/dense/BiasAdd/ReadVariableOp6multi_head_self_attention/dense/BiasAdd/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2x
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:multi_head_self_attention/dense_1/Tensordot/ReadVariableOp: : : : : : : :	 :
 : : : : : : :& "
 
_user_specified_nameinputs: 
�	
�
B__inference_dense_7_layer_call_and_return_conditional_losses_69807

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	�i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
��
�0
__inference__traced_save_70313
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopP
Lsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopR
Nsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_dense_kernel_read_readvariableopU
Qsavev2_transformer_block_multi_head_self_attention_dense_bias_read_readvariableopY
Usavev2_transformer_block_multi_head_self_attention_dense_1_kernel_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_dense_1_bias_read_readvariableopY
Usavev2_transformer_block_multi_head_self_attention_dense_2_kernel_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_dense_2_bias_read_readvariableopY
Usavev2_transformer_block_multi_head_self_attention_dense_3_kernel_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_dense_3_bias_read_readvariableopJ
Fsavev2_transformer_block_sequential_dense_4_kernel_read_readvariableopH
Dsavev2_transformer_block_sequential_dense_4_bias_read_readvariableopJ
Fsavev2_transformer_block_sequential_dense_5_kernel_read_readvariableopH
Dsavev2_transformer_block_sequential_dense_5_bias_read_readvariableopJ
Fsavev2_transformer_block_layer_normalization_gamma_read_readvariableopI
Esavev2_transformer_block_layer_normalization_beta_read_readvariableopL
Hsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopK
Gsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableopW
Ssavev2_adam_token_and_position_embedding_embedding_embeddings_m_read_readvariableopY
Usavev2_adam_token_and_position_embedding_embedding_1_embeddings_m_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_block_multi_head_self_attention_dense_bias_m_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_1_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_1_bias_m_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_2_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_2_bias_m_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_3_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_3_bias_m_read_readvariableopQ
Msavev2_adam_transformer_block_sequential_dense_4_kernel_m_read_readvariableopO
Ksavev2_adam_transformer_block_sequential_dense_4_bias_m_read_readvariableopQ
Msavev2_adam_transformer_block_sequential_dense_5_kernel_m_read_readvariableopO
Ksavev2_adam_transformer_block_sequential_dense_5_bias_m_read_readvariableopQ
Msavev2_adam_transformer_block_layer_normalization_gamma_m_read_readvariableopP
Lsavev2_adam_transformer_block_layer_normalization_beta_m_read_readvariableopS
Osavev2_adam_transformer_block_layer_normalization_1_gamma_m_read_readvariableopR
Nsavev2_adam_transformer_block_layer_normalization_1_beta_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableopW
Ssavev2_adam_token_and_position_embedding_embedding_embeddings_v_read_readvariableopY
Usavev2_adam_token_and_position_embedding_embedding_1_embeddings_v_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_block_multi_head_self_attention_dense_bias_v_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_1_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_1_bias_v_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_2_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_2_bias_v_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_3_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_3_bias_v_read_readvariableopQ
Msavev2_adam_transformer_block_sequential_dense_4_kernel_v_read_readvariableopO
Ksavev2_adam_transformer_block_sequential_dense_4_bias_v_read_readvariableopQ
Msavev2_adam_transformer_block_sequential_dense_5_kernel_v_read_readvariableopO
Ksavev2_adam_transformer_block_sequential_dense_5_bias_v_read_readvariableopQ
Msavev2_adam_transformer_block_layer_normalization_gamma_v_read_readvariableopP
Lsavev2_adam_transformer_block_layer_normalization_beta_v_read_readvariableopS
Osavev2_adam_transformer_block_layer_normalization_1_gamma_v_read_readvariableopR
Nsavev2_adam_transformer_block_layer_normalization_1_beta_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_69f124632b604522aa06b01ce4a5f3d8/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �+
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:Q*�+
value�+B�+QB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE�
SaveV2/shape_and_slicesConst"/device:CPU:0*�
value�B�QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:Q�/
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopLsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopNsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_dense_kernel_read_readvariableopQsavev2_transformer_block_multi_head_self_attention_dense_bias_read_readvariableopUsavev2_transformer_block_multi_head_self_attention_dense_1_kernel_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_dense_1_bias_read_readvariableopUsavev2_transformer_block_multi_head_self_attention_dense_2_kernel_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_dense_2_bias_read_readvariableopUsavev2_transformer_block_multi_head_self_attention_dense_3_kernel_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_dense_3_bias_read_readvariableopFsavev2_transformer_block_sequential_dense_4_kernel_read_readvariableopDsavev2_transformer_block_sequential_dense_4_bias_read_readvariableopFsavev2_transformer_block_sequential_dense_5_kernel_read_readvariableopDsavev2_transformer_block_sequential_dense_5_bias_read_readvariableopFsavev2_transformer_block_layer_normalization_gamma_read_readvariableopEsavev2_transformer_block_layer_normalization_beta_read_readvariableopHsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopGsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableopSsavev2_adam_token_and_position_embedding_embedding_embeddings_m_read_readvariableopUsavev2_adam_token_and_position_embedding_embedding_1_embeddings_m_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_kernel_m_read_readvariableopXsavev2_adam_transformer_block_multi_head_self_attention_dense_bias_m_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_1_kernel_m_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_1_bias_m_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_2_kernel_m_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_2_bias_m_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_3_kernel_m_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_3_bias_m_read_readvariableopMsavev2_adam_transformer_block_sequential_dense_4_kernel_m_read_readvariableopKsavev2_adam_transformer_block_sequential_dense_4_bias_m_read_readvariableopMsavev2_adam_transformer_block_sequential_dense_5_kernel_m_read_readvariableopKsavev2_adam_transformer_block_sequential_dense_5_bias_m_read_readvariableopMsavev2_adam_transformer_block_layer_normalization_gamma_m_read_readvariableopLsavev2_adam_transformer_block_layer_normalization_beta_m_read_readvariableopOsavev2_adam_transformer_block_layer_normalization_1_gamma_m_read_readvariableopNsavev2_adam_transformer_block_layer_normalization_1_beta_m_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopSsavev2_adam_token_and_position_embedding_embedding_embeddings_v_read_readvariableopUsavev2_adam_token_and_position_embedding_embedding_1_embeddings_v_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_kernel_v_read_readvariableopXsavev2_adam_transformer_block_multi_head_self_attention_dense_bias_v_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_1_kernel_v_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_1_bias_v_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_2_kernel_v_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_2_bias_v_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_3_kernel_v_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_3_bias_v_read_readvariableopMsavev2_adam_transformer_block_sequential_dense_4_kernel_v_read_readvariableopKsavev2_adam_transformer_block_sequential_dense_4_bias_v_read_readvariableopMsavev2_adam_transformer_block_sequential_dense_5_kernel_v_read_readvariableopKsavev2_adam_transformer_block_sequential_dense_5_bias_v_read_readvariableopMsavev2_adam_transformer_block_layer_normalization_gamma_v_read_readvariableopLsavev2_adam_transformer_block_layer_normalization_beta_v_read_readvariableopOsavev2_adam_transformer_block_layer_normalization_1_gamma_v_read_readvariableopNsavev2_adam_transformer_block_layer_normalization_1_beta_v_read_readvariableop"/device:CPU:0*_
dtypesU
S2Q	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:�:�:
��:�:	�:: : : : : :	�:
��:
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:�:�:�:�: : :�:�:
��:�:	�::	�:
��:
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:�:�:�:�:�:�:
��:�:	�::	�:
��:
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:�:�:�:�: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :M :N :O :P :Q :R 
�
�
*__inference_sequential_layer_call_fn_69962

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-66812*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_66811*
Tout
2*-
config_proto

GPU

CPU2*0J 8*-
_output_shapes
:�����������*
Tin	
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������"
identityIdentity:output:0*<
_input_shapes+
):�����������::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
�
�
%__inference_model_layer_call_fn_68042
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26*,
_gradient_op_typePartitionedCall-68013*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_68012*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:���������*&
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*�
_input_shapes~
|:����������::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
�
�
3__inference_batch_normalization_layer_call_fn_69652

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-66931*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_66930*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*5
_output_shapes#
!:��������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*D
_input_shapes3
1:�������������������::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
<
input_11
serving_default_input_1:0����������;
dense_70
StatefulPartitionedCall:0���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model"}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "input_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 3273], "config": {"batch_input_shape": [null, 3273], "dtype": "float32", "sparse": false, "name": "input_1"}}
�
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�
att
ffn

layernorm1

layernorm2
dropout1
dropout2
trainable_variables
 regularization_losses
!	variables
"	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TransformerBlock", "name": "transformer_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(trainable_variables
)regularization_losses
*	variables
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 192}}}}
�
,trainable_variables
-regularization_losses
.	variables
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
0trainable_variables
1regularization_losses
2	variables
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�

4kernel
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}}}
�

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}}
�
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate$m�%m�4m�5m�:m�;m�Em�Fm�Gm�Hm�Im�Jm�Km�Lm�Mm�Nm�Om�Pm�Qm�Rm�Sm�Tm�Um�Vm�$v�%v�4v�5v�:v�;v�Ev�Fv�Gv�Hv�Iv�Jv�Kv�Lv�Mv�Nv�Ov�Pv�Qv�Rv�Sv�Tv�Uv�Vv�"
	optimizer
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
$18
%19
420
521
:22
;23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
$18
%19
&20
'21
422
523
:24
;25"
trackable_list_wrapper
�
Wlayer_regularization_losses

trainable_variables
regularization_losses
Xmetrics

Ylayers
	variables
Znon_trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
[layer_regularization_losses
trainable_variables
\metrics

]layers
regularization_losses
	variables
^non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
E
embeddings
_trainable_variables
`regularization_losses
a	variables
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, null], "config": {"name": "embedding", "trainable": true, "batch_input_shape": [null, null], "dtype": "float32", "input_dim": 30, "output_dim": 192, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}
�
F
embeddings
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, null], "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": [null, null], "dtype": "float32", "input_dim": 3273, "output_dim": 192, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}}
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
�
glayer_regularization_losses
trainable_variables
hmetrics

ilayers
regularization_losses
	variables
jnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
kquery_dense
l	key_dense
mvalue_dense
ncombine_heads
otrainable_variables
pregularization_losses
q	variables
r	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MultiHeadSelfAttention", "name": "multi_head_self_attention", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�
slayer-0
tlayer-1
utrainable_variables
vregularization_losses
w	variables
x	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 192, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 192, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
�
yaxis
	Sgamma
Tbeta
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}
�
~axis
	Ugamma
Vbeta
trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LayerNormalization", "name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15"
trackable_list_wrapper
�
 �layer_regularization_losses
trainable_variables
�metrics
�layers
 regularization_losses
!	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&�2batch_normalization/gamma
':%�2batch_normalization/beta
0:.� (2batch_normalization/moving_mean
4:2� (2#batch_normalization/moving_variance
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
$0
%1
&2
'3"
trackable_list_wrapper
�
 �layer_regularization_losses
(trainable_variables
�metrics
�layers
)regularization_losses
*	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
,trainable_variables
�metrics
�layers
-regularization_losses
.	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
0trainable_variables
�metrics
�layers
1regularization_losses
2	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 
��2dense_6/kernel
:�2dense_6/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
 �layer_regularization_losses
6trainable_variables
�metrics
�layers
7regularization_losses
8	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_7/kernel
:2dense_7/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
�
 �layer_regularization_losses
<trainable_variables
�metrics
�layers
=regularization_losses
>	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
D:B	�21token_and_position_embedding/embedding/embeddings
G:E
��23token_and_position_embedding/embedding_1/embeddings
L:J
��28transformer_block/multi_head_self_attention/dense/kernel
E:C�26transformer_block/multi_head_self_attention/dense/bias
N:L
��2:transformer_block/multi_head_self_attention/dense_1/kernel
G:E�28transformer_block/multi_head_self_attention/dense_1/bias
N:L
��2:transformer_block/multi_head_self_attention/dense_2/kernel
G:E�28transformer_block/multi_head_self_attention/dense_2/bias
N:L
��2:transformer_block/multi_head_self_attention/dense_3/kernel
G:E�28transformer_block/multi_head_self_attention/dense_3/bias
?:=
��2+transformer_block/sequential/dense_4/kernel
8:6�2)transformer_block/sequential/dense_4/bias
?:=
��2+transformer_block/sequential/dense_5/kernel
8:6�2)transformer_block/sequential/dense_5/bias
::8�2+transformer_block/layer_normalization/gamma
9:7�2*transformer_block/layer_normalization/beta
<::�2-transformer_block/layer_normalization_1/gamma
;:9�2,transformer_block/layer_normalization_1/beta
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
�
 �layer_regularization_losses
_trainable_variables
�metrics
�layers
`regularization_losses
a	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
F0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
F0"
trackable_list_wrapper
�
 �layer_regularization_losses
ctrainable_variables
�metrics
�layers
dregularization_losses
e	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Gkernel
Hbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 192, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}}}
�

Ikernel
Jbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 192, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}}}
�

Kkernel
Lbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 192, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}}}
�

Mkernel
Nbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 192, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}}}
X
G0
H1
I2
J3
K4
L5
M6
N7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
G0
H1
I2
J3
K4
L5
M6
N7"
trackable_list_wrapper
�
 �layer_regularization_losses
otrainable_variables
�metrics
�layers
pregularization_losses
q	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Okernel
Pbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}}}
�

Qkernel
Rbias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 192, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
<
O0
P1
Q2
R3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
O0
P1
Q2
R3"
trackable_list_wrapper
�
 �layer_regularization_losses
utrainable_variables
vregularization_losses
�metrics
�layers
w	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
�
 �layer_regularization_losses
ztrainable_variables
�metrics
�layers
{regularization_losses
|	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
�
 �layer_regularization_losses
trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�
_fn_kwargs
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
k0
l1
m2
n3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�metrics
�layers
�regularization_losses
�	variables
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
-:+�2 Adam/batch_normalization/gamma/m
,:*�2Adam/batch_normalization/beta/m
':%
��2Adam/dense_6/kernel/m
 :�2Adam/dense_6/bias/m
&:$	�2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
I:G	�28Adam/token_and_position_embedding/embedding/embeddings/m
L:J
��2:Adam/token_and_position_embedding/embedding_1/embeddings/m
Q:O
��2?Adam/transformer_block/multi_head_self_attention/dense/kernel/m
J:H�2=Adam/transformer_block/multi_head_self_attention/dense/bias/m
S:Q
��2AAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m
L:J�2?Adam/transformer_block/multi_head_self_attention/dense_1/bias/m
S:Q
��2AAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m
L:J�2?Adam/transformer_block/multi_head_self_attention/dense_2/bias/m
S:Q
��2AAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m
L:J�2?Adam/transformer_block/multi_head_self_attention/dense_3/bias/m
D:B
��22Adam/transformer_block/sequential/dense_4/kernel/m
=:;�20Adam/transformer_block/sequential/dense_4/bias/m
D:B
��22Adam/transformer_block/sequential/dense_5/kernel/m
=:;�20Adam/transformer_block/sequential/dense_5/bias/m
?:=�22Adam/transformer_block/layer_normalization/gamma/m
>:<�21Adam/transformer_block/layer_normalization/beta/m
A:?�24Adam/transformer_block/layer_normalization_1/gamma/m
@:>�23Adam/transformer_block/layer_normalization_1/beta/m
-:+�2 Adam/batch_normalization/gamma/v
,:*�2Adam/batch_normalization/beta/v
':%
��2Adam/dense_6/kernel/v
 :�2Adam/dense_6/bias/v
&:$	�2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
I:G	�28Adam/token_and_position_embedding/embedding/embeddings/v
L:J
��2:Adam/token_and_position_embedding/embedding_1/embeddings/v
Q:O
��2?Adam/transformer_block/multi_head_self_attention/dense/kernel/v
J:H�2=Adam/transformer_block/multi_head_self_attention/dense/bias/v
S:Q
��2AAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v
L:J�2?Adam/transformer_block/multi_head_self_attention/dense_1/bias/v
S:Q
��2AAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v
L:J�2?Adam/transformer_block/multi_head_self_attention/dense_2/bias/v
S:Q
��2AAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v
L:J�2?Adam/transformer_block/multi_head_self_attention/dense_3/bias/v
D:B
��22Adam/transformer_block/sequential/dense_4/kernel/v
=:;�20Adam/transformer_block/sequential/dense_4/bias/v
D:B
��22Adam/transformer_block/sequential/dense_5/kernel/v
=:;�20Adam/transformer_block/sequential/dense_5/bias/v
?:=�22Adam/transformer_block/layer_normalization/gamma/v
>:<�21Adam/transformer_block/layer_normalization/beta/v
A:?�24Adam/transformer_block/layer_normalization_1/gamma/v
@:>�23Adam/transformer_block/layer_normalization_1/beta/v
�2�
%__inference_model_layer_call_fn_68869
%__inference_model_layer_call_fn_68042
%__inference_model_layer_call_fn_67971
%__inference_model_layer_call_fn_68900�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_66654�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *'�$
"�
input_1����������
�2�
@__inference_model_layer_call_and_return_conditional_losses_68512
@__inference_model_layer_call_and_return_conditional_losses_67862
@__inference_model_layer_call_and_return_conditional_losses_68838
@__inference_model_layer_call_and_return_conditional_losses_67901�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
<__inference_token_and_position_embedding_layer_call_fn_68933�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_68926�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_transformer_block_layer_call_fn_69520
1__inference_transformer_block_layer_call_fn_69541�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_transformer_block_layer_call_and_return_conditional_losses_69499
L__inference_transformer_block_layer_call_and_return_conditional_losses_69231�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
3__inference_batch_normalization_layer_call_fn_69743
3__inference_batch_normalization_layer_call_fn_69652
3__inference_batch_normalization_layer_call_fn_69661
3__inference_batch_normalization_layer_call_fn_69734�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69620
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69643
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69702
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69725�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_global_max_pooling1d_layer_call_fn_66989�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_66980�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
)__inference_dropout_2_layer_call_fn_69773
)__inference_dropout_2_layer_call_fn_69778�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dropout_2_layer_call_and_return_conditional_losses_69763
D__inference_dropout_2_layer_call_and_return_conditional_losses_69768�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_dense_6_layer_call_fn_69796�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_6_layer_call_and_return_conditional_losses_69789�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_7_layer_call_fn_69814�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_7_layer_call_and_return_conditional_losses_69807�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2B0
#__inference_signature_wrapper_68121input_1
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_sequential_layer_call_fn_66819
*__inference_sequential_layer_call_fn_69962
*__inference_sequential_layer_call_fn_69953
*__inference_sequential_layer_call_fn_66797�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_sequential_layer_call_and_return_conditional_losses_69879
E__inference_sequential_layer_call_and_return_conditional_losses_66764
E__inference_sequential_layer_call_and_return_conditional_losses_66776
E__inference_sequential_layer_call_and_return_conditional_losses_69944�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_4_layer_call_fn_70004�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_4_layer_call_and_return_conditional_losses_69997�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_5_layer_call_fn_70045�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_5_layer_call_and_return_conditional_losses_70038�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
 __inference__wrapped_model_66654�FEGHIJKLMNSTOPQRUV'$&%45:;1�.
'�$
"�
input_1����������
� "1�.
,
dense_7!�
dense_7����������
%__inference_model_layer_call_fn_68900pFEGHIJKLMNSTOPQRUV'$&%45:;8�5
.�+
!�
inputs����������
p 

 
� "�����������
D__inference_dropout_2_layer_call_and_return_conditional_losses_69763^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
L__inference_transformer_block_layer_call_and_return_conditional_losses_69499zGHIJKLMNSTOPQRUV9�6
/�,
&�#
inputs�����������
p 
� "+�(
!�
0�����������
� �
1__inference_transformer_block_layer_call_fn_69520mGHIJKLMNSTOPQRUV9�6
/�,
&�#
inputs�����������
p
� "������������{
'__inference_dense_7_layer_call_fn_69814P:;0�-
&�#
!�
inputs����������
� "�����������
D__inference_dropout_2_layer_call_and_return_conditional_losses_69768^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_69879rOPQR=�:
3�0
&�#
inputs�����������
p

 
� "+�(
!�
0�����������
� �
B__inference_dense_5_layer_call_and_return_conditional_losses_70038hQR5�2
+�(
&�#
inputs�����������
� "+�(
!�
0�����������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_69944rOPQR=�:
3�0
&�#
inputs�����������
p 

 
� "+�(
!�
0�����������
� �
3__inference_batch_normalization_layer_call_fn_69652q&'$%A�>
7�4
.�+
inputs�������������������
p
� "&�#��������������������
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69620~&'$%A�>
7�4
.�+
inputs�������������������
p
� "3�0
)�&
0�������������������
� �
*__inference_sequential_layer_call_fn_66819fOPQR>�;
4�1
'�$
input_1�����������
p 

 
� "�������������
1__inference_transformer_block_layer_call_fn_69541mGHIJKLMNSTOPQRUV9�6
/�,
&�#
inputs�����������
p 
� "�������������
%__inference_model_layer_call_fn_68869pFEGHIJKLMNSTOPQRUV&'$%45:;8�5
.�+
!�
inputs����������
p

 
� "�����������
3__inference_batch_normalization_layer_call_fn_69661q'$&%A�>
7�4
.�+
inputs�������������������
p 
� "&�#��������������������
@__inference_model_layer_call_and_return_conditional_losses_68512}FEGHIJKLMNSTOPQRUV&'$%45:;8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������
� �
4__inference_global_max_pooling1d_layer_call_fn_66989jE�B
;�8
6�3
inputs'���������������������������
� "!��������������������
B__inference_dense_7_layer_call_and_return_conditional_losses_69807]:;0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
#__inference_signature_wrapper_68121�FEGHIJKLMNSTOPQRUV'$&%45:;<�9
� 
2�/
-
input_1"�
input_1����������"1�.
,
dense_7!�
dense_7����������
B__inference_dense_6_layer_call_and_return_conditional_losses_69789^450�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69643~'$&%A�>
7�4
.�+
inputs�������������������
p 
� "3�0
)�&
0�������������������
� �
*__inference_sequential_layer_call_fn_66797fOPQR>�;
4�1
'�$
input_1�����������
p

 
� "�������������
<__inference_token_and_position_embedding_layer_call_fn_68933QFE+�(
!�
�
x����������
� "�������������
3__inference_batch_normalization_layer_call_fn_69734a&'$%9�6
/�,
&�#
inputs�����������
p
� "�������������
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69702n&'$%9�6
/�,
&�#
inputs�����������
p
� "+�(
!�
0�����������
� �
3__inference_batch_normalization_layer_call_fn_69743a'$&%9�6
/�,
&�#
inputs�����������
p 
� "������������|
'__inference_dense_6_layer_call_fn_69796Q450�-
&�#
!�
inputs����������
� "������������
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_68926^FE+�(
!�
�
x����������
� "+�(
!�
0�����������
� �
%__inference_model_layer_call_fn_67971qFEGHIJKLMNSTOPQRUV&'$%45:;9�6
/�,
"�
input_1����������
p

 
� "�����������
N__inference_batch_normalization_layer_call_and_return_conditional_losses_69725n'$&%9�6
/�,
&�#
inputs�����������
p 
� "+�(
!�
0�����������
� �
L__inference_transformer_block_layer_call_and_return_conditional_losses_69231zGHIJKLMNSTOPQRUV9�6
/�,
&�#
inputs�����������
p
� "+�(
!�
0�����������
� �
@__inference_model_layer_call_and_return_conditional_losses_68838}FEGHIJKLMNSTOPQRUV'$&%45:;8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������
� �
*__inference_sequential_layer_call_fn_69953eOPQR=�:
3�0
&�#
inputs�����������
p

 
� "�������������
E__inference_sequential_layer_call_and_return_conditional_losses_66764sOPQR>�;
4�1
'�$
input_1�����������
p

 
� "+�(
!�
0�����������
� ~
)__inference_dropout_2_layer_call_fn_69773Q4�1
*�'
!�
inputs����������
p
� "������������
*__inference_sequential_layer_call_fn_69962eOPQR=�:
3�0
&�#
inputs�����������
p 

 
� "�������������
B__inference_dense_4_layer_call_and_return_conditional_losses_69997hOP5�2
+�(
&�#
inputs�����������
� "+�(
!�
0�����������
� �
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_66980wE�B
;�8
6�3
inputs'���������������������������
� ".�+
$�!
0������������������
� �
@__inference_model_layer_call_and_return_conditional_losses_67901~FEGHIJKLMNSTOPQRUV'$&%45:;9�6
/�,
"�
input_1����������
p 

 
� "%�"
�
0���������
� �
'__inference_dense_4_layer_call_fn_70004[OP5�2
+�(
&�#
inputs�����������
� "������������~
)__inference_dropout_2_layer_call_fn_69778Q4�1
*�'
!�
inputs����������
p 
� "������������
E__inference_sequential_layer_call_and_return_conditional_losses_66776sOPQR>�;
4�1
'�$
input_1�����������
p 

 
� "+�(
!�
0�����������
� �
%__inference_model_layer_call_fn_68042qFEGHIJKLMNSTOPQRUV'$&%45:;9�6
/�,
"�
input_1����������
p 

 
� "�����������
@__inference_model_layer_call_and_return_conditional_losses_67862~FEGHIJKLMNSTOPQRUV&'$%45:;9�6
/�,
"�
input_1����������
p

 
� "%�"
�
0���������
� �
'__inference_dense_5_layer_call_fn_70045[QR5�2
+�(
&�#
inputs�����������
� "������������