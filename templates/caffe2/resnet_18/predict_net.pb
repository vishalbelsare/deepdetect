
 =
data
conv1_wconv1"Conv*

stride*
pad*

kernel|
conv1
conv1_scale

conv1_bias

conv1_mean
	conv1_varconv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWL
conv1_unique
conv1_w_secondconv1_internal"Mul*
axis*
	broadcast>
conv1_internal
conv1_bconv1"Add*
axis*
	broadcast
conv1conv1"ReluW
conv1pool1"MaxPool*

stride*
pad*

kernel*
order"NCHW*

legacy_padP
pool1
res2a_branch2a_wres2a_branch2a"Conv*

stride*
pad*

kernel�
res2a_branch2a
res2a_branch2a_scale
res2a_branch2a_bias
res2a_branch2a_mean
res2a_branch2a_varres2a_branch2a_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res2a_branch2a_unique
res2a_branch2a_w_secondres2a_branch2a_internal"Mul*
axis*
	broadcastY
res2a_branch2a_internal
res2a_branch2a_bres2a_branch2a"Add*
axis*
	broadcast&
res2a_branch2ares2a_branch2a"ReluY
res2a_branch2a
res2a_branch2b_wres2a_branch2b"Conv*

stride*
pad*

kernel�
res2a_branch2b
res2a_branch2b_scale
res2a_branch2b_bias
res2a_branch2b_mean
res2a_branch2b_varres2a_branch2b_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res2a_branch2b_unique
res2a_branch2b_w_secondres2a_branch2b_internal"Mul*
axis*
	broadcastY
res2a_branch2b_internal
res2a_branch2b_bres2a_branch2b"Add*
axis*
	broadcast#
pool1
res2a_branch2bres2a"Sum
res2ares2a"ReluP
res2a
res2b_branch2a_wres2b_branch2a"Conv*

stride*
pad*

kernel�
res2b_branch2a
res2b_branch2a_scale
res2b_branch2a_bias
res2b_branch2a_mean
res2b_branch2a_varres2b_branch2a_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res2b_branch2a_unique
res2b_branch2a_w_secondres2b_branch2a_internal"Mul*
axis*
	broadcastY
res2b_branch2a_internal
res2b_branch2a_bres2b_branch2a"Add*
axis*
	broadcast&
res2b_branch2ares2b_branch2a"ReluY
res2b_branch2a
res2b_branch2b_wres2b_branch2b"Conv*

stride*
pad*

kernel�
res2b_branch2b
res2b_branch2b_scale
res2b_branch2b_bias
res2b_branch2b_mean
res2b_branch2b_varres2b_branch2b_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res2b_branch2b_unique
res2b_branch2b_w_secondres2b_branch2b_internal"Mul*
axis*
	broadcastY
res2b_branch2b_internal
res2b_branch2b_bres2b_branch2b"Add*
axis*
	broadcast#
res2a
res2b_branch2bres2b"Sum
res2bres2b"ReluN
res2b
res3a_branch1_wres3a_branch1"Conv*

stride*
pad *

kernel�
res3a_branch1
res3a_branch1_scale
res3a_branch1_bias
res3a_branch1_mean
res3a_branch1_varres3a_branch1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWd
res3a_branch1_unique
res3a_branch1_w_secondres3a_branch1_internal"Mul*
axis*
	broadcastV
res3a_branch1_internal
res3a_branch1_bres3a_branch1"Add*
axis*
	broadcastP
res2b
res3a_branch2a_wres3a_branch2a"Conv*

stride*
pad*

kernel�
res3a_branch2a
res3a_branch2a_scale
res3a_branch2a_bias
res3a_branch2a_mean
res3a_branch2a_varres3a_branch2a_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res3a_branch2a_unique
res3a_branch2a_w_secondres3a_branch2a_internal"Mul*
axis*
	broadcastY
res3a_branch2a_internal
res3a_branch2a_bres3a_branch2a"Add*
axis*
	broadcast&
res3a_branch2ares3a_branch2a"ReluY
res3a_branch2a
res3a_branch2b_wres3a_branch2b"Conv*

stride*
pad*

kernel�
res3a_branch2b
res3a_branch2b_scale
res3a_branch2b_bias
res3a_branch2b_mean
res3a_branch2b_varres3a_branch2b_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res3a_branch2b_unique
res3a_branch2b_w_secondres3a_branch2b_internal"Mul*
axis*
	broadcastY
res3a_branch2b_internal
res3a_branch2b_bres3a_branch2b"Add*
axis*
	broadcast+
res3a_branch1
res3a_branch2bres3a"Sum
res3ares3a"ReluP
res3a
res3b_branch2a_wres3b_branch2a"Conv*

stride*
pad*

kernel�
res3b_branch2a
res3b_branch2a_scale
res3b_branch2a_bias
res3b_branch2a_mean
res3b_branch2a_varres3b_branch2a_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res3b_branch2a_unique
res3b_branch2a_w_secondres3b_branch2a_internal"Mul*
axis*
	broadcastY
res3b_branch2a_internal
res3b_branch2a_bres3b_branch2a"Add*
axis*
	broadcast&
res3b_branch2ares3b_branch2a"ReluY
res3b_branch2a
res3b_branch2b_wres3b_branch2b"Conv*

stride*
pad*

kernel�
res3b_branch2b
res3b_branch2b_scale
res3b_branch2b_bias
res3b_branch2b_mean
res3b_branch2b_varres3b_branch2b_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res3b_branch2b_unique
res3b_branch2b_w_secondres3b_branch2b_internal"Mul*
axis*
	broadcastY
res3b_branch2b_internal
res3b_branch2b_bres3b_branch2b"Add*
axis*
	broadcast#
res3a
res3b_branch2bres3b"Sum
res3bres3b"ReluN
res3b
res4a_branch1_wres4a_branch1"Conv*

stride*
pad *

kernel�
res4a_branch1
res4a_branch1_scale
res4a_branch1_bias
res4a_branch1_mean
res4a_branch1_varres4a_branch1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWd
res4a_branch1_unique
res4a_branch1_w_secondres4a_branch1_internal"Mul*
axis*
	broadcastV
res4a_branch1_internal
res4a_branch1_bres4a_branch1"Add*
axis*
	broadcastP
res3b
res4a_branch2a_wres4a_branch2a"Conv*

stride*
pad*

kernel�
res4a_branch2a
res4a_branch2a_scale
res4a_branch2a_bias
res4a_branch2a_mean
res4a_branch2a_varres4a_branch2a_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res4a_branch2a_unique
res4a_branch2a_w_secondres4a_branch2a_internal"Mul*
axis*
	broadcastY
res4a_branch2a_internal
res4a_branch2a_bres4a_branch2a"Add*
axis*
	broadcast&
res4a_branch2ares4a_branch2a"ReluY
res4a_branch2a
res4a_branch2b_wres4a_branch2b"Conv*

stride*
pad*

kernel�
res4a_branch2b
res4a_branch2b_scale
res4a_branch2b_bias
res4a_branch2b_mean
res4a_branch2b_varres4a_branch2b_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res4a_branch2b_unique
res4a_branch2b_w_secondres4a_branch2b_internal"Mul*
axis*
	broadcastY
res4a_branch2b_internal
res4a_branch2b_bres4a_branch2b"Add*
axis*
	broadcast+
res4a_branch1
res4a_branch2bres4a"Sum
res4ares4a"ReluP
res4a
res4b_branch2a_wres4b_branch2a"Conv*

stride*
pad*

kernel�
res4b_branch2a
res4b_branch2a_scale
res4b_branch2a_bias
res4b_branch2a_mean
res4b_branch2a_varres4b_branch2a_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res4b_branch2a_unique
res4b_branch2a_w_secondres4b_branch2a_internal"Mul*
axis*
	broadcastY
res4b_branch2a_internal
res4b_branch2a_bres4b_branch2a"Add*
axis*
	broadcast&
res4b_branch2ares4b_branch2a"ReluY
res4b_branch2a
res4b_branch2b_wres4b_branch2b"Conv*

stride*
pad*

kernel�
res4b_branch2b
res4b_branch2b_scale
res4b_branch2b_bias
res4b_branch2b_mean
res4b_branch2b_varres4b_branch2b_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res4b_branch2b_unique
res4b_branch2b_w_secondres4b_branch2b_internal"Mul*
axis*
	broadcastY
res4b_branch2b_internal
res4b_branch2b_bres4b_branch2b"Add*
axis*
	broadcast#
res4a
res4b_branch2bres4b"Sum
res4bres4b"ReluN
res4b
res5a_branch1_wres5a_branch1"Conv*

stride*
pad *

kernel�
res5a_branch1
res5a_branch1_scale
res5a_branch1_bias
res5a_branch1_mean
res5a_branch1_varres5a_branch1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWd
res5a_branch1_unique
res5a_branch1_w_secondres5a_branch1_internal"Mul*
axis*
	broadcastV
res5a_branch1_internal
res5a_branch1_bres5a_branch1"Add*
axis*
	broadcastP
res4b
res5a_branch2a_wres5a_branch2a"Conv*

stride*
pad*

kernel�
res5a_branch2a
res5a_branch2a_scale
res5a_branch2a_bias
res5a_branch2a_mean
res5a_branch2a_varres5a_branch2a_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res5a_branch2a_unique
res5a_branch2a_w_secondres5a_branch2a_internal"Mul*
axis*
	broadcastY
res5a_branch2a_internal
res5a_branch2a_bres5a_branch2a"Add*
axis*
	broadcast&
res5a_branch2ares5a_branch2a"ReluY
res5a_branch2a
res5a_branch2b_wres5a_branch2b"Conv*

stride*
pad*

kernel�
res5a_branch2b
res5a_branch2b_scale
res5a_branch2b_bias
res5a_branch2b_mean
res5a_branch2b_varres5a_branch2b_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res5a_branch2b_unique
res5a_branch2b_w_secondres5a_branch2b_internal"Mul*
axis*
	broadcastY
res5a_branch2b_internal
res5a_branch2b_bres5a_branch2b"Add*
axis*
	broadcast+
res5a_branch1
res5a_branch2bres5a"Sum
res5ares5a"ReluP
res5a
res5b_branch2a_wres5b_branch2a"Conv*

stride*
pad*

kernel�
res5b_branch2a
res5b_branch2a_scale
res5b_branch2a_bias
res5b_branch2a_mean
res5b_branch2a_varres5b_branch2a_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res5b_branch2a_unique
res5b_branch2a_w_secondres5b_branch2a_internal"Mul*
axis*
	broadcastY
res5b_branch2a_internal
res5b_branch2a_bres5b_branch2a"Add*
axis*
	broadcast&
res5b_branch2ares5b_branch2a"ReluY
res5b_branch2a
res5b_branch2b_wres5b_branch2b"Conv*

stride*
pad*

kernel�
res5b_branch2b
res5b_branch2b_scale
res5b_branch2b_bias
res5b_branch2b_mean
res5b_branch2b_varres5b_branch2b_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWg
res5b_branch2b_unique
res5b_branch2b_w_secondres5b_branch2b_internal"Mul*
axis*
	broadcastY
res5b_branch2b_internal
res5b_branch2b_bres5b_branch2b"Add*
axis*
	broadcast#
res5a
res5b_branch2bres5b"Sum
res5bres5b"Reluo
res5bpool5"AveragePool*

stride*
pad *

kernel *
order"NCHW*

legacy_pad*
global_pooling'
pool5
fc1000_w
fc1000_bfc1000"FC
fc1000prob"Softmax:data:conv1_w:conv1_scale:
conv1_bias:
conv1_mean:	conv1_var:conv1_w_second:conv1_b:res2a_branch2a_w:res2a_branch2a_scale:res2a_branch2a_bias:res2a_branch2a_mean:res2a_branch2a_var:res2a_branch2a_w_second:res2a_branch2a_b:res2a_branch2b_w:res2a_branch2b_scale:res2a_branch2b_bias:res2a_branch2b_mean:res2a_branch2b_var:res2a_branch2b_w_second:res2a_branch2b_b:res2b_branch2a_w:res2b_branch2a_scale:res2b_branch2a_bias:res2b_branch2a_mean:res2b_branch2a_var:res2b_branch2a_w_second:res2b_branch2a_b:res2b_branch2b_w:res2b_branch2b_scale:res2b_branch2b_bias:res2b_branch2b_mean:res2b_branch2b_var:res2b_branch2b_w_second:res2b_branch2b_b:res3a_branch1_w:res3a_branch1_scale:res3a_branch1_bias:res3a_branch1_mean:res3a_branch1_var:res3a_branch1_w_second:res3a_branch1_b:res3a_branch2a_w:res3a_branch2a_scale:res3a_branch2a_bias:res3a_branch2a_mean:res3a_branch2a_var:res3a_branch2a_w_second:res3a_branch2a_b:res3a_branch2b_w:res3a_branch2b_scale:res3a_branch2b_bias:res3a_branch2b_mean:res3a_branch2b_var:res3a_branch2b_w_second:res3a_branch2b_b:res3b_branch2a_w:res3b_branch2a_scale:res3b_branch2a_bias:res3b_branch2a_mean:res3b_branch2a_var:res3b_branch2a_w_second:res3b_branch2a_b:res3b_branch2b_w:res3b_branch2b_scale:res3b_branch2b_bias:res3b_branch2b_mean:res3b_branch2b_var:res3b_branch2b_w_second:res3b_branch2b_b:res4a_branch1_w:res4a_branch1_scale:res4a_branch1_bias:res4a_branch1_mean:res4a_branch1_var:res4a_branch1_w_second:res4a_branch1_b:res4a_branch2a_w:res4a_branch2a_scale:res4a_branch2a_bias:res4a_branch2a_mean:res4a_branch2a_var:res4a_branch2a_w_second:res4a_branch2a_b:res4a_branch2b_w:res4a_branch2b_scale:res4a_branch2b_bias:res4a_branch2b_mean:res4a_branch2b_var:res4a_branch2b_w_second:res4a_branch2b_b:res4b_branch2a_w:res4b_branch2a_scale:res4b_branch2a_bias:res4b_branch2a_mean:res4b_branch2a_var:res4b_branch2a_w_second:res4b_branch2a_b:res4b_branch2b_w:res4b_branch2b_scale:res4b_branch2b_bias:res4b_branch2b_mean:res4b_branch2b_var:res4b_branch2b_w_second:res4b_branch2b_b:res5a_branch1_w:res5a_branch1_scale:res5a_branch1_bias:res5a_branch1_mean:res5a_branch1_var:res5a_branch1_w_second:res5a_branch1_b:res5a_branch2a_w:res5a_branch2a_scale:res5a_branch2a_bias:res5a_branch2a_mean:res5a_branch2a_var:res5a_branch2a_w_second:res5a_branch2a_b:res5a_branch2b_w:res5a_branch2b_scale:res5a_branch2b_bias:res5a_branch2b_mean:res5a_branch2b_var:res5a_branch2b_w_second:res5a_branch2b_b:res5b_branch2a_w:res5b_branch2a_scale:res5b_branch2a_bias:res5b_branch2a_mean:res5b_branch2a_var:res5b_branch2a_w_second:res5b_branch2a_b:res5b_branch2b_w:res5b_branch2b_scale:res5b_branch2b_bias:res5b_branch2b_mean:res5b_branch2b_var:res5b_branch2b_w_second:res5b_branch2b_b:fc1000_w:fc1000_bBprob