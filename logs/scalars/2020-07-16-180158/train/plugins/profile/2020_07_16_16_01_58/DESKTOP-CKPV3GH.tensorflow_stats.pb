"�e
VHostIDLE"IDLE(133333��@9fffffNp@A33333��@IfffffNp@a?�ڌ��?i?�ڌ��?�Unknown
tHostMatMul" gradient_tape/model/dense/MatMul(1�����Iu@9�����Iu@A�����Iu@I�����Iu@a䯍��a�?i<E�?6��?�Unknown
xHostMatMul"$gradient_tape/model/dense_4/MatMul_1(1fffff�e@9fffff�e@Afffff�e@Ifffff�e@a�1���2�?iZH�ee�?�Unknown
jHost_FusedMatMul"model/dense/Relu(1fffff&e@9fffff&e@Afffff&e@Ifffff&e@a�ƽ
�2�?i�$���(�?�Unknown
vHostMatMul""gradient_tape/model/dense_4/MatMul(1����̬R@9����̬R@A����̬R@I����̬R@a$fH1�?i�����?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1fffffFP@9fffffFP@AfffffFP@IfffffFP@a~���?izG�^��?�Unknown
pHostConcatV2"model/concatenate_1/concat(1�����YB@9�����YB@A�����YB@I�����YB@a{��ECw�?il^�;��?�Unknown
oHost_FusedMatMul"model/dense_4/BiasAdd(133333�@@933333�@@A33333�@@I33333�@@a=pcci��?i���b�Y�?�Unknown
�	HostDataset"=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate(1������7@9������7@A������5@I������5@aG�N�}?ib�
	Ó�?�Unknown
v
HostMatMul""gradient_tape/model/dense_3/MatMul(1ffffff/@9ffffff/@Affffff/@Iffffff/@ab�F��t?i�k�����?�Unknown
dHostDataset"Iterator::Model(13333334@93333334@A      ,@I      ,@aav�؂�r?ivpI����?�Unknown
oHost_FusedMatMul"model/dense_3/BiasAdd(1333333*@9333333*@A333333*@I333333*@a��4Rwq?ib��7��?�Unknown
^HostGatherV2"GatherV2(1������(@9������(@A������(@I������(@aa�jCfp?i򆾰&�?�Unknown
vHostMatMul""gradient_tape/model/dense_1/MatMul(1������&@9������&@A������&@I������&@a*�|�%fn?i�n�E�?�Unknown
xHostMatMul"$gradient_tape/model/dense_3/MatMul_1(1333333&@9333333&@A333333&@I333333&@a�\�tZ�m?i$�>�b�?�Unknown
fHostGreaterEqual"GreaterEqual(1      &@9      &@A      &@I      &@at���Um?i��IU��?�Unknown
vHostMatMul""gradient_tape/model/dense_2/MatMul(1������!@9������!@A������!@I������!@a�w�gEwg?i1鱚|��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_8/ResourceApplyGradientDescent(1ffffff!@9ffffff!@Affffff!@Iffffff!@ab�Q�3g?i�:g����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@aq!SDd?i�o����?�Unknown
lHost_FusedMatMul"model/dense_1/Relu(1������@9������@A������@I������@a��=
3c?iV��&��?�Unknown
�HostDataset"3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat(1333333!@9333333!@Affffff@Iffffff@a�=%���b?iE{8���?�Unknown
qHostDataset"Iterator::Model::ParallelMap(1������@9������@A������@I������@a�<lCe�`?i��{���?�Unknown
xHostMatMul"$gradient_tape/model/dense_2/MatMul_1(1      @9      @A      @I      @a9]M���_?i1Zɝ	�?�Unknown
nHostConcatV2"model/concatenate/concat(1      @9      @A      @I      @a9]M���_?i�48���?�Unknown
gHostStridedSlice"strided_slice(1������@9������@A������@I������@a��W4w_?iǸcAY)�?�Unknown
�HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1������@9������@A������@I������@a*�|�%f^?i �)T�8�?�Unknown
xHostMatMul"$gradient_tape/model/dense_1/MatMul_1(1ffffff@9ffffff@Affffff@Iffffff@a�#7'��]?i��=#{G�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_9/ResourceApplyGradientDescent(1������@9������@A������@I������@a�\��\?i���j�U�?�Unknown
lHost_FusedMatMul"model/dense_2/Relu(1333333@9333333@A333333@I333333@a��
�b�Y?inj�b�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a�:���W?i��n�?�Unknown
vHostDataset"!Iterator::Model::ParallelMap::Zip(1�����C@9�����C@A������@I������@a�w�gEwW?iM��iz�?�Unknown
\ HostArgMax"ArgMax_1(1������@9������@A������@I������@a�Zi�6fV?i�������?�Unknown
`!HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a�Zi�6fV?i�n"�ϐ�?�Unknown
l"HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a�Zi�6fV?iT#q���?�Unknown
i#HostMean"mean_squared_error/Mean(1������@9������@A������@I������@a�Zi�6fV?iؿ6��?�Unknown
t$HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff @Affffff@Iffffff @a��#8��U?i��[�$��?�Unknown
�%HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a&>��'UU?iYE~ϼ�?�Unknown
�&HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @a&>��'UU?i%�.z��?�Unknown
�'HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a&>��'UU?iD7�$��?�Unknown
�(HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a˯�m��T?i�O����?�Unknown
Z)HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@aq!SDT?i--����?�Unknown
X*HostCast"Cast_2(1������@9������@A������@I������@a����S?i���ˊ��?�Unknown
�+HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1������@9������@A������@I������@a����S?i�:v�h��?�Unknown
x,HostSlice"%gradient_tape/model/concatenate/Slice(1������@9������@A������@I������@a����S?i��G]F�?�Unknown
�-HostSquaredDifference"$mean_squared_error/SquaredDifference(1������@9������@A������@I������@a����S?iUH&$�?�Unknown
|.HostSlice")gradient_tape/model/concatenate_1/Slice_1(1������	@9������	@A������	@I������	@aR˱��Q?i;�m���?�Unknown
z/HostReluGrad"$gradient_tape/model/dense_2/ReluGrad(1������@9������@A������@I������@a�<lCe�P?iYW���?�Unknown
�0HostBiasAddGrad"/gradient_tape/model/dense_4/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a�<lCe�P?iw�5'�?�Unknown
�1HostBiasAddGrad"/gradient_tape/model/dense_3/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�@���N?i~���.�?�Unknown
|2HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�#7'��M?i�Kwh6�?�Unknown
�3HostReadVariableOp"$model/dense_2/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�#7'��M?i�|�=�?�Unknown
�4HostReadVariableOp"$model/dense_4/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a�\��L?i�D؟E�?�Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_2(1������@9������@A������@I������@ae� ���K?i���L�?�Unknown
V6HostSum"Sum_2(1������@9������@A������@I������@ae� ���K?iU!`�R�?�Unknown
z7HostSlice"'gradient_tape/model/concatenate/Slice_1(1������@9������@A������@I������@ae� ���K?iL�E@�Y�?�Unknown
x8HostReluGrad""gradient_tape/model/dense/ReluGrad(1      @9      @A      @I      @a�͕�q�J?i�·܉`�?�Unknown
d9HostTanh"model/dense_3/Tanh(1      @9      @A      @I      @a�͕�q�J?i2�)y4g�?�Unknown
�:HostBiasAddGrad"-gradient_tape/model/dense/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a��
�b�I?i���њm�?�Unknown
�;HostReadVariableOp"$model/dense_1/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a�w�gEwG?i��B�xs�?�Unknown
V<HostAddN"AddN(1������ @9������ @A������ @I������ @a�Zi�6fF?iSB�0y�?�Unknown
�=HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������ @9������ @A������ @I������ @a�Zi�6fF?i�����~�?�Unknown
X>HostCast"Cast_3(1       @9       @A       @I       @a&>��'UE?i:T���?�Unknown
�?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a&>��'UE?i�{RV��?�Unknown
z@HostReluGrad"$gradient_tape/model/dense_1/ReluGrad(1       @9       @A       @I       @a&>��'UE?iZ�o����?�Unknown
uAHostSum"$mean_squared_error/weighted_loss/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?aq!SDD?i"ر����?�Unknown
XBHostEqual"Equal(1�������?9�������?A�������?I�������?aR˱��A?i��� ��?�Unknown
�CHostDataset"-Iterator::Model::ParallelMap::Zip[0]::FlatMap(13333339@93333339@A�������?I�������?aR˱��A?i1E��?�Unknown
}DHostMaximum"(gradient_tape/mean_squared_error/Maximum(1�������?9�������?A�������?I�������?aR˱��A?i{]0T���?�Unknown
jEHostSigmoid"model/dense_4/Sigmoid(1�������?9�������?A�������?I�������?aR˱��A?i�Z�ͤ�?�Unknown
�FHostDataset"?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(1      �?9      �?A      �?I      �?a9]M���??i�҆ͨ�?�Unknown
�GHostBiasAddGrad"/gradient_tape/model/dense_2/BiasAdd/BiasAddGrad(1      �?9      �?A      �?I      �?a9]M���??iF�I~ͬ�?�Unknown
�HHostBiasAddGrad"/gradient_tape/model/dense_1/BiasAdd/BiasAddGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�#7'��=?i*�2���?�Unknown
bIHostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?ae� ���;?iG� � ��?�Unknown
uJHostSub"$gradient_tape/mean_squared_error/sub(1�������?9�������?A�������?I�������?ae� ���;?id3x��?�Unknown
�KHostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?ae� ���;?i�PE���?�Unknown
�LHostDataset"LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(1�������?9�������?A�������?I�������?a�w�gEw7?iO�j޽�?�Unknown
TMHostMul"Mul(1�������?9�������?A�������?I�������?a�w�gEw7?i�M�S���?�Unknown
jNHostReadVariableOp"ReadVariableOp(1�������?9�������?A�������?I�������?a�w�gEw7?i.LL<���?�Unknown
sOHostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�w�gEw7?i�J�$���?�Unknown
}PHostRealDiv"(gradient_tape/mean_squared_error/truediv(1�������?9�������?A�������?I�������?a�w�gEw7?iLI����?�Unknown
zQHostTanhGrad"$gradient_tape/model/dense_3/TanhGrad(1�������?9�������?A�������?I�������?a�w�gEw7?i�GS����?�Unknown
uRHostMul"$gradient_tape/mean_squared_error/Mul(1      �?9      �?A      �?I      �?a&>��'U5?i��M�3��?�Unknown
|SHostDivNoNan"&mean_squared_error/weighted_loss/value(1      �?9      �?A      �?I      �?a&>��'U5?ik�G@���?�Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?a��=
33?il���D��?�Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a��=
33?imq����?�Unknown
XVHostCast"Cast_4(1�������?9�������?A�������?I�������?a��=
33?in*d��?�Unknown
wWHostCast"%gradient_tape/mean_squared_error/Cast(1�������?9�������?A�������?I�������?a��=
33?io�f�w��?�Unknown
uXHostSum"$gradient_tape/mean_squared_error/Sum(1�������?9�������?A�������?I�������?a��=
33?ip��&���?�Unknown
YHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1�������?9�������?A�������?I�������?a��=
33?iqU��D��?�Unknown
ZHostReadVariableOp"#model/dense_2/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��=
33?ir>���?�Unknown
[HostReadVariableOp"#model/dense_3/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��=
33?isǅJ��?�Unknown
`\HostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?aR˱��1?i��h3��?�Unknown
�]HostSigmoidGrad"/gradient_tape/model/dense_4/Sigmoid/SigmoidGrad(1�������?9�������?A�������?I�������?aR˱��1?i��U��?�Unknown
}^HostReadVariableOp"!model/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aR˱��1?i
E�w��?�Unknown
_HostReadVariableOp"#model/dense_1/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aR˱��1?iW �����?�Unknown
`HostReadVariableOp"#model/dense_4/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aR˱��1?i�6o޻��?�Unknown
�aHostDataset"MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�#7'��-?i�Q����?�Unknown
ubHostReadVariableOp"div_no_nan/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�#7'��-?it4�w��?�Unknown
wcHostMul"&gradient_tape/mean_squared_error/mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�#7'��-?i�lU��?�Unknown
~dHostReadVariableOp""model/dense/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�#7'��-?iX�E3��?�Unknown
XeHostCast"Cast_1(1333333�?9333333�?A333333�?I333333�?a��
�b�)?i�(����?�Unknown
wfHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a��
�b�)?i��Xrf��?�Unknown
ygHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a��
�b�)?iYv� ��?�Unknown
�hHostReadVariableOp"$model/dense_3/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��
�b�)?iG�����?�Unknown
�iHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a&>��'U%?i�t5����?�Unknown
wjHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?aR˱��!?i     �?�Unknown*�e
tHostMatMul" gradient_tape/model/dense/MatMul(1�����Iu@9�����Iu@A�����Iu@I�����Iu@a|X;�\�?i|X;�\�?�Unknown
xHostMatMul"$gradient_tape/model/dense_4/MatMul_1(1fffff�e@9fffff�e@Afffff�e@Ifffff�e@a(m ? �?i��q||�?�Unknown
jHost_FusedMatMul"model/dense/Relu(1fffff&e@9fffff&e@Afffff&e@Ifffff&e@a���nb�?i�4�Y��?�Unknown
vHostMatMul""gradient_tape/model/dense_4/MatMul(1����̬R@9����̬R@A����̬R@I����̬R@a[��E��?i�{���?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1fffffFP@9fffffFP@AfffffFP@IfffffFP@a�{C>��?iP����/�?�Unknown
pHostConcatV2"model/concatenate_1/concat(1�����YB@9�����YB@A�����YB@I�����YB@a��%�~*�?i�S�� �?�Unknown
oHost_FusedMatMul"model/dense_4/BiasAdd(133333�@@933333�@@A33333�@@I33333�@@af��.Pݛ?i�k�o���?�Unknown
�HostDataset"=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate(1������7@9������7@A������5@I������5@a��_Q4�?i]jT:��?�Unknown
v	HostMatMul""gradient_tape/model/dense_3/MatMul(1ffffff/@9ffffff/@Affffff/@Iffffff/@a_b�Aω?i��-w��?�Unknown
d
HostDataset"Iterator::Model(13333334@93333334@A      ,@I      ,@a�v��?iNbd�R�?�Unknown
oHost_FusedMatMul"model/dense_3/BiasAdd(1333333*@9333333*@A333333*@I333333*@a������?iP�����?�Unknown
^HostGatherV2"GatherV2(1������(@9������(@A������(@I������(@a����b8�?i{g�/���?�Unknown
vHostMatMul""gradient_tape/model/dense_1/MatMul(1������&@9������&@A������&@I������&@aR^�g���?i��!��D�?�Unknown
xHostMatMul"$gradient_tape/model/dense_3/MatMul_1(1333333&@9333333&@A333333&@I333333&@a':��_?�?i�QX/���?�Unknown
fHostGreaterEqual"GreaterEqual(1      &@9      &@A      &@I      &@ao�z�I�?i�;W���?�Unknown
vHostMatMul""gradient_tape/model/dense_2/MatMul(1������!@9������!@A������!@I������!@a�]���|?i)�z��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_8/ResourceApplyGradientDescent(1ffffff!@9ffffff!@Affffff!@Iffffff!@a�1(v��|?i�Ggs�H�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a�}����x?i� '�z�?�Unknown
lHost_FusedMatMul"model/dense_1/Relu(1������@9������@A������@I������@a���(�w?iq:��?�Unknown
�HostDataset"3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat(1333333!@9333333!@Affffff@Iffffff@a�ZѲ�Ww?iͰjl���?�Unknown
qHostDataset"Iterator::Model::ParallelMap(1������@9������@A������@I������@a���xbt?iБ�]��?�Unknown
xHostMatMul"$gradient_tape/model/dense_2/MatMul_1(1      @9      @A      @I      @a���"�s?i�͡#)�?�Unknown
nHostConcatV2"model/concatenate/concat(1      @9      @A      @I      @a���"�s?ij��P�?�Unknown
gHostStridedSlice"strided_slice(1������@9������@A������@I������@a69P��es?i�H��cw�?�Unknown
�HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1������@9������@A������@I������@aR^�g��r?i��ߜ�?�Unknown
xHostMatMul"$gradient_tape/model/dense_1/MatMul_1(1ffffff@9ffffff@Affffff@Iffffff@a��+uir?i{s�����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_9/ResourceApplyGradientDescent(1������@9������@A������@I������@a�E��q?i��T<4��?�Unknown
lHost_FusedMatMul"model/dense_2/Relu(1333333@9333333@A333333@I333333@a�
	�6�o?i��r��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a�y�*3�m?i,��["�?�Unknown
vHostDataset"!Iterator::Model::ParallelMap::Zip(1�����C@9�����C@A������@I������@a�]���l?i�,ƂJ?�?�Unknown
\HostArgMax"ArgMax_1(1������@9������@A������@I������@aQ��/�k?i�����Z�?�Unknown
` HostGatherV2"
GatherV2_1(1������@9������@A������@I������@aQ��/�k?i�<I�v�?�Unknown
l!HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@aQ��/�k?i��
%��?�Unknown
i"HostMean"mean_squared_error/Mean(1������@9������@A������@I������@aQ��/�k?ioL�Aí�?�Unknown
t#HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff @Affffff@Iffffff @amI��j?i}i���?�Unknown
�$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a�3�ЂMj?i����?�Unknown
�%HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @a�3�ЂMj?i�Ͷ T��?�Unknown
�&HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a�3�ЂMj?i�����?�Unknown
�'HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a�XGX,�i?ir���F1�?�Unknown
Z(HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a�}����h?i𣿥CJ�?�Unknown
X)HostCast"Cast_2(1������@9������@A������@I������@a�qgTh?i�'%�b�?�Unknown
�*HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1������@9������@A������@I������@a�qgTh?i6����z�?�Unknown
x+HostSlice"%gradient_tape/model/concatenate/Slice(1������@9������@A������@I������@a�qgTh?i���#A��?�Unknown
�,HostSquaredDifference"$mean_squared_error/SquaredDifference(1������@9������@A������@I������@a�qgTh?i|j]����?�Unknown
|-HostSlice")gradient_tape/model/concatenate_1/Slice_1(1������	@9������	@A������	@I������	@ao\[�
e?i��jr���?�Unknown
z.HostReluGrad"$gradient_tape/model/dense_2/ReluGrad(1������@9������@A������@I������@a���xbd?iZ�����?�Unknown
�/HostBiasAddGrad"/gradient_tape/model/dense_4/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a���xbd?iܦ�ce��?�Unknown
�0HostBiasAddGrad"/gradient_tape/model/dense_3/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a����c?i��8/w��?�Unknown
|1HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a��+uib?i�qd���?�Unknown
�2HostReadVariableOp"$model/dense_2/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a��+uib?i�!�J!�?�Unknown
�3HostReadVariableOp"$model/dense_4/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a�E��a?i�fC83�?�Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_2(1������@9������@A������@I������@a;�:�a?i�@~ $D�?�Unknown
V5HostSum"Sum_2(1������@9������@A������@I������@a;�:�a?i��<U�?�Unknown
z6HostSlice"'gradient_tape/model/concatenate/Slice_1(1������@9������@A������@I������@a;�:�a?iQ��Uf�?�Unknown
x7HostReluGrad""gradient_tape/model/dense/ReluGrad(1      @9      @A      @I      @a6`o�qp`?i�d��v�?�Unknown
d8HostTanh"model/dense_3/Tanh(1      @9      @A      @I      @a6`o�qp`?i�xt6��?�Unknown
�9HostBiasAddGrad"-gradient_tape/model/dense/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�
	�6�_?i�����?�Unknown
�:HostReadVariableOp"$model/dense_1/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a�]���\?if�u��?�Unknown
V;HostAddN"AddN(1������ @9������ @A������ @I������ @aQ��/�[?i[��E��?�Unknown
�<HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������ @9������ @A������ @I������ @aQ��/�[?iP��-��?�Unknown
X=HostCast"Cast_3(1       @9       @A       @I       @a�3�ЂMZ?ij�E�:��?�Unknown
�>HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a�3�ЂMZ?i�A��a��?�Unknown
z?HostReluGrad"$gradient_tape/model/dense_1/ReluGrad(1       @9       @A       @I       @a�3�ЂMZ?i��r���?�Unknown
u@HostSum"$mean_squared_error/weighted_loss/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�}����X?i݈���?�Unknown
XAHostEqual"Equal(1�������?9�������?A�������?I�������?ao\[�
U?i�6�D���?�Unknown
�BHostDataset"-Iterator::Model::ParallelMap::Zip[0]::FlatMap(13333339@93333339@A�������?I�������?ao\[�
U?i9��
�?�Unknown
}CHostMaximum"(gradient_tape/mean_squared_error/Maximum(1�������?9�������?A�������?I�������?ao\[�
U?i瑚��?�Unknown
jDHostSigmoid"model/dense_4/Sigmoid(1�������?9�������?A�������?I�������?ao\[�
U?i�?!{�?�Unknown
�EHostDataset"?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(1      �?9      �?A      �?I      �?a���"�S?ih�/��(�?�Unknown
�FHostBiasAddGrad"/gradient_tape/model/dense_2/BiasAdd/BiasAddGrad(1      �?9      �?A      �?I      �?a���"�S?i;�=��2�?�Unknown
�GHostBiasAddGrad"/gradient_tape/model/dense_1/BiasAdd/BiasAddGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��+uiR?i3��W<�?�Unknown
bHHostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a;�:�Q?iQ
�D�?�Unknown
uIHostSub"$gradient_tape/mean_squared_error/sub(1�������?9�������?A�������?I�������?a;�:�Q?iow $M�?�Unknown
�JHostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a;�:�Q?i��+��U�?�Unknown
�KHostDataset"LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(1�������?9�������?A�������?I�������?a�]���L?i�{X;�\�?�Unknown
TLHostMul"Mul(1�������?9�������?A�������?I�������?a�]���L?i]��'d�?�Unknown
jMHostReadVariableOp"ReadVariableOp(1�������?9�������?A�������?I�������?a�]���L?iŪ��ck�?�Unknown
sNHostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�]���L?i-B�`�r�?�Unknown
}OHostRealDiv"(gradient_tape/mean_squared_error/truediv(1�������?9�������?A�������?I�������?a�]���L?i��
�y�?�Unknown
zPHostTanhGrad"$gradient_tape/model/dense_3/TanhGrad(1�������?9�������?A�������?I�������?a�]���L?i�p7���?�Unknown
uQHostMul"$gradient_tape/mean_squared_error/Mul(1      �?9      �?A      �?I      �?a�3�ЂMJ?i���/���?�Unknown
|RHostDivNoNan"&mean_squared_error/weighted_loss/value(1      �?9      �?A      �?I      �?a�3�ЂMJ?iʟ�=��?�Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?a���(�G?iɋۚ(��?�Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a���(�G?i{M���?�Unknown
XUHostCast"Cast_4(1�������?9�������?A�������?I�������?a���(�G?i-S����?�Unknown
wVHostCast"%gradient_tape/mean_squared_error/Cast(1�������?9�������?A�������?I�������?a���(�G?i�Ў���?�Unknown
uWHostSum"$gradient_tape/mean_squared_error/Sum(1�������?9�������?A�������?I�������?a���(�G?i����ԫ�?�Unknown
XHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1�������?9�������?A�������?I�������?a���(�G?iCTο��?�Unknown
YHostReadVariableOp"#model/dense_2/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a���(�G?i�Bت��?�Unknown
ZHostReadVariableOp"#model/dense_3/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a���(�G?i��}╽�?�Unknown
`[HostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?ao\[�
E?i~.A����?�Unknown
�\HostSigmoidGrad"/gradient_tape/model/dense_4/Sigmoid/SigmoidGrad(1�������?9�������?A�������?I�������?ao\[�
E?iU�J��?�Unknown
}]HostReadVariableOp"!model/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?ao\[�
E?i,���]��?�Unknown
^HostReadVariableOp"#model/dense_1/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?ao\[�
E?i3�����?�Unknown
_HostReadVariableOp"#model/dense_4/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?ao\[�
E?iډNe���?�Unknown
�`HostDataset"MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��+uiB?i�u��}��?�Unknown
uaHostReadVariableOp"div_no_nan/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��+uiB?i�a���?�Unknown
wbHostMul"&gradient_tape/mean_squared_error/mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��+uiB?i�M/}���?�Unknown
~cHostReadVariableOp""model/dense/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��+uiB?i�9z�L��?�Unknown
XdHostCast"Cast_1(1333333�?9333333�?A333333�?I333333�?a�
	�6�??i�L�>��?�Unknown
weHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�
	�6�??i<�0��?�Unknown
yfHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�
	�6�??i-���"��?�Unknown
�gHostReadVariableOp"$model/dense_3/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�
	�6�??iN>����?�Unknown
�hHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a�3�ЂM:?i�T�^��?�Unknown
wiHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?ao\[�
5?i      �?�Unknown