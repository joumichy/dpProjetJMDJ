"�H
VHostIDLE"IDLE(1fffff'�@9DDDDD�@Afffff'�@IDDDDD�@a��i
E�?i��i
E�?�Unknown
nHostDataset"Iterator::Model::Prefetch(1����L�@9����L�@A����L�@I����L�@aMji�V��?iu�%�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     K�@9     K�@A     K�@I     K�@a�2t^�>�?iq�C�zm�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�����/�@9�����/�@A�����/�@I�����/�@a �j��?ia�付��?�Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1fffffJ�@9fffffJ�@AfffffJ�@IfffffJ�@a��Ԁ�ɟ?iV��7��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�����`@9�����`@A�����`@I�����`@as��8i?iɫ� ��?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1fffff&H@9fffff&H@Afffff&H@Ifffff&H@aL�J���R?i4Q�w���?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(133333�E@933333�E@A33333�E@I33333�E@a�Ԇ1Q?i&Sg;,��?�Unknown
�	HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      D@9      D@A      D@I      D@a��	]UO?i���{��?�Unknown
�
HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1������?@9������?@A������?@I������?@a����H?i	�~�;��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1ffffff2@9ffffff2@Affffff2@Iffffff2@a�̢,S�<?icj����?�Unknown
dHostDataset"Iterator::Model(1����L�@9����L�@A      .@I      .@a���7?iS#���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1������-@9������-@A������-@I������-@a��"�W7?iu{�����?�Unknown
fHostGreaterEqual"GreaterEqual(1������&@9������&@A������&@I������&@a���5޳1?i�6�{���?�Unknown
gHostTanh"sequential/dense/Tanh(1ffffff@9ffffff@Affffff@Iffffff@aG���>&?i�/jK��?�Unknown
zHostStridedSlice" sequential/flatten/strided_slice(1������@9������@A������@I������@a�L`��$?i�e\K���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@a���m#?i�׻���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a��k��"?i0�b����?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a��k��"?i�8	�+��?�Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1������@9������@A������@I������@a���d|"?i"�SuS��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a�Ō�;!?izT,.g��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1������@9������@A������@I������@a�̞:�J ?ig���k��?�Unknown
ZHostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a��#�l�?i�GW�k��?�Unknown
iHostMean"mean_squared_error/Mean(1ffffff@9ffffff@Affffff@Iffffff@a��#�l�?i���3k��?�Unknown
\HostArgMax"ArgMax_1(1      @9      @A      @I      @a��	]U?i�x��e��?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a	3�Е�?iu x�[��?�Unknown
�HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a=f�D*?i('�!L��?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a=f�D*?i�M�<��?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a�����2?i#RYZ��?�Unknown
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1������@9������@A������@I������@a3o|�?i��9����?�Unknown
XHostCast"Cast_2(1333333@9333333@A333333@I333333@aAfU��?iH8�~���?�Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1������@9������@A������@I������@av�;��Q?i%����?�Unknown
l!HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@av�;��Q?i��w��?�Unknown
�"HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@aEf����?i�Z�6��?�Unknown
i#HostTanh"sequential/dense_1/Tanh(1      @9      @A      @I      @a���'��?iݖe����?�Unknown
�$HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a���m?i�O�����?�Unknown
g%HostStridedSlice"strided_slice(1������@9������@A������@I������@a���m?i��^��?�Unknown
r&HostPack" sequential/flatten/Reshape/shape(1333333@9333333@A333333@I333333@a3��.,?i; <����?�Unknown
V'HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@aNf�RË?iΖV:��?�Unknown
�(HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@a=f�D*?i(��n���?�Unknown
X)HostCast"Cast_3(1ffffff@9ffffff@Affffff@Iffffff@a�̢,S�?i�\L�%��?�Unknown
�*HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������ @9������ @A������ @I������ @av�;��Q
?i�M����?�Unknown
u+HostSum"$mean_squared_error/weighted_loss/Sum(1������ @9������ @A������ @I������ @av�;��Q
?i�>tI���?�Unknown
},HostTanhGrad"'gradient_tape/sequential/dense/TanhGrad(1       @9       @A       @I       @a����	?i�Ϋ�\��?�Unknown
-HostTanhGrad")gradient_tape/sequential/dense_1/TanhGrad(1       @9       @A       @I       @a����	?i�^�����?�Unknown
�.HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a����	?i��%��?�Unknown
X/HostEqual"Equal(1�������?9�������?A�������?I�������?a�̠��?ir��O��?�Unknown
}0HostRealDiv"(gradient_tape/mean_squared_error/truediv(1333333�?9333333�?A333333�?I333333�?a3m�HN?i'+�����?�Unknown
}1HostMaximum"(gradient_tape/mean_squared_error/Maximum(1�������?9�������?A�������?I�������?a~�9�q?i8��$��?�Unknown
j2HostReadVariableOp"ReadVariableOp(1      �?9      �?A      �?I      �?a��k��?i%���o��?�Unknown
X3HostCast"Cast_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?aNf�RË?io/����?�Unknown
T4HostMul"Mul(1ffffff�?9ffffff�?Affffff�?Iffffff�?aNf�RË?i�zO���?�Unknown
|5HostDivNoNan"&mean_squared_error/weighted_loss/value(1�������?9�������?A�������?I�������?a�̞:�J ?i4e�z=��?�Unknown
b6HostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a3o|��>i���t��?�Unknown
w7HostCast"%gradient_tape/mean_squared_error/Cast(1�������?9�������?A�������?I�������?a3o|��>i�ī��?�Unknown
u8HostMul"$gradient_tape/mean_squared_error/Mul(1�������?9�������?A�������?I�������?a3o|��>i�ߟ����?�Unknown
9HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1�������?9�������?A�������?I�������?a3o|��>i����?�Unknown
u:HostSub"$gradient_tape/mean_squared_error/sub(1�������?9�������?A�������?I�������?a3o|��>i�1�3Q��?�Unknown
�;HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      �?9      �?A      �?I      �?a�����>i��+U���?�Unknown
�<HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a�����>i���v���?�Unknown
o=HostSigmoid"sequential/dense_2/Sigmoid(1      �?9      �?A      �?I      �?a�����>i��c����?�Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?a�̠���>i����?�Unknown
s?HostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�̠���>i>X��A��?�Unknown
y@HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�̠���>i��!�n��?�Unknown
uAHostSum"$gradient_tape/mean_squared_error/Sum(1�������?9�������?A�������?I�������?a�̠���>i�&a���?�Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a~�9�q�>i5-D,���?�Unknown
`CHostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?a~�9�q�>i�3'G���?�Unknown
�DHostSigmoidGrad"4gradient_tape/sequential/dense_2/Sigmoid/SigmoidGrad(1�������?9�������?A�������?I�������?a~�9�q�>i:
b��?�Unknown
XEHostCast"Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aNf�RË�>i�ߐy7��?�Unknown
�FHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aNf�RË�>ie��Z��?�Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a=f�D*�>i;�A�x��?�Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a=f�D*�>il����?�Unknown
wIHostMul"&gradient_tape/mean_squared_error/mul_1(1333333�?9333333�?A333333�?I333333�?a=f�D*�>i�S�ʹ��?�Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      �?9      �?A      �?I      �?a�����>i�7d����?�Unknown
�KHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a�����>i�2����?�Unknown
�LHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a�����>i�������?�Unknown*�G
nHostDataset"Iterator::Model::Prefetch(1����L�@9����L�@A����L�@I����L�@a��:{���?i��:{���?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     K�@9     K�@A     K�@I     K�@a�F�4�?i�j7'T�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�����/�@9�����/�@A�����/�@I�����/�@a
	�,�?�?i髪��?�Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1fffffJ�@9fffffJ�@AfffffJ�@IfffffJ�@a5�����?i��M3|��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�����`@9�����`@A�����`@I�����`@al����͂?i�Z�	��?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1fffff&H@9fffff&H@Afffff&H@Ifffff&H@a!`{W�4l?iF����9�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(133333�E@933333�E@A33333�E@I33333�E@a6�Rx!�i?iB)\�S�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      D@9      D@A      D@I      D@aH/o�\g?iq�1.�j�?�Unknown
�	HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1������?@9������?@A������?@I������?@an飒(�b?iZ<�Vz}�?�Unknown
}
HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1ffffff2@9ffffff2@Affffff2@Iffffff2@aV�Q��}U?iB�:,9��?�Unknown
dHostDataset"Iterator::Model(1����L�@9����L�@A      .@I      .@avc �Q?i��J����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1������-@9������-@A������-@I������-@aw���'gQ?i��G���?�Unknown
fHostGreaterEqual"GreaterEqual(1������&@9������&@A������&@I������&@a0랮{eJ?iv_�H��?�Unknown
gHostTanh"sequential/dense/Tanh(1ffffff@9ffffff@Affffff@Iffffff@a}�,ڕ@?i�`�n��?�Unknown
zHostStridedSlice" sequential/flatten/strided_slice(1������@9������@A������@I������@a
�oLM??i$^ �W��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@aOǍJ�<?ir����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a#���<?i"NE�w��?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a#���<?i6�����?�Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1������@9������@A������@I������@a'G��|�;?i_{��j��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a5瓷�9?i�mC���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1������@9������@A������@I������@aA�}�FK8?i��p���?�Unknown
ZHostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@aD�vϬ�7?ii��夿�?�Unknown
iHostMean"mean_squared_error/Mean(1ffffff@9ffffff@Affffff@Iffffff@aD�vϬ�7?i:{P[���?�Unknown
\HostArgMax"ArgMax_1(1      @9      @A      @I      @aH/o�\7?i )�݊��?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1������@9������@A������@I������@aL�g�x�6?i��lg��?�Unknown
�HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@aO`��l6?i+¢5��?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@aO`��l6?i;�~���?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @aZwJ�5?i�W�f���?�Unknown
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1������@9������@A������@I������@a^C�v�4?i�|55��?�Unknown
XHostCast"Cast_2(1333333@9333333@A333333@I333333@aa�;��4?igg���?�Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1������@9������@A������@I������@afo4C�3?i�}�+��?�Unknown
l HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@afo4C�3?i������?�Unknown
�!HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@at��1?ieWA����?�Unknown
i"HostTanh"sequential/dense_1/Tanh(1      @9      @A      @I      @a/Z0?i�7�>���?�Unknown
�#HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1������@9������@A������@I������@aOǍJ�,?i������?�Unknown
g$HostStridedSlice"strided_slice(1������@9������@A������@I������@aOǍJ�,?ip�8(���?�Unknown
r%HostPack" sequential/flatten/Reshape/shape(1333333@9333333@A333333@I333333@a*贈�+?iKc�3��?�Unknown
V&HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a1?���)*?i�dNQ���?�Unknown
�'HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@aO`��l&?i�z<=��?�Unknown
X(HostCast"Cast_3(1ffffff@9ffffff@Affffff@Iffffff@aV�Q��}%?i�O�����?�Unknown
�)HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������ @9������ @A������ @I������ @afo4C�#?i/�����?�Unknown
u*HostSum"$mean_squared_error/weighted_loss/Sum(1������ @9������ @A������ @I������ @afo4C�#?iv�K���?�Unknown
}+HostTanhGrad"'gradient_tape/sequential/dense/TanhGrad(1       @9       @A       @I       @am�%�"?i�=�3��?�Unknown
,HostTanhGrad")gradient_tape/sequential/dense_1/TanhGrad(1       @9       @A       @I       @am�%�"?i..�^��?�Unknown
�-HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @am�%�"?i�-���?�Unknown
X.HostEqual"Equal(1�������?9�������?A�������?I�������?a|_)�� ?i������?�Unknown
}/HostRealDiv"(gradient_tape/mean_squared_error/truediv(1333333�?9333333�?A333333�?I333333�?a_�i��?i��&���?�Unknown
}0HostMaximum"(gradient_tape/mean_squared_error/Maximum(1�������?9�������?A�������?I�������?a�Ձ~�?i[�Z���?�Unknown
j1HostReadVariableOp"ReadVariableOp(1      �?9      �?A      �?I      �?a#���?i �m�d��?�Unknown
X2HostCast"Cast_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a1?���)?i�v��5��?�Unknown
T3HostMul"Mul(1ffffff�?9ffffff�?Affffff�?Iffffff�?a1?���)?i�Y6��?�Unknown
|4HostDivNoNan"&mean_squared_error/weighted_loss/value(1�������?9�������?A�������?I�������?aA�}�FK?i�O�����?�Unknown
b5HostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a^C�v�?i�Gn��?�Unknown
w6HostCast"%gradient_tape/mean_squared_error/Cast(1�������?9�������?A�������?I�������?a^C�v�?i���w��?�Unknown
u7HostMul"$gradient_tape/mean_squared_error/Mul(1�������?9�������?A�������?I�������?a^C�v�?i�����?�Unknown
8HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1�������?9�������?A�������?I�������?a^C�v�?i'xn_[��?�Unknown
u9HostSub"$gradient_tape/mean_squared_error/sub(1�������?9�������?A�������?I�������?a^C�v�?i@B&����?�Unknown
�:HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      �?9      �?A      �?I      �?am�%�?in˞S���?�Unknown
�;HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?am�%�?i�T�*��?�Unknown
o<HostSigmoid"sequential/dense_2/Sigmoid(1      �?9      �?A      �?I      �?am�%�?i�ݏT���?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?a|_)��?i&��F��?�Unknown
s>HostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a|_)��?iPno���?�Unknown
y?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a|_)��?i��;�S��?�Unknown
u@HostSum"$gradient_tape/mean_squared_error/Sum(1�������?9�������?A�������?I�������?a|_)��?i��t����?�Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a�Ձ~�?i.o#R��?�Unknown
`BHostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?a�Ձ~�?i�i����?�Unknown
�CHostSigmoidGrad"4gradient_tape/sequential/dense_2/Sigmoid/SigmoidGrad(1�������?9�������?A�������?I�������?a�Ձ~�?i�cWA��?�Unknown
XDHostCast"Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a1?���)
?iK�����?�Unknown
�EHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a1?���)
?i��ؤ��?�Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aO`��l?i:'TXl��?�Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?aO`��l?i������?�Unknown
wHHostMul"&gradient_tape/mean_squared_error/mul_1(1333333�?9333333�?A333333�?I333333�?aO`��l?i>2K���?�Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      �?9      �?A      �?I      �?am�%�?i�v�j��?�Unknown
�JHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?am�%�?il��?���?�Unknown
�KHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?am�%�?i     �?�Unknown