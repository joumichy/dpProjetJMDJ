"�J
VHostIDLE"IDLE(1�����ѭ@9��
�@A�����ѭ@I��
�@a���gٽ�?i���gٽ�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333�@933333�@A33333�@I33333�@a�a���?i���d�?�Unknown
�HostDataset"0Iterator::Model::Prefetch::FlatMap[0]::Generator(1     ��@9     ��@A     ��@I     ��@ab�s9�?i�����?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�����p�@9�����p�@A�����p�@I�����p�@a�^yw��?i��G�0J�?�Unknown
ZHostPyFunc"PyFunc(1����̾�@9����̾�@A����̾�@I����̾�@a�L���?i[�}:�?�Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1�����]�@9�����]�@A�����]�@I�����]�@a1>us3H�?i!*��C�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      S@9      S@A      S@I      S@azr�b��v?i7Q�Op�?�Unknown
dHostDataset"Iterator::Model(1ffffff>@9ffffff>@A������:@I������:@a�9�V+�_?i#��u#��?�Unknown
�	HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1     �7@9     �7@A     �7@I     �7@aa�G���[?i �M���?�Unknown
t
Host_FusedMatMul"sequential/dense_1/BiasAdd(1fffff�4@9fffff�4@Afffff�4@Ifffff�4@aS䓟��X?i��A���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1ffffff0@9ffffff0@Affffff0@Iffffff0@al]�_��S?iAl�9P��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������'@9������'@A������'@I������'@aE5;,GL?i{��U��?�Unknown
fHostGreaterEqual"GreaterEqual(1ffffff&@9ffffff&@Affffff&@Iffffff&@a�fԴ��J?i(��y���?�Unknown
zHostStridedSlice" sequential/flatten/strided_slice(1������@9������@A������@I������@aE5;,G<?i�7�"���?�Unknown
�HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1������@9������@A������@I������@a-V�ܒ!;?i��U��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a�fԴ��:?iGfL;��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aX�ƛ�7?i '��4��?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1������@9������@A������@I������@a,9�~�4?iFq���?�Unknown
ZHostArgMax"ArgMax(1������@9������@A������@I������@a�L����3?i0��R��?�Unknown
iHostMean"mean_squared_error/Mean(1������@9������@A������@I������@a�L����3?i�����?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@al]�_��3?if� C��?�Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@al]�_��3?i��*~���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@aT~8<�2?i����?�Unknown
\HostArgMax"ArgMax_1(1ffffff@9ffffff@Affffff@Iffffff@aȎk�a2?i4��QH��?�Unknown
nHostDataset"Iterator::Model::Prefetch(1ffffff@9ffffff@Affffff@Iffffff@aȎk�a2?i�'���?�Unknown
XHostCast"Cast_2(1������@9������@A������@I������@a��ј�"1?i�&�s���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1333333@9333333@A333333@I333333@a��7I�.0?i�MS���?�Unknown
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��B>j/?i2{�����?�Unknown
�HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a���Ղ-?iR�A$���?�Unknown
rHostPack" sequential/flatten/Reshape/shape(1������@9������@A������@I������@a���Ղ-?ir�Q\��?�Unknown
�HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a�$T!�,?i�/�C%��?�Unknown
g HostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a�fԴ��*?i:}<����?�Unknown
t!Host_FusedMatMul"sequential/dense_2/BiasAdd(1������@9������@A������@I������@aq��P�(?iE�=�[��?�Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@a@�lv��&?i>�R���?�Unknown
g#HostTanh"sequential/dense/Tanh(1333333@9333333@A333333@I333333@a@�lv��&?i�,�6��?�Unknown
�$HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a(�&3�%?i_4���?�Unknown
X%HostEqual"Equal(1�������?9�������?A�������?I�������?a��ј�"!?i/�8_���?�Unknown
�&HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1�������?9�������?A�������?I�������?a��ј�"!?iJ,����?�Unknown
�'HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��ј�"!?ie�����?�Unknown
�(HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1333333�?9333333�?A333333�?I333333�?a��7I�. ?i�L�����?�Unknown
X)HostCast"Cast_3(1�������?9�������?A�������?I�������?a�;�v?i���X���?�Unknown
�*HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1�������?9�������?A�������?I�������?a�;�v?i�����?�Unknown
w+HostDataset""Iterator::Model::Prefetch::FlatMap(1     ��@9     ��@A      �?I      �?a�$T!�?i� *����?�Unknown
T,HostMul"Mul(1      �?9      �?A      �?I      �?a�$T!�?i"�4�~��?�Unknown
j-HostReadVariableOp"ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��?i�g�<T��?�Unknown
u.HostSum"$mean_squared_error/weighted_loss/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��?ih�z)��?�Unknown
|/HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��?i������?�Unknown
}0HostMaximum"(gradient_tape/mean_squared_error/Maximum(1�������?9�������?A�������?I�������?aq��P�?ib����?�Unknown
o1HostSigmoid"sequential/dense_2/Sigmoid(1�������?9�������?A�������?I�������?aq��P�?i�����?�Unknown
X2HostCast"Cast_4(1333333�?9333333�?A333333�?I333333�?a@�lv��?i|�A��?�Unknown
u3HostSub"$gradient_tape/mean_squared_error/sub(1333333�?9333333�?A333333�?I333333�?a@�lv��?i�u�K���?�Unknown
u4HostMul"$gradient_tape/mean_squared_error/Mul(1�������?9�������?A�������?I�������?a,9�~�?i�/�ן��?�Unknown
�5HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a,9�~�?iu��cG��?�Unknown
s6HostReadVariableOp"SGD/Cast/ReadVariableOp(1      �?9      �?A      �?I      �?a�m8
?i�������?�Unknown
b7HostDivNoNan"div_no_nan_1(1      �?9      �?A      �?I      �?a�m8
?i�iOx��?�Unknown
w8HostCast"%gradient_tape/mean_squared_error/Cast(1      �?9      �?A      �?I      �?a�m8
?i�)V��?�Unknown
}9HostRealDiv"(gradient_tape/mean_squared_error/truediv(1      �?9      �?A      �?I      �?a�m8
?i!겦���?�Unknown
�:HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      �?9      �?A      �?I      �?a�m8
?iL�d�@��?�Unknown
;HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1�������?9�������?A�������?I�������?a��ј�"?i�p����?�Unknown
<HostTanhGrad")gradient_tape/sequential/dense_1/TanhGrad(1�������?9�������?A�������?I�������?a��ј�"?if7>"S��?�Unknown
�=HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��ј�"?i���7���?�Unknown
`>HostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?a�;�v?i���V��?�Unknown
w?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�;�v?iӗ�����?�Unknown
}@HostTanhGrad"'gradient_tape/sequential/dense/TanhGrad(1�������?9�������?A�������?I�������?a�;�v?i�d"�I��?�Unknown
�AHostSigmoidGrad"4gradient_tape/sequential/dense_2/Sigmoid/SigmoidGrad(1�������?9�������?A�������?I�������?a�;�v?i�1J����?�Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��
?i-?.��?�Unknown
wCHostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��
?iW�ޘ��?�Unknown
uDHostSum"$gradient_tape/mean_squared_error/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��
?i���|��?�Unknown
�EHostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��
?i�~�n��?�Unknown
�FHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��
?iMR�����?�Unknown
iGHostTanh"sequential/dense_1/Tanh(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�fԴ��
?i�%�YC��?�Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a@�lv��?iS�8����?�Unknown
XIHostCast"Cast_1(1333333�?9333333�?A333333�?I333333�?a@�lv��?i�� ���?�Unknown
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a@�lv��?i��t�U��?�Unknown
yKHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a@�lv��?io����?�Unknown
wLHostMul"&gradient_tape/mean_squared_error/mul_1(1333333�?9333333�?A333333�?I333333�?a@�lv��?i#f�K��?�Unknown
�MHostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333�?9333333�?A333333�?I333333�?a@�lv��?i�?N�g��?�Unknown
�NHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1333333�?9333333�?A333333�?I333333�?a@�lv��?i�����?�Unknown
VOHostSum"Sum_2(1�������?9�������?A�������?I�������?a�;�v�>i     �?�Unknown*�J
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333�@933333�@A33333�@I33333�@a���4���?i���4���?�Unknown
�HostDataset"0Iterator::Model::Prefetch::FlatMap[0]::Generator(1     ��@9     ��@A     ��@I     ��@a���S��?i>B���?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�����p�@9�����p�@A�����p�@I�����p�@a�_�?��?i6����?�Unknown
ZHostPyFunc"PyFunc(1����̾�@9����̾�@A����̾�@I����̾�@a�@�_�@�?i^OV�)�?�Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1�����]�@9�����]�@A�����]�@I�����]�@a���c׆�?ivB���?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      S@9      S@A      S@I      S@acr��G?i���39�?�Unknown
dHostDataset"Iterator::Model(1ffffff>@9ffffff>@A������:@I������:@aߜ���e?i�D6xO�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1     �7@9     �7@A     �7@I     �7@a����9Xc?ik$/�qb�?�Unknown
t	Host_FusedMatMul"sequential/dense_1/BiasAdd(1fffff�4@9fffff�4@Afffff�4@Ifffff�4@a2sR4a?i��>�s�?�Unknown
{
HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1ffffff0@9ffffff0@Affffff0@Iffffff0@ar'�1 [?i���	&��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������'@9������'@A������'@I������'@a����LmS?i�
6�܊�?�Unknown
fHostGreaterEqual"GreaterEqual(1ffffff&@9ffffff&@Affffff&@Iffffff&@a��E�kpR?ii-����?�Unknown
zHostStridedSlice" sequential/flatten/strided_slice(1������@9������@A������@I������@a����LmC?i�h+9��?�Unknown
�HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1������@9������@A������@I������@a�}���B?iK��f���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a��E�kpB?i����=��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a���*�v@?i�V ,[��?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1������@9������@A������@I������@a1�H���<?i��e���?�Unknown
ZHostArgMax"ArgMax(1������@9������@A������@I������@a]ZhR��;?i�yo��?�Unknown
iHostMean"mean_squared_error/Mean(1������@9������@A������@I������@a]ZhR��;?iZ���?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@ar'�1 ;?i �m�D��?�Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@ar'�1 ;?i%�Ӑ���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a���ޮ9?i��lں�?�Unknown
\HostArgMax"ArgMax_1(1ffffff@9ffffff@Affffff@Iffffff@a����H9?i�5���?�Unknown
nHostDataset"Iterator::Model::Prefetch(1ffffff@9ffffff@Affffff@Iffffff@a����H9?i%����?�Unknown
XHostCast"Cast_2(1������@9������@A������@I������@a�(Ǐ�7?i�x���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1333333@9333333@A333333@I333333@a��N�c6?i��� ���?�Unknown
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1ffffff
@9ffffff
@Affffff
@Iffffff
@a$�v.Z�5?i��ǋ���?�Unknown
�HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1������@9������@A������@I������@aQ*��-j4?iU|��#��?�Unknown
rHostPack" sequential/flatten/Reshape/shape(1������@9������@A������@I������@aQ*��-j4?i/C���?�Unknown
�HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @ag�%͗�3?i��<J)��?�Unknown
gHostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a��E�kp2?i�\�Ww��?�Unknown
t Host_FusedMatMul"sequential/dense_2/BiasAdd(1������@9������@A������@I������@a�+eK?1?i0ɗ?���?�Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@a׋	&�/?i�����?�Unknown
g"HostTanh"sequential/dense/Tanh(1333333@9333333@A333333@I333333@a׋	&�/?ibjZĎ��?�Unknown
�#HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a&)��J.?i���ss��?�Unknown
X$HostEqual"Equal(1�������?9�������?A�������?I�������?a�(Ǐ�'?ig������?�Unknown
�%HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1�������?9�������?A�������?I�������?a�(Ǐ�'?iڥ�j��?�Unknown
�&HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�(Ǐ�'?iM�Ri���?�Unknown
�'HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1333333�?9333333�?A333333�?I333333�?a��N�c&?i��W�K��?�Unknown
X(HostCast"Cast_3(1�������?9�������?A�������?I�������?a;]�%?iq�Ԝ��?�Unknown
�)HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1�������?9�������?A�������?I�������?a;]�%?i�Q� ���?�Unknown
w*HostDataset""Iterator::Model::Prefetch::FlatMap(1     ��@9     ��@A      �?I      �?ag�%͗�#?i�#V*��?�Unknown
T+HostMul"Mul(1      �?9      �?A      �?I      �?ag�%͗�#?iC��3f��?�Unknown
j,HostReadVariableOp"ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp"?i���:���?�Unknown
u-HostSum"$mean_squared_error/weighted_loss/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp"?i�~DA���?�Unknown
|.HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp"?iNC�G���?�Unknown
}/HostMaximum"(gradient_tape/mean_squared_error/Maximum(1�������?9�������?A�������?I�������?a�+eK?!?i���;���?�Unknown
o0HostSigmoid"sequential/dense_2/Sigmoid(1�������?9�������?A�������?I�������?a�+eK?!?i���/���?�Unknown
X1HostCast"Cast_4(1333333�?9333333�?A333333�?I333333�?a׋	&�?i@X���?�Unknown
u2HostSub"$gradient_tape/mean_squared_error/sub(1333333�?9333333�?A333333�?I333333�?a׋	&�?i� H����?�Unknown
u3HostMul"$gradient_tape/mean_squared_error/Mul(1�������?9�������?A�������?I�������?a1�H���?iҚ�����?�Unknown
�4HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a1�H���?i5!����?�Unknown
s5HostReadVariableOp"SGD/Cast/ReadVariableOp(1      �?9      �?A      �?I      �?a��uW?iX��J���?�Unknown
b6HostDivNoNan"div_no_nan_1(1      �?9      �?A      �?I      �?a��uW?i�Mrn��?�Unknown
w7HostCast"%gradient_tape/mean_squared_error/Cast(1      �?9      �?A      �?I      �?a��uW?i���@��?�Unknown
}8HostRealDiv"(gradient_tape/mean_squared_error/truediv(1      �?9      �?A      �?I      �?a��uW?if�}��?�Unknown
�9HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      �?9      �?A      �?I      �?a��uW?iX�k9���?�Unknown
:HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1�������?9�������?A�������?I�������?a�(Ǐ�?i�pP���?�Unknown
;HostTanhGrad")gradient_tape/sequential/dense_1/TanhGrad(1�������?9�������?A�������?I�������?a�(Ǐ�?i��4�a��?�Unknown
�<HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�(Ǐ�?im4��?�Unknown
`=HostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?a;]�?i6�9����?�Unknown
w>HostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a;]�?iiMZ`p��?�Unknown
}?HostTanhGrad"'gradient_tape/sequential/dense/TanhGrad(1�������?9�������?A�������?I�������?a;]�?i��z���?�Unknown
�@HostSigmoidGrad"4gradient_tape/sequential/dense_2/Sigmoid/SigmoidGrad(1�������?9�������?A�������?I�������?a;]�?i�-�����?�Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp?i���U��?�Unknown
wBHostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp?i)�S����?�Unknown
uCHostSum"$gradient_tape/mean_squared_error/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp?iVT�|��?�Unknown
�DHostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp?i�����?�Unknown
�EHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp?i�i���?�Unknown
iFHostTanh"sequential/dense_1/Tanh(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��E�kp?i�zŠ6��?�Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a׋	&�?i�]���?�Unknown
XHHostCast"Cast_1(1333333�?9333333�?A333333�?I333333�?a׋	&�?i)#��3��?�Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a׋	&�?iOw����?�Unknown
yJHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a׋	&�?iu�&c0��?�Unknown
wKHostMul"&gradient_tape/mean_squared_error/mul_1(1333333�?9333333�?A333333�?I333333�?a׋	&�?i��Ӯ��?�Unknown
�LHostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333�?9333333�?A333333�?I333333�?a׋	&�?i�sWD-��?�Unknown
�MHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1333333�?9333333�?A333333�?I333333�?a׋	&�?i��ﴫ��?�Unknown
VNHostSum"Sum_2(1�������?9�������?A�������?I�������?a;]�?i      �?�Unknown