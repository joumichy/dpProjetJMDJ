"�N
VHostIDLE"IDLE(1�������@9B�A<�@A�������@IB�A<�@aE���*1�?iE���*1�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1fffff��@9fffff��@Afffff��@Ifffff��@aI����#�?i5&� �!�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1ffffft�@9ffffft�@Affffft�@Iffffft�@a��C��?ixw�dw�?�Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1�����Q�@9�����Q�@A�����Q�@I�����Q�@ax����i�?ig�T��#�?�Unknown
^HostGatherV2"GatherV2(1     ��@9     ��@A     ��@I     ��@a�1�o0��?i�]Q�z�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1fffffFf@9fffffFf@AfffffFf@IfffffFf@aC/kf�?i^�i=���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1������e@9������e@A������e@I������e@aTOpF\�?i��*W#q�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1fffff�`@9fffff�`@Afffff�`@Ifffff�`@ah3����?i|Hѝ_��?�Unknown
t	Host_FusedMatMul"sequential/dense_1/BiasAdd(1fffff&O@9fffff&O@Afffff&O@Ifffff&O@a�Z�ϓ�u?i2�p�H��?�Unknown
{
HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1����̌F@9����̌F@A����̌F@I����̌F@az��d��o?i��m�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1     �D@9     �D@A     �D@I     �D@a,��@m?i���R8�?�Unknown
�HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1fffff�;@9fffff�;@Afffff�;@Ifffff�;@a[%�*)�c?i=*���K�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(133333�7@933333�7@A33333�7@I33333�7@a 3��T�`?ip�6�\�?�Unknown
�HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1fffff�0@9fffff�0@Afffff�0@Ifffff�0@a0�����W?i�x�_�h�?�Unknown
�HostDataset"3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat(13333333@93333333@A������/@I������/@aV��EV?i�{�B�s�?�Unknown
dHostDataset"Iterator::Model(1������4@9������4@A333333+@I333333+@a����+S?iZ�0S}�?�Unknown
fHostGreaterEqual"GreaterEqual(1333333&@9333333&@A333333&@I333333&@a�oEIZKO?i�@&��?�Unknown
gHostTanh"sequential/dense/Tanh(1      !@9      !@A      !@I      !@a������G?i鈻#��?�Unknown
iHostMean"mean_squared_error/Mean(1      @9      @A      @I      @a���}%E?i�Ph m��?�Unknown
qHostDataset"Iterator::Model::ParallelMap(1      @9      @A      @I      @a�S4�C?ik}\��?�Unknown
�HostDataset"=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate(1������"@9������"@A������@I������@aլ�(B?iV���ޙ�?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@aT��[I�@?iö�i��?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1������@9������@A������@I������@a����@?i0�����?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1������@9������@A������@I������@a����@?i��b��?�Unknown
gHostStridedSlice"strided_slice(1333333@9333333@A333333@I333333@a?v�{�=?ite�̩�?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a~'Ro1<?i��O�R��?�Unknown
ZHostArgMax"ArgMax(1333333@9333333@A333333@I333333@a>��ɼ;?i�������?�Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a
�#��_9?i]����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1������@9������@A������@I������@a���W�8?iO������?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a���W�8?iA�����?�Unknown
XHostCast"Cast_2(1333333@9333333@A333333@I333333@a=��t�>8?i�hŢ��?�Unknown
� HostSquaredDifference"$mean_squared_error/SquaredDifference(1������@9������@A������@I������@aג�0��7?i�ykw��?�Unknown
|!HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a	���6?i�}�5���?�Unknown
X"HostEqual"Equal(1333333@9333333@A333333@I333333@a���c��5?i�����?�Unknown
v#HostDataset"!Iterator::Model::ParallelMap::Zip(1�����A@9�����A@A333333@I333333@a���c��5?i~x�c��?�Unknown
�$HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a���c��5?i��EO#��?�Unknown
\%HostArgMax"ArgMax_1(1ffffff@9ffffff@Affffff@Iffffff@a<��@m5?i��I����?�Unknown
�&HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a<��@m5?i��M�~��?�Unknown
i'HostTanh"sequential/dense_1/Tanh(1������@9������@A������@I������@a֟����4?iMT)<��?�Unknown
V(HostSum"Sum_2(1������@9������@A������@I������@ao�O��L4?iA>�ͣ��?�Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@a����+3?i6>I	��?�Unknown
�*HostDataset"?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333�?A333333@I333333�?a����+3?i+���n��?�Unknown
�+HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������	@9������	@A������	@I������	@aլ�(2?i!Ȱ)���?�Unknown
�,HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1������@9������@A������@I������@an�KB�z1?i������?�Unknown
�-HostDataset"MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a���Z0?iN0����?�Unknown
�.HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a�x��!R-?i�M���?�Unknown
X/HostCast"Cast_3(1ffffff@9ffffff@Affffff@Iffffff@aq�WA
�)?i��^��?�Unknown
�0HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@aq�WA
�)?i�!�����?�Unknown
w1HostReadVariableOp"div_no_nan_1/ReadVariableOp(1������@9������@A������@I������@a���W�(?i�����?�Unknown
u2HostSum"$mean_squared_error/weighted_loss/Sum(1������@9������@A������@I������@a���W�(?i�?����?�Unknown
}3HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a	���&?i�������?�Unknown
�4HostDataset"-Iterator::Model::ParallelMap::Zip[0]::FlatMap(1      &@9      &@A333333�?I333333�?a����+#?i^�hr���?�Unknown
}5HostTanhGrad"'gradient_tape/sequential/dense/TanhGrad(1�������?9�������?A�������?I�������?aլ�("?iY�$���?�Unknown
�6HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?aլ�("?iT�y����?�Unknown
X7HostCast"Cast_4(1      �?9      �?A      �?I      �?a��u� ?i�a�~��?�Unknown
8HostTanhGrad")gradient_tape/sequential/dense_1/TanhGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?asn_놓?iʼ ��?�Unknown
|9HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff�?9ffffff�?Affffff�?Iffffff�?asn_놓?i�H����?�Unknown
j:HostReadVariableOp"ReadVariableOp(1�������?9�������?A�������?I�������?a�x��!R?iA�VH���?�Unknown
u;HostSub"$gradient_tape/mean_squared_error/sub(1�������?9�������?A�������?I�������?a�x��!R?i��e����?�Unknown
}<HostRealDiv"(gradient_tape/mean_squared_error/truediv(1�������?9�������?A�������?I�������?a�x��!R?i9�tj���?�Unknown
b=HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a>��ɼ?i5�Z���?�Unknown
T>HostMul"Mul(1�������?9�������?A�������?I�������?a���W�?i��k[��?�Unknown
u?HostMul"$gradient_tape/mean_squared_error/Mul(1�������?9�������?A�������?I�������?a���W�?i-r��!��?�Unknown
�@HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      �?9      �?A      �?I      �?a	���?i*�kU���?�Unknown
oAHostSigmoid"sequential/dense_2/Sigmoid(1      �?9      �?A      �?I      �?a	���?i'� Ŋ��?�Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?ao�O��L?i��m)-��?�Unknown
XCHostCast"Cast_1(1�������?9�������?A�������?I�������?ao�O��L?i!iڍ���?�Unknown
DHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1�������?9�������?A�������?I�������?ao�O��L?i�#G�q��?�Unknown
wEHostMul"&gradient_tape/mean_squared_error/mul_1(1�������?9�������?A�������?I�������?ao�O��L?i޳V��?�Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?aլ�(?i�����?�Unknown
�GHostSigmoidGrad"4gradient_tape/sequential/dense_2/Sigmoid/SigmoidGrad(1�������?9�������?A�������?I�������?aլ�(?iF<	5��?�Unknown
`HHostDivNoNan"
div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?asn_놓?i��WW���?�Unknown
wIHostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?asn_놓?i�s�1��?�Unknown
uJHostSum"$gradient_tape/mean_squared_error/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?asn_놓?i�N����?�Unknown
�KHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?asn_놓?i��A.��?�Unknown
sLHostReadVariableOp"SGD/Cast/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a>��ɼ?i#�����?�Unknown
uMHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a>��ɼ?i	J����?�Unknown
�NHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a>��ɼ?iq�
s��?�Unknown
�OHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a>��ɼ?i�wM���?�Unknown
�PHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a>��ɼ?i�j�K��?�Unknown
yQHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a	���?i�_5ȥ��?�Unknown
wRHostCast"%gradient_tape/mean_squared_error/Cast(1      �?9      �?A      �?I      �?a	���?i�������?�Unknown*�M
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1fffff��@9fffff��@Afffff��@Ifffff��@a.��B��?i.��B��?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1ffffft�@9ffffft�@Affffft�@Iffffft�@aA� �v��?i�"�>��?�Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1�����Q�@9�����Q�@A�����Q�@I�����Q�@a�Q�é�?i�i�ԯ��?�Unknown
^HostGatherV2"GatherV2(1     ��@9     ��@A     ��@I     ��@a�C7���?i 򄻢��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1fffffFf@9fffffFf@AfffffFf@IfffffFf@a�m|���?i[h_?��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1������e@9������e@A������e@I������e@a�>:R��?iS:��\�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1fffff�`@9fffff�`@Afffff�`@Ifffff�`@ac���?i!���V��?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1fffff&O@9fffff&O@Afffff&O@Ifffff&O@a#w60)��?i�gyV�F�?�Unknown
{	HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1����̌F@9����̌F@A����̌F@I����̌F@a/
�(��z?i��|z|�?�Unknown
}
HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1     �D@9     �D@A     �D@I     �D@a#�dd��x?is����?�Unknown
�HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1fffff�;@9fffff�;@Afffff�;@Ifffff�;@a&��Ӥp?i`u�G��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(133333�7@933333�7@A33333�7@I33333�7@agm�Fl?il:⌎��?�Unknown
�HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1fffff�0@9fffff�0@Afffff�0@Ifffff�0@aY����)d?i��i���?�Unknown
�HostDataset"3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat(13333333@93333333@A������/@I������/@aE�H��b?i%�$L��?�Unknown
dHostDataset"Iterator::Model(1������4@9������4@A333333+@I333333+@a���9`?iۤ9�"�?�Unknown
fHostGreaterEqual"GreaterEqual(1333333&@9333333&@A333333&@I333333&@a3���|Z?i(�
0�?�Unknown
gHostTanh"sequential/dense/Tanh(1      !@9      !@A      !@I      !@a�c� hHT?iZ�+�.:�?�Unknown
iHostMean"mean_squared_error/Mean(1      @9      @A      @I      @av���Q?i����!C�?�Unknown
qHostDataset"Iterator::Model::ParallelMap(1      @9      @A      @I      @a:���P?iU�t�{K�?�Unknown
�HostDataset"=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate(1������"@9������"@A������@I������@a�G�X'�N?i��J^S�?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a Z�EH(L?i�g\p(Z�?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1������@9������@A������@I������@a���
�3K?i�'l�`�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1������@9������@A������@I������@a���
�3K?ib��g�g�?�Unknown
gHostStridedSlice"strided_slice(1333333@9333333@A333333@I333333@af�s�<KI?iKD7n�?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a	HK���G?i��dt�?�Unknown
ZHostArgMax"ArgMax(1333333@9333333@A333333@I333333@a˵��]�F?i�8|�y�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @aoZ]��yE?i���$�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1������@9������@A������@I������@aP����D?i���d��?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@aP����D?i)M�Ǥ��?�Unknown
XHostCast"Cast_2(1333333@9333333@A333333@I333333@a1ȗo~�D?i3$'Ǝ�?�Unknown
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1������@9������@A������@I������@a�4�QD?i[���ȓ�?�Unknown
| HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a�lo��C?i6������?�Unknown
X!HostEqual"Equal(1333333@9333333@A333333@I333333@a���˜B?i_��5��?�Unknown
v"HostDataset"!Iterator::Model::ParallelMap::Zip(1�����A@9�����A@A333333@I333333@a���˜B?i���ݡ�?�Unknown
�#HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a���˜B?i�%�R���?�Unknown
\$HostArgMax"ArgMax_1(1ffffff@9ffffff@Affffff@Iffffff@a�ک\�"B?i(P����?�Unknown
�%HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a�ک\�"B?i�zh����?�Unknown
i&HostTanh"sequential/dense_1/Tanh(1������@9������@A������@I������@axG�r�A?icL����?�Unknown
V'HostSum"Sum_2(1������@9������@A������@I������@aYH�!F.A?iuŠPK��?�Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@a���9@?i#���Y��?�Unknown
�)HostDataset"?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333�?A333333@I333333�?a���9@?i�TGh��?�Unknown
�*HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������	@9������	@A������	@I������	@a�G�X'�>?ik��9��?�Unknown
�+HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a|��Ζ=?i�(Å���?�Unknown
�,HostDataset"MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a �a��;?i�48Ib��?�Unknown
�-HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1������@9������@A������@I������@aG���8?i�67k|��?�Unknown
X.HostCast"Cast_3(1ffffff@9ffffff@Affffff@Iffffff@a�#�G�5?i/��:��?�Unknown
�/HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a�#�G�5?i'Il���?�Unknown
w0HostReadVariableOp"div_no_nan_1/ReadVariableOp(1������@9������@A������@I������@aP����4?iXƪa���?�Unknown
u1HostSum"$mean_squared_error/weighted_loss/Sum(1������@9������@A������@I������@aP����4?i�eW9��?�Unknown
}2HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a�lo��3?i�S6���?�Unknown
�3HostDataset"-Iterator::Model::ParallelMap::Zip[0]::FlatMap(1      &@9      &@A333333�?I333333�?a���90?io7�s���?�Unknown
}4HostTanhGrad"'gradient_tape/sequential/dense/TanhGrad(1�������?9�������?A�������?I�������?a�G�X'�.?i��1&���?�Unknown
�5HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?a�G�X'�.?i�M��t��?�Unknown
X6HostCast"Cast_4(1      �?9      �?A      �?I      �?a>#'�t�,?i)���>��?�Unknown
7HostTanhGrad")gradient_tape/sequential/dense_1/TanhGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���m¹*?i�Y����?�Unknown
|8HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���m¹*?i�3C8���?�Unknown
j9HostReadVariableOp"ReadVariableOp(1�������?9�������?A�������?I�������?aG���(?i��BI#��?�Unknown
u:HostSub"$gradient_tape/mean_squared_error/sub(1�������?9�������?A�������?I�������?aG���(?i�5BZ���?�Unknown
};HostRealDiv"(gradient_tape/mean_squared_error/truediv(1�������?9�������?A�������?I�������?aG���(?iӶAk=��?�Unknown
b<HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a˵��]�&?i.����?�Unknown
T=HostMul"Mul(1�������?9�������?A�������?I�������?aP����$?i׮�����?�Unknown
u>HostMul"$gradient_tape/mean_squared_error/Mul(1�������?9�������?A�������?I�������?aP����$?i�~{�K��?�Unknown
�?HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      �?9      �?A      �?I      �?a�lo��#?iw�V}��?�Unknown
o@HostSigmoid"sequential/dense_2/Sigmoid(1      �?9      �?A      �?I      �?a�lo��#?inl�Ů��?�Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?aYH�!F.!?i������?�Unknown
XBHostCast"Cast_1(1�������?9�������?A�������?I�������?aYH�!F.!?i��R����?�Unknown
CHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1�������?9�������?A�������?I�������?aYH�!F.!?i=Ǵr���?�Unknown
wDHostMul"&gradient_tape/mean_squared_error/mul_1(1�������?9�������?A�������?I�������?aYH�!F.!?i��W���?�Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a�G�X'�?i�Q����?�Unknown
�FHostSigmoidGrad"4gradient_tape/sequential/dense_2/Sigmoid/SigmoidGrad(1�������?9�������?A�������?I�������?a�G�X'�?i�p�	���?�Unknown
`GHostDivNoNan"
div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���m¹?i�ݟ׸��?�Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���m¹?ifJ�����?�Unknown
uIHostSum"$gradient_tape/mean_squared_error/Sum(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���m¹?iF��sd��?�Unknown
�JHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���m¹?i&$�A:��?�Unknown
sKHostReadVariableOp"SGD/Cast/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a˵��]�?iT8Ƅ���?�Unknown
uLHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a˵��]�?i�L�Ǩ��?�Unknown
�MHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a˵��]�?i�`�
`��?�Unknown
�NHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a˵��]�?i�t�M��?�Unknown
�OHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a˵��]�?i�v����?�Unknown
yPHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a�lo��?i�D;Hg��?�Unknown
wQHostCast"%gradient_tape/mean_squared_error/Cast(1      �?9      �?A      �?I      �?a�lo��?i     �?�Unknown