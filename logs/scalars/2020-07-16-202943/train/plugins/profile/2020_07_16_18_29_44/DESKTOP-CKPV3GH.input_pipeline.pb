	46<��%@46<��%@!46<��%@	��u+��?��u+��?!��u+��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$46<��%@�-����?An��R%@Y�?Ƭ?*	������Q@2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenatevq�-�?!s���0F@)���B�i�?1�MmjS�D@:Preprocessing2F
Iterator::ModelM�St$�?!�^��׽?@)�q����?1����5@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatA��ǘ��?!��./@)�j+��݃?1���G?+@:Preprocessing2S
Iterator::Model::ParallelMapy�&1�|?!�Ԧ6��#@)y�&1�|?1�Ԧ6��#@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip��+e�?!E(B�Q@)�J�4q?19�as�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��镲�?!B�P�bG@)_�Q�[?1�g<�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorǺ���V?!��׽�u�?)Ǻ���V?1��׽�u�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor/n��R?!\���?)/n��R?1\���?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C��6J?! ��G?��?)-C��6J?1 ��G?��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�-����?�-����?!�-����?      ��!       "      ��!       *      ��!       2	n��R%@n��R%@!n��R%@:      ��!       B      ��!       J	�?Ƭ?�?Ƭ?!�?Ƭ?R      ��!       Z	�?Ƭ?�?Ƭ?!�?Ƭ?JCPU_ONLY