	?���??���?!?���?	x��{�u2@x��{�u2@!x��{�u2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?���?�b�=y�?Aݵ�|г�?Y�HP��?*	������V@2F
Iterator::Model���S㥫?!�����M@)��@��Ǩ?1��>���J@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!���#89@)�g��s��?1�#���>7@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat;�O��n�?!�k(���#@)vq�-�?12��tS!@:Preprocessing2S
Iterator::Model::ParallelMapǺ���v?!��#��@)Ǻ���v?1��#��@:Preprocessing2X
!Iterator::Model::ParallelMap::ZipΈ����?!Q^CyeD@)_�Q�k?11��t�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapa��+e�?!�YLg1;@)��H�}]?1Cy�5��?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor/n��R?!�t�YL�?)/n��R?1�t�YL�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor����MbP?!���b:��?)����MbP?1���b:��?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C��6J?!�}��?)-C��6J?1�}��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 18.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2B31.8 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�b�=y�?�b�=y�?!�b�=y�?      ��!       "      ��!       *      ��!       2	ݵ�|г�?ݵ�|г�?!ݵ�|г�?:      ��!       B      ��!       J	�HP��?�HP��?!�HP��?R      ��!       Z	�HP��?�HP��?!�HP��?JCPU_ONLY