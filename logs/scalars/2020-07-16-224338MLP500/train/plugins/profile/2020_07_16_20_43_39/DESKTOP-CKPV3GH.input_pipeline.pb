	�����@�����@!�����@	�匃Y}�?�匃Y}�?!�匃Y}�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�����@�3��7��?A�Ǻ�M@Y��#����?*	�����YK@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeata2U0*��?!��ތA@)vq�-�?1Hb���<@:Preprocessing2F
Iterator::Model�0�*�?!��y�B@)_�Qڋ?1x uB��8@:Preprocessing2S
Iterator::Model::ParallelMapy�&1�|?!NSZ5�)@)y�&1�|?1NSZ5�)@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateΈ����?!��S�w 1@)-C��6z?1�Z�\~f'@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�R�!�u�?!��nQ�+O@)�q����o?1��� �@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor_�Q�k?!x uB��@)_�Q�k?1x uB��@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�����g?!Hb���4@)�����g?1Hb���4@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�I+��?!f��4@)_�Q�[?1x uB��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�3��7��?�3��7��?!�3��7��?      ��!       "      ��!       *      ��!       2	�Ǻ�M@�Ǻ�M@!�Ǻ�M@:      ��!       B      ��!       J	��#����?��#����?!��#����?R      ��!       Z	��#����?��#����?!��#����?JCPU_ONLY