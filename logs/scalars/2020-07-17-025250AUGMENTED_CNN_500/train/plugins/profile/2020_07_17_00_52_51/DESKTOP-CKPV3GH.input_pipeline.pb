	�Q�
@�Q�
@!�Q�
@	�S���7�?�S���7�?!�S���7�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�Q�
@�� ��?A�QI���@Y�D���J�?*	�������@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�>W[���?!A�7%�X@)�>W[���?1A�7%�X@:Preprocessing2F
Iterator::Model���_vO�?!���`>�?)�{�Pk�?1GWD�!�?:Preprocessing2P
Iterator::Model::Prefetchŏ1w-!o?!���E��?)ŏ1w-!o?1���E��?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�c�]K��?!�#�|�X@)�J�4a?1n����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*moderate2A4.1 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�� ��?�� ��?!�� ��?      ��!       "      ��!       *      ��!       2	�QI���@�QI���@!�QI���@:      ��!       B      ��!       J	�D���J�?�D���J�?!�D���J�?R      ��!       Z	�D���J�?�D���J�?!�D���J�?JCPU_ONLY