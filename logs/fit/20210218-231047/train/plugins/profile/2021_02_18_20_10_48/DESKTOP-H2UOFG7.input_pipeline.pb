	46<�;@46<�;@!46<�;@	��,�m��?��,�m��?!��,�m��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$46<�;@9��v���?A+��	�;@Y����?*	fffff�R@2F
Iterator::Modeln���?!�ț9J@)tF��_�?1�fN��?@:Preprocessing2U
Iterator::Model::ParallelMapV2� �	��?!$�A���4@)� �	��?1$�A���4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���H�?!bM�.�+5@)��<,Ԋ?1rn=�q1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatelxz�,C�?!-&p� `2@);�O��n�?1�t����'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*�s?!��ӝߐ@)a2U0*�s?1��ӝߐ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQ�|a�?!u�7d��G@)HP�s�r?1���+�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ���f?!{�v���@)Ǻ���f?1{�v���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap� �	��?!$�A���4@)-C��6Z?1���?@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��,�m��?I�i=I��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	9��v���?9��v���?!9��v���?      ��!       "      ��!       *      ��!       2	+��	�;@+��	�;@!+��	�;@:      ��!       B      ��!       J	����?����?!����?R      ��!       Z	����?����?!����?b      ��!       JCPU_ONLYY��,�m��?b q�i=I��X@