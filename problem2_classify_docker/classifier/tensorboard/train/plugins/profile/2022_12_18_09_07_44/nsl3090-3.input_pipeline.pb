	��Ӹ7�>@��Ӹ7�>@!��Ӹ7�>@      ��!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'��Ӹ7�>@�Y���
�?1�G�3@I�P�J%@r0*	�(\���]@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat)<hv�[�?!6���B<@)��~31]�?1S4��U�:@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�0e���?!#�ĔQB@)�A���?1�$�Yh,:@:Preprocessing2T
Iterator::Root::ParallelMapV29+�&�|�?!�C_��0@)9+�&�|�?1�C_��0@:Preprocessing2E
Iterator::Rootn��)"�?!!�&?@)���XǑ?1纫���,@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�ƻ#c��?!�&�_��$@)�ƻ#c��?1�&�_��$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�(�'�$�?!�wZ6Q@)�|@�3is?1Ʉ��@�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor31]��_?!P �
��?)31]��_?1P �
��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�34.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIv
��>�A@Q�z9��P@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Y���
�?�Y���
�?!�Y���
�?      ��!       "	�G�3@�G�3@!�G�3@*      ��!       2      ��!       :	�P�J%@�P�J%@!�P�J%@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qv
��>�A@y�z9��P@