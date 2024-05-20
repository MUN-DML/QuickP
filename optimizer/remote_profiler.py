# https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace
import tensorflow as tf

# E.g. your worker IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you
# would like to schedule start of profiling 1 second from now, for a
# duration of 2 seconds.
options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)
options['delay_ms'] = 1000
tf.profiler.experimental.client.trace(
    'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',
    'gs://your_tb_dir',
    2000,
    options=options)