"""
References:
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/exporting_models.md
https://github.com/tensorflow/models/issues/1988

Unfortunately, the tutorial for saving a model for inference "freezes" the
variables in place and makes them unservable by tensorflow_serving.
export_inference_graph.py exports an empty "variables" directory, which needs to
be populated.

The below script, which is a modified version of export_inference_graph, will
save the model in an "unfrozen" state, servable via TensorFlow Serving.
"""

import logging
import os
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.exporter import (input_placeholder_fn_map,
                                       add_output_tensor_nodes,
                                       write_graph_and_checkpoint)
                                       


flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('input_shape', None,
                    'If input_type is `image_tensor`, this can explicitly set '
                    'the shape of this input tensor to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. If not specified, for an `image_tensor, the '
                    'default shape will be partially specified as '
                    '`[None, None, None, 3]`.')
flags.DEFINE_string('pipeline_config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')

flags.DEFINE_boolean('use_side_inputs', False,
                     'If True, uses side inputs as well as image inputs.')
flags.DEFINE_string('side_input_shapes', None,
                    'If use_side_inputs is True, this explicitly sets '
                    'the shape of the side input tensors to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. A `/` denotes a break, starting the shape of '
                    'the next side input tensor. This flag is required if '
                    'using side inputs.')
flags.DEFINE_string('side_input_types', None,
                    'If use_side_inputs is True, this explicitly sets '
                    'the type of the side input tensors. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of types, each of `string`, `integer`, or `float`. '
                    'This flag is required if using side inputs.')
flags.DEFINE_string('side_input_names', None,
                    'If use_side_inputs is True, this explicitly sets '
                    'the names of the side input tensors required by the model '
                    'assuming the names will be a comma-separated list of '
                    'strings. This flag is required if using side inputs.')
flags.DEFINE_string('additional_output_tensor_names', None,
                    'Additional Tensors to output, to be specified as a comma '
                    'separated list of tensor names.')
FLAGS = flags.FLAGS


def _write_saved_model(saved_model_path,
                       trained_checkpoint_prefix,
                       inputs,
                       outputs):
    """Writes SavedModel to disk.
    Args:
      saved_model_path: Path to write SavedModel.
      trained_checkpoint_prefix: path to trained_checkpoint_prefix.
      inputs: The input image tensor to use for detection.
      outputs: A tensor dictionary containing the outputs of a DetectionModel.
    """
    saver = tf.train.Saver()
    with session.Session() as sess:
        saver.restore(sess, trained_checkpoint_prefix)
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

        tensor_info_inputs = {
              'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
        tensor_info_outputs = {}
        for k, v in outputs.items():
            tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

        detection_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                  inputs=tensor_info_inputs,
                  outputs=tensor_info_outputs,
                  method_name=signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.tag_constants.SERVING],
              signature_def_map={
                  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                      detection_signature,
              },
          )
        builder.save()


def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            optimize_graph=False,
                            output_collection_name='inference_op'):
    """Export helper."""
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory,
                                     'frozen_inference_graph.pb')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    model_path = os.path.join(output_directory, 'model.ckpt')

    if input_type not in input_placeholder_fn_map:
        raise ValueError('Unknown input type: {}'.format(input_type))
    placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type]()
    inputs = tf.to_float(input_tensors)
    preprocessed_inputs = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(preprocessed_inputs)
    postprocessed_tensors = detection_model.postprocess(output_tensors)
    outputs = _add_output_tensor_nodes(postprocessed_tensors,
                                       output_collection_name)

    saver = None
    if use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver()
    input_saver_def = saver.as_saver_def()

    _write_graph_and_checkpoint(
        inference_graph_def=tf.get_default_graph().as_graph_def(),
        model_path=model_path,
        input_saver_def=input_saver_def,
        trained_checkpoint_prefix=trained_checkpoint_prefix)

    _write_saved_model(saved_model_path,
                       trained_checkpoint_prefix,
                       inputs,
                       outputs)


def export_inference_graph(input_type,
                           pipeline_config,
                           trained_checkpoint_prefix,
                           output_directory,
                           optimize_graph=False,
                           output_collection_name='inference_op'):
    """Exports inference graph for the model specified in the pipeline config.

    Args:
      input_type: Type of input for the graph. Can be one of [`image_tensor`,
        `tf_example`].
      pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
      trained_checkpoint_prefix: Path to the trained checkpoint file.
      output_directory: Path to write outputs.
      optimize_graph: Whether to optimize graph using Grappler.
      output_collection_name: Name of collection to add output tensors to.
        If None, does not add output tensors to a collection.
    """
    detection_model = model_builder.build(pipeline_config.model,
                                          is_training=False)
    _export_inference_graph(input_type, detection_model,
                            pipeline_config.eval_config.use_moving_averages,
                            trained_checkpoint_prefix, output_directory,
                            optimize_graph, output_collection_name)


def main(_):
    assert FLAGS.pipeline_config_path, '`pipeline_config_path` is missing'
    assert FLAGS.trained_checkpoint_prefix, (
           '`trained_checkpoint_prefix` is missing')
    assert FLAGS.output_directory, '`output_directory` is missing'

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.input_shape.split(',')
        ]
    else:
        input_shape = None
    if FLAGS.use_side_inputs:
        side_input_shapes, side_input_names, side_input_types = (
            exporter.parse_side_inputs(
                FLAGS.side_input_shapes,
                FLAGS.side_input_names,
                FLAGS.side_input_types))
    else:
        side_input_shapes = None
        side_input_names = None
        side_input_types = None    
     
    if FLAGS.additional_output_tensor_names:
        additional_output_tensor_names = list(
            FLAGS.additional_output_tensor_names.split(','))
    else:
        additional_output_tensor_names = None
    
    export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
        FLAGS.output_directory)


if __name__ == '__main__':
    tf.app.run()
