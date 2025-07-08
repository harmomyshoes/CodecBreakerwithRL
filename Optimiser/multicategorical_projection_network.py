import gin
import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.specs import distribution_spec

class MultiCategoricalDistributionBlock(tfp.distributions.Blockwise):

    def __init__(self, logits, categories_shape):
        self.categories_shape = categories_shape
        distribs = self._create_distrib(logits)
        super().__init__(distributions = distribs)
        self._add_logits_to_params(logits)

    def _create_distrib(self, logits):
        logits = tf.split(logits, self.categories_shape, -1)
        distribs = [tfp.distributions.Categorical(logits = logits_split) for 
                        logits_split in logits]
        return distribs

    def _mode(self):
        return self._flatten_and_concat_event(
            self._distribution.mode()
        )
    
    def _add_logits_to_params(self, logits):
        self._parameters['logits'] = logits

@gin.configurable
class MultiCategoricalProjectionNetwork(network.DistributionNetwork):
    """Generates a set of tfp.distribution.Categorical by predicting logits"""
    def __init__(self, 
                 sample_spec, 
                 logits_init_output_factor=0.1,
                 name='MultiCategoricalProjectionNetwork'):
    
        """Creates an instance of MultiCategoricalProjectionNetwork

        Args:
          sample_spec: A `tensor_spec.BoundedTensorSpec` detailing the shape
            dtypes of samples pulled from the output distribution.
          logits_init_output_factor: Output factor for initializing kernal 
            logits weights.
          name: A string representing the name of the network.
        """

        #self.categories_shape = sample_spec.maximum
        self.categories_shape = sample_spec.maximum - sample_spec.minimum + 1
        self.n_unique_categories = np.sum(self.categories_shape)

        output_spec = self._output_distribution_spec([self.n_unique_categories], sample_spec, name)

        super(MultiCategoricalProjectionNetwork, self).__init__(
            input_tensor_spec=None,
            state_spec = (),
            output_spec=output_spec,
            name=name
        )

        if not tensor_spec.is_bounded(sample_spec):
            raise ValueError(
                'sample_spec must be bounded. Got: %s.' % type(sample_spec))

        if not tensor_spec.is_discrete(sample_spec):
            raise ValueError('sample_spec must be discrete. Got: %s.' % sample_spec)        

        self._sample_spec = sample_spec

        self._projection_layer = tf.keras.layers.Dense(
            self.n_unique_categories,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=logits_init_output_factor),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='logits'
            )       

    def _output_distribution_spec(self, output_shape, sample_spec, network_name):
        input_param_spec = {
            'logits':
                tensor_spec.TensorSpec(
                    shape=output_shape,
                    dtype=tf.float32,
                    name=network_name + '_logits'
                )
        }
        return distribution_spec.DistributionSpec(
            MultiCategoricalDistributionBlock,
            input_param_spec,
            sample_spec=sample_spec,
            categories_shape = self.categories_shape)

    def call(self, inputs, outer_rank, training=False, mask=None):
        #masks not implemented yet
        
        batch_squash = utils.BatchSquash(outer_rank)
        inputs = batch_squash.flatten(inputs)
        inputs = tf.cast(inputs, tf.float32)

        logits = self._projection_layer(inputs, training=training)
        logits = tf.reshape(logits, [-1] + [self.n_unique_categories])
        logits = batch_squash.unflatten(logits)

        return self.output_spec.build_distribution(logits= logits), ()