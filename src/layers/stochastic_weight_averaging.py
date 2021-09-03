from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator


class StochasticWeightAveraging(object):
    """
    Stochastic Weight Averaging ensemble method (Izmailov et al.,  https://arxiv.org/abs/1803.05407)

    SWA averages weights during the learning procedure.

    SWA combines the idea of averaging models within the weight space and a specific learning rate scheduling.
    The goal is to average models from a region around the same local minimum. (you can't average models from different
    local minimum, it will produce an averaged model placed outside of theses regions). This scheduling aims to go down
    in one local minimum when SWA starts.

    Evaluations that use averaged parameters require to fit Batch Norm statistics (mean and variance): Moving Free Batch
    Norm layer (in moving_free_batch_normalization.py) provides an easy way to perform this operation.

    The `apply()` methods adds shadow copies of the trained variables and add ops that averaged the variables in their
    shadow copies. The `apply()` might be used several times at the end of the training when falling in a local minimum
    are. The average() and average_name() methods give access to the shadow variables and their names. They are useful
    when building an evaluation model, or when restoring a model from a checkpoint file. They help use the moving
    averages in place of the last trained values for evaluations.

    SWA is different from Exponential Moving Average (EMA) method, which perform exponential averages of the trainable
    weights. Here it's a classic average. To compute the averages when `apply()` is called, `_n_models` stores the
    actual number of updates.

    To evaluate the model with SWA weights, you can:
    *   Replace variables used in the model with `assign()` operations.
    *   Build a model that uses the shadow variables instead of the variables.
        For this, use the `average()` method which returns the shadow variable
        for a given variable.
    *   Build a model normally but load the checkpoint files to evaluate by using
        the shadow variable names.  For this use the `average_name()` method.  See
        the `tf.train.Saver` for more
        information on restoring saved variables.


    SWA might be used to all trainable variables involved in the model, so be careful when providing `var_list` in
    `apply()` method.

    """

    def __init__(self, name="StochasticWeightAveraging"):
        self._name = name
        self._averages = {}
        self._n_models = None

    def apply(self, var_list=None):

        if var_list is None:
            var_list = variables.trainable_variables()

        for var in var_list:
            if var.dtype.base_dtype not in [dtypes.float16, dtypes.float32,
                                            dtypes.float64]:
                raise TypeError("The variables must be half, float, or double: %s" %
                                var.name)

            if var not in self._averages:
                # For variables: to lower communication bandwidth across devices we keep
                # the moving averages on the same device as the variables. For other
                # tensors, we rely on the existing device allocation mechanism.
                with ops.init_scope():
                    if isinstance(var, variables.Variable):
                        avg = slot_creator.create_slot(var,
                                                       var.initialized_value(),
                                                       self.name,
                                                       colocate_with_primary=True)
                        # NOTE(mrry): We only add `tf.Variable` objects to the
                        # `MOVING_AVERAGE_VARIABLES` collection.
                        ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
                    else:
                        avg = slot_creator.create_zeros_slot(
                            var,
                            self.name,
                            colocate_with_primary=(var.op.type in ["Variable",
                                                                   "VariableV2",
                                                                   "VarHandleOp"]))
                self._averages[var] = avg

        with ops.device('/cpu:0'):
            self._n_models = variable_scope.get_variable(shape=[],
                                                         dtype=dtypes.float32,
                                                         name='n_models',
                                                         initializer=init_ops.constant_initializer(0.),
                                                         trainable=False)

        with ops.name_scope(self.name) as scope:
            updates = []
            for var in var_list:
                updates.append(assign_stochastic_average(self._averages[var], var, self._n_models))
            with ops.control_dependencies(updates):
                update_n_models = state_ops.assign_add(self._n_models, 1., name=scope)
            return update_n_models

    @property
    def name(self):
        return self._name

    @property
    def n_models(self):
        return self._n_models

    def average(self, var):
        """Returns the `Variable` holding the average of `var`.
        Args:
          var: A `Variable` object.
        Returns:
          A `Variable` object or `None` if the moving average of `var`
          is not maintained.
        """
        return self._averages.get(var, None)

    def average_name(self, var):
        """Returns the name of the `Variable` holding the average for `var`.
        The typical scenario for `ExponentialMovingAverage` is to compute moving
        averages of variables during training, and restore the variables from the
        computed moving averages during evaluations.
        To restore variables, you have to know the name of the shadow variables.
        That name and the original variable can then be passed to a `Saver()` object
        to restore the variable from the moving average value with:
          `saver = tf.train.Saver({ema.average_name(var): var})`
        `average_name()` can be called whether or not `apply()` has been called.
        Args:
          var: A `Variable` object.
        Returns:
          A string: The name of the variable that will be used or was used
          by the `ExponentialMovingAverage class` to hold the moving average of
          `var`.
        """
        if var in self._averages:
            return self._averages[var].op.name
        return ops.get_default_graph().unique_name(
            var.op.name + "/" + self.name, mark_as_used=False)

    def variables_to_restore(self, moving_avg_variables=None):
        """Returns a map of names to `Variables` to restore.
        If a variable has a moving average, use the moving average variable name as
        the restore name; otherwise, use the variable name.
        For example,
        ```python
          variables_to_restore = ema.variables_to_restore()
          saver = tf.train.Saver(variables_to_restore)
        ```
        Below is an example of such mapping:
        ```
          conv/batchnorm/gamma/ExponentialMovingAverage: conv/batchnorm/gamma,
          conv_4/conv2d_params/ExponentialMovingAverage: conv_4/conv2d_params,
          global_step: global_step
        ```
        Args:
          moving_avg_variables: a list of variables that require to use of the
            moving variable name to be restored. If None, it will default to
            variables.moving_average_variables() + variables.trainable_variables()
        Returns:
          A map from restore_names to variables. The restore_name can be the
          moving_average version of the variable name if it exist, or the original
          variable name.
        """
        name_map = {}
        if moving_avg_variables is None:
            # Include trainable variables and variables which have been explicitly
            # added to the moving_average_variables collection.
            moving_avg_variables = variables.trainable_variables()
            moving_avg_variables += variables.moving_average_variables()
        # Remove duplicates
        moving_avg_variables = set(moving_avg_variables)
        # Collect all the variables with moving average,
        for v in moving_avg_variables:
            name_map[self.average_name(v)] = v
        # Make sure we restore variables without moving averages as well.
        moving_avg_variable_names = set([v.name for v in moving_avg_variables])
        for v in list(set(variables.global_variables())):
            if v.name not in moving_avg_variable_names and v.op.name not in name_map:
                name_map[v.op.name] = v
        return name_map


def assign_stochastic_average(variable, value, n_model, name=None):
    with ops.name_scope(name, "AssignStochasticAvg", [variable, value, n_model]) as scope:
        with ops.colocate_with(variable):
            variable_swa = variable*n_model + value
            variable_swa /= n_model + 1
            return state_ops.assign(variable, variable_swa, name=scope)