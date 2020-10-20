
from mxnet import gluon, ndarray, symbol
from mxnet.gluon import rnn

def _get_rnn_cell(mode, num_layers, input_size, hidden_size,
                  dropout, var_drop_in, var_drop_state, var_drop_out,
                  skip_connection, proj_size=None):
    """create rnn cell given specs

    Parameters
    ----------
    mode : str
        The type of RNN cell to use. Options are 'rnn_tanh', 'rnn_relu', 'lstm', 'lstmp', 'gru'.
    num_layers : int
        The number of RNN cells in the encoder.
    input_size : int
        The initial input size of in the RNN cell.
    hidden_size : int
        The hidden size of the RNN cell.
    dropout : float
        The dropout rate to use for encoder output.
    var_drop_in: float
        The variational dropout rate for inputs. Won’t apply dropout if it equals 0.
    var_drop_state: float
        The variational dropout rate for state inputs on the first state channel.
        Won’t apply dropout if it equals 0.
    var_drop_out: float
        The variational dropout rate for outputs. Won’t apply dropout if it equals 0.
    skip_connection : bool
        Whether to add skip connections (add RNN cell input to output)
    proj_size : int
        The projection size of each LSTMPCell cell.
        Only available when the mode=lstmpc.

    """

    if mode == 'lstmps': 
        assert proj_size is not None, \
            'proj_size takes effect only when mode is lstmp'

    rnn_cell = rnn.HybridSequentialRNNCell()
    with rnn_cell.name_scope():
        for i in range(num_layers):
            if mode == 'rnn_relu':
                cell = rnn.RNNCell(hidden_size, 'relu', input_size=input_size)
            elif mode == 'rnn_tanh':
                cell = rnn.RNNCell(hidden_size, 'tanh', input_size=input_size)
            elif mode == 'lstm':
                cell = rnn.LSTMCell(hidden_size, input_size=input_size)
            elif mode == 'lstmp':
                cell = gluon.contrib.rnn.LSTMPCell(hidden_size, input_size, proj_size)
            elif mode == 'gru':
                cell = rnn.GRUCell(hidden_size, input_size=input_size)
           
            if var_drop_in + var_drop_state + var_drop_out != 0:
                cell = gluon.contrib.rnn.VariationalDropoutCell(cell,
                                                          var_drop_in,
                                                          var_drop_state,
                                                          var_drop_out)

            if skip_connection:
                cell = rnn.ResidualCell(cell)

            rnn_cell.add(cell)

            if i != num_layers - 1 and dropout != 0:
                rnn_cell.add(rnn.DropoutCell(dropout))

    return rnn_cell

class RNNLayer(gluon.HybridBlock):
    r"""
    Parameters
    ----------
    mode : str
        The type of RNN cell to use. Options are 'rnn_tanh', 'rnn_relu', 'lstm', 'lstmp', 'gru'.
    num_layers : int
        The number of RNN cells in the encoder.
    hidden_size : int
        The hidden size of the RNN cell.
    dropout : float
        The dropout rate to use for encoder output.
    layout : str, default 'TNC'
        The format of input and output tensors. T, N and C stand for
        sequence length, batch size, and feature dimensions respectively.
    skip_connection : bool
        Whether to add skip connections (add RNN cell input to output)
    input_size : int
        The initial input size of in the RNN cell.
    proj_size : int
        The projection size of each LSTMPCell cell
    bidirectional: bool, default False
        If `True`, becomes a bidirectional RNN.
    
    """
    def __init__(self, mode, num_layers, hidden_size, dropout=0.0, layout='TNC',
                 skip_connection=True, input_size=0, proj_size=None, bidirectional=False, **kwargs):
        super(RNNLayer, self).__init__(**kwargs)

        assert layout in ('TNC', 'NTC'), \
            "Invalid layout %s; must be one of ['TNC' or 'NTC']"%layout
        
        self._layout = layout
        self._mode = mode
        self._num_layers = num_layers
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._skip_connection = skip_connection
        self._proj_size = proj_size
        self._bidirectional = bidirectional

        with self.name_scope():
            input_size = self._input_size
            self.forward_layers = self._make_layers(mode, num_layers, 
                                                    input_size, hidden_size, 
                                                    dropout, 0, 0, 0,
                                                    skip_connection, proj_size)
            if bidirectional:
                input_size = self._input_size
                self.backward_layers = self._make_layers(mode, num_layers, 
                                                        input_size, hidden_size,  
                                                        dropout, 0, 0, 0,
                                                        skip_connection, proj_size)


    def _make_layers(self, mode, num_layers, input_size, hidden_size,
                     dropout, var_drop_in, var_drop_state, var_drop_out,
                     skip_connection, proj_size=None):
        layers = rnn.HybridSequentialRNNCell()
        for idx in range(num_layers):
            if idx == num_layers - 1:
                dropout = 0
            if idx == 0:
                sc = False
            else:
                sc = skip_connection
            layer = _get_rnn_cell(mode, 1, input_size, hidden_size, dropout, 
                                    var_drop_in, var_drop_state, var_drop_out,
                                    sc, proj_size)
            layers.add(layer)
            input_size = hidden_size
        return layers

    def begin_state(self, func, **kwargs):
        forward_state = [self.forward_layers[0][0].begin_state(func=func, **kwargs)
                            for _ in range(self._num_layers)]
        if not self._bidirectional:
            return [forward_state]
        
        backward_state = [self.backward_layers[0][0].begin_state(func=func, **kwargs)
                            for _ in range(self._num_layers)]

        return [forward_state, backward_state]

    def hybrid_forward(self, F, inputs, states, mask=None):
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Defines the forward computation for cache cell. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.

        Parameters
        ----------
        inputs : NDArray
            The input data layout='TNC'.
        states : Tuple[List[List[NDArray]]]
            The states. including:
            states[0] indicates the states used in forward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
            states[1] indicates the states used in backward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).

        Returns
        --------
        out: NDArray
            The output data with shape (num_layers, seq_len, batch_size).
        [states_forward, states_backward] : List
            Including:
            states_forward: The out states from forward layer,
            which has the same structure with *states[0]*.
            states_backward: The out states from backward layer,
            which has the same structure with *states[1]*.
        mask:NDArray
            (batch_size, num_steps)
        """
        
        if mask is not None:
            sequence_length = mask.sum(axis=1)
        
        if self._layout == 'NTC':
            inputs = F.swapaxes(inputs, dim1=0, dim2=1)

        outputs_forward = []
        outputs_backward = []

        for layer_index in range(self._num_layers):
            if layer_index == 0:
                layer_inputs = inputs
            else:
                layer_inputs = outputs_forward[layer_index-1]
            output, states[0][layer_index] = F.contrib.foreach(
                self.forward_layers[layer_index],
                layer_inputs,
                states[0][layer_index])
            outputs_forward.append(output)
            if not self._bidirectional:
                continue

            if layer_index == 0:
                layer_inputs = inputs
            else:
                layer_inputs = outputs_backward[layer_index-1]

            if mask is not None:
                layer_inputs = F.SequenceReverse(layer_inputs,
                                                 sequence_length=sequence_length,
                                                 use_sequence_length=True, axis=0)
            else:
                layer_inputs = F.SequenceReverse(layer_inputs, axis=0)
            output, states[1][layer_index] = F.contrib.foreach(
                self.backward_layers[layer_index],
                layer_inputs,
                states[1][layer_index])
            if mask is not None:
                backward_out = F.SequenceReverse(output,
                                                 sequence_length=sequence_length,
                                                 use_sequence_length=True, axis=0)
            else:
                backward_out = F.SequenceReverse(output, axis=0)
            outputs_backward.append(backward_out)
        
        if self._bidirectional:
            out = F.concat(*[outputs_forward[-1], outputs_backward[-1]], dim=-1)
        else:
            out = outputs_forward[-1]
        if self._layout == 'NTC':
            out = F.swapaxes(out, dim1=0, dim2=1)
        return out, states

    def __call__(self, inputs, states=None, mask=None, **kwargs):
        if states is None:
            if isinstance(inputs, ndarray.NDArray):
                batch_size = inputs.shape[0]
                states = self.begin_state(func=ndarray.zeros, ctx=inputs.context, 
                                          batch_size=batch_size, dtype=inputs.dtype)
            else:
                states = self.begin_state(func=symbol.zeros)
           
        return super(RNNLayer, self).__call__(inputs, states, mask, **kwargs)