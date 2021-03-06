��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq c__main__
Net
qX,   D:\Desk\DTU\DeepLearn\DLProj\convlstmpara.pyqX  class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv1d(in_channels = 1,
                               out_channels = 10,
                               kernel_size = 15,
                               stride = 1)
        
        self.pool1 = nn.MaxPool1d(4, stride = 2)
        
        self.bn1 = nn.BatchNorm1d(10)
        
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 30, num_layers = 1, bias=True, dropout=0.5)
        
        self.l_out = nn.Linear(in_features=1200,
                            out_features=output_size,
                            bias=False)
        
        
    def forward(self, x):
        features = []
        out = {}
        
        # Output layer
        y = self.conv1(x)
        y = self.dropout(y)
        y = self.bn1(y)
        y = relu(y)
        y = self.pool1(y)
        y = y.view(-1, 10*117)
        features.append(y)
        
        z = x.permute(2,0,1)        
        z, (h, c) = self.lstm(z)
        z = h.view(-1, 30)
        z = relu(z)
        features.append(z)
        
        features_final = torch.cat(features, dim=1)
        
        out = self.l_out(features_final)
        out = torch.sigmoid(out)
        return out
qtqQ)�q}q(X   _backendqctorch.nn.backends.thnn
_get_thnn_function_backend
q)Rq	X   _parametersq
ccollections
OrderedDict
q)RqX   _buffersqh)RqX   _backward_hooksqh)RqX   _forward_hooksqh)RqX   _forward_pre_hooksqh)RqX   _state_dict_hooksqh)RqX   _load_state_dict_pre_hooksqh)RqX   _modulesqh)Rq(X   dropoutq(h ctorch.nn.modules.dropout
Dropout
qXM   C:\Program Files (x86)\Anaconda\lib\site-packages\torch\nn\modules\dropout.pyqX5  class Dropout(_DropoutNd):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)
qtqQ)�q }q!(hh	h
h)Rq"hh)Rq#hh)Rq$hh)Rq%hh)Rq&hh)Rq'hh)Rq(hh)Rq)X   trainingq*�X   pq+G?�      X   inplaceq,�ubX   conv1q-(h ctorch.nn.modules.conv
Conv1d
q.XJ   C:\Program Files (x86)\Anaconda\lib\site-packages\torch\nn\modules\conv.pyq/X�  class Conv1d(_ConvNd):
    r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters,
          of size
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    .. note::

        Depending of the size of your kernel, several (of the last)
        columns of the input might be lost, because it is a valid
        `cross-correlation`_, and not a full `cross-correlation`_.
        It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(C_\text{in}=C_{in}, C_\text{out}=C_{in} \times K, ..., \text{groups}=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \text{kernel\_size}}`

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
q0tq1Q)�q2}q3(hh	h
h)Rq4(X   weightq5ctorch._utils
_rebuild_parameter
q6ctorch._utils
_rebuild_tensor_v2
q7((X   storageq8ctorch
FloatStorage
q9X   2034580781152q:X   cuda:0q;K�Ntq<QK K
KK�q=KKK�q>�h)Rq?tq@RqA�h)RqB�qCRqDX   biasqEh6h7((h8h9X   2034580781440qFX   cuda:0qGK
NtqHQK K
�qIK�qJ�h)RqKtqLRqM�h)RqN�qORqPuhh)RqQhh)RqRhh)RqShh)RqThh)RqUhh)RqVhh)RqWh*�X   in_channelsqXKX   out_channelsqYK
X   kernel_sizeqZK�q[X   strideq\K�q]X   paddingq^K �q_X   dilationq`K�qaX
   transposedqb�X   output_paddingqcK �qdX   groupsqeKX   padding_modeqfX   zerosqgubX   pool1qh(h ctorch.nn.modules.pooling
MaxPool1d
qiXM   C:\Program Files (x86)\Anaconda\lib\site-packages\torch\nn\modules\pooling.pyqjX  class MaxPool1d(_MaxPoolNd):
    r"""Applies a 1D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Examples::

        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def forward(self, input):
        return F.max_pool1d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
qktqlQ)�qm}qn(hh	h
h)Rqohh)Rqphh)Rqqhh)Rqrhh)Rqshh)Rqthh)Rquhh)Rqvh*�hZKh\Kh^K h`KX   return_indicesqw�X	   ceil_modeqx�ubX   bn1qy(h ctorch.nn.modules.batchnorm
BatchNorm1d
qzXO   C:\Program Files (x86)\Anaconda\lib\site-packages\torch\nn\modules\batchnorm.pyq{XV  class BatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
q|tq}Q)�q~}q(hh	h
h)Rq�(h5h6h7((h8h9X   2034580778656q�X   cuda:0q�K
Ntq�QK K
�q�K�q��h)Rq�tq�Rq��h)Rq��q�Rq�hEh6h7((h8h9X   2034580782592q�X   cuda:0q�K
Ntq�QK K
�q�K�q��h)Rq�tq�Rq��h)Rq��q�Rq�uhh)Rq�(X   running_meanq�h7((h8h9X   2034580780672q�X   cuda:0q�K
Ntq�QK K
�q�K�q��h)Rq�tq�Rq�X   running_varq�h7((h8h9X   2034580781536q�X   cuda:0q�K
Ntq�QK K
�q�K�q��h)Rq�tq�Rq�X   num_batches_trackedq�h7((h8ctorch
LongStorage
q�X   2034580777120q�X   cuda:0q�KNtq�QK ))�h)Rq�tq�Rq�uhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�h*�X   num_featuresq�K
X   epsq�G>�����h�X   momentumq�G?�������X   affineq��X   track_running_statsq��ubX   lstmq�(h ctorch.nn.modules.rnn
LSTM
q�XI   C:\Program Files (x86)\Anaconda\lib\site-packages\torch\nn\modules\rnn.pyq�X'$  class LSTM(RNNBase):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{(t-1)}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the cell state for `t = seq_len`.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. include:: cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """
    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tuple[Tensor, Tensor], Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    def permute_hidden(self, hx, permutation):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor], int, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, self._get_flat_weights(), self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, self._get_flat_weights(), self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]

        return output, hidden

    @torch._jit_internal.export
    def forward_tensor(self, input, hx=None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)

        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch._jit_internal.export
    def forward_packed(self, input, hx=None):
        # type: (PackedSequence, Optional[Tuple[Tensor, Tensor]]) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]  # noqa
        input, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)

        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)

        output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch._jit_internal.ignore
    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)
q�tq�Q)�q�}q�(hh	h
h)Rq�(X   weight_ih_l0q�h6h7((h8h9X   2034580778560q�X   cuda:0q�MxNtq�QK KxK�q�KK�qʉh)Rq�tq�Rq͈h)Rq·q�Rq�X   weight_hh_l0q�h6h7((h8h9X   2034580778560q�X   cuda:0q�MxNtq�QKxKxK�q�KK�q։h)Rq�tq�Rqوh)Rqڇq�Rq�X
   bias_ih_l0q�h6h7((h8h9X   2034580778560q�X   cuda:0q�MxNtq�QM�Kx�q�K�q�h)Rq�tq�Rq�h)Rq�q�Rq�X
   bias_hh_l0q�h6h7((h8h9X   2034580778560q�X   cuda:0q�MxNtq�QM Kx�q�K�q�h)Rq�tq�Rq�h)Rq�q�Rq�uhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�h*�X   modeq�X   LSTMq�X
   input_sizeq�KX   hidden_sizeq�KX
   num_layersr   KhE�X   batch_firstr  �hG?�      X   bidirectionalr  �X   _all_weightsr  ]r  ]r  (X   weight_ih_l0r  X   weight_hh_l0r  X
   bias_ih_l0r  X
   bias_hh_l0r	  eaubX   l_outr
  (h ctorch.nn.modules.linear
Linear
r  XL   C:\Program Files (x86)\Anaconda\lib\site-packages\torch\nn\modules\linear.pyr  X�	  class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
r  tr  Q)�r  }r  (hh	h
h)Rr  (h5h6h7((h8h9X   2034580779520r  X   cuda:0r  M�Ntr  QK KM��r  M�K�r  �h)Rr  tr  Rr  �h)Rr  �r  Rr  hENuhh)Rr  hh)Rr  hh)Rr  hh)Rr   hh)Rr!  hh)Rr"  hh)Rr#  h*�X   in_featuresr$  M�X   out_featuresr%  Kubuh*�ub.�]q (X   2034580777120qX   2034580778560qX   2034580778656qX   2034580779520qX   2034580780672qX   2034580781152qX   2034580781440qX   2034580781536qX   2034580782592q	e.       R      x      �Ad=��=�c�;�����<�i=X9�����<,d:�a����`�B��=��"���<l�=o���6O��[�<Ts�Jn�<�L*��A�=Ln�=�M4�ѷ<�ؼ��1���=�.�����<xV$=9�O�;=��J=�Nz=�k�=T8=�86=�tT�j��=�,i=�vU=X7滸̄=dV"=ۥ3�bqϻ��-=�H=Q�=�V��'<kJ=�Ի�=�*~��=8�$=��e=d�[=O� <���ta���w��3c=��k=%�D��QD=��=M&���ݨ��l�<���=[h=z�<Tł���=��6=k⫼\�j=��=��<#�<�=��+�V����8x�+/o<c�=�X=Hߖ=��=�Т�E�м������g�H=L���@�=z0�=S�μ	��=��<VT꼸!�=]l�=�%�=T����Ѽ�WǼ���=��=�+�=뒨=�ͼ ��=��=%��=��=� �V|:>��9�笹C��L�:=�:��ʺ��:�s�:#�p��'��Q����ު:*H�:��H� [�2C�:�.�:N͂����:Dw�:2)�ަt9˾�:�w��	��ޑ�[;��9,�;^��:��x��[;�u+�:0n�:���@�� z�9U�5���۹^�9�V�:���蒺������8}�j:	���-G��-�:ɡ��أ�GsL��X��O���ԋ:��n:���8�='��,	:XH��rl:u+:�X�Ϗ����r9�ut9��ع)Fz9��9�$��͑���aE:��)9�t9u�M:H�Թ�Q{9`3�9�藹8~�9��9g�<:_E:��9�r�g�鹊҇�c�V:��:o�c9s��::g޸6�,�� x������c��)==�qx��&�q�L�8�i:Uq,�Oĉ����:�*|9m¹Q�$�tl3�����P̹_��:�W�:��q�.���9f�&��8�9��N�"[���9wC�)�8�̹�:�: ��:�}8���Z:��:Q�͹�傺G4U:�S�:�
��_]�$�:��L:�ɹ�:.ē:Ihk����ч�:u�-�e[h�D�������t:��n:�v��5R5�ɬm8C���Q�|:���:�6���S:~��:Ý�����$s� gN:+_�:*�����X�b�:�F::���7+�:c��:\�`��֊�#֋:��*�c�W�����}~:.f:��׻��� ���z�)�;���;R����X�;)*L<y�M;"-��`V(<��;�,�;9�w��F���<<n�;s�l�N��;)<�N�\a*�ۅ%<n¥��ּ������<�@�QX�;�^��%�A��N�8ƪ��ta<:@,I:xܹlD:�A:6�o��F�� z�.W:@A:o�=�n��T5:��:&n��M�@:��5:n�1���칄�,:�1�&��KB�=�����9�>,:�f�<Xf�<�D������P0=�/=�7 ��2=+ro=��8�g��z�`��<��0=�r���l6�PN,=�=��޼�-=��\=<6L��_=>�4=f�ڼV'�iە�5����؟=��=-"+�|�P9�ȳ:eH�:V�������:E���L��uC����:ܯ��������S���>i";Y�"��^�t��:�c��� ���F�.���HX�:�%;h�*��	�)!@��� `�:B(:e����j���S�� ��p�#�ѹ1A;��r�V<��8V:�K��h��B�:���9�&��A¹����xU��i-���:�:{m�<L�Q��9"&�.*�9�N������Q:��9�>J�؅��&:�!:�6M:�::�ѹ����?:>K�9x�#:�cZ:O��	�:$�:���!W:��	:��:��\:�� ::D��G��-aŹS�:�:�9*�:�E <\};<��� �H&�<OW�<=�����y<���<�m��p�#��y�B�<s��<	9a�M����<��h<]�"��<|�<q�V���E�<�q�>��������h�<�bt<� �^z:�X�v8�Nǹv�:)ϋ:0'8���V:`��:�)�˗ǹ�y�kQ:d|�:�߄��[���: �I:��ù�A�:S�:�Ge������I�:Sz$��)f�$���l!�l}:�i:�8P���m�X��8�\=��$:�K1:�a����:�I:���a8���
��Y:��(:P$k�i��}�?:9�:�k1���,:��7:��6�=˚���5:�+�����2���#���O7�9�a:�~�<�DI<Oһ�뻪i�;噧;�����;R]�;v,}�����ƚ��a��3�;U��<��N�LO�;E��;0!���R;��;�x�;�$2=�*:M3���>�0S����l<�U�<��;��8<i�><�'μ��Ƽ	�=��)=hw�� =,�O=��j��S˼�ҳ�6=�:=g��?0/�yx(=Cu=��¼�7
=a�S=v,ܼ�.<��.=�@���S��@�X���j=$=I���|�ظ�_D8qJ�>r�9tj�9Ր)���9a�9!��7��K���W�1�9Q��9	��Gф�f�9��9? L�֪�9�q�9:˹�\59���9}+�J��z7�}	�Xǔ�2��9`��:��:��ٸ��M��@q��W��
`9�Y�4�z��ڤ6�)�q�Ҵ�:�l�"!���`�:$5�9e���@�uT��̐�KϹ
x�:�k�:1p��{�B��P�9�I=�g�:h��52S��)��I�3�`h8;l����y:�:ȶ4��kQ:�'�:Š�:����p�D�K:�̀:�	���QV��Z�:��D:����:��:�q^�v_���:�:��x9a�#t��J �k|:�tc:��s<�S�<�3��%¼�7�=#�=���o��<��<��������aG��x�</E=ޱ��
��	=6��<dl¼��=��*=˚*���=��<ѹ������=�ߪ&��$%=���<��������:x#V:���Q���e�:�I��X#��[ ;=�^:T����ĸ�r��{��߅�:n�	��r$��\:q\��%��f��Z�T.��f0:x9�:��1;Jl��l8�U� �
�z�9:"�u�~����;`�&;E�ں���:�X;�������w̻+�;��;,���l �0;;���:F����";��T;U�"�d�����[;��J��"��L».M�=��;�e ;��Z<��<U����/��!��<�i=��ٻZq�<}�*=�G�H���ĝ�]d�<�o�<0�q�xi�	=���<v硼�t	=�$=/�o��Ͱ<��=xT�����������=Eo�<-�m:4
�9:��%�0ߧ8%w�8��o�L��8�9h8��-��Z-����:��-8���8͗:�RK�d�D8u �83Q,�#$�8p�8��o:��:DbS8����d��X9�c�W:$:��8�<��<<����j�	��j�;fo�;Кa9$��;�P};����e����������b�;�=R�^�z� ;�^�;����4;�ߩ;�@9��P=\t�����Q><���@�<�T�<���;�X��"j�η?9���:o�(�8�;�p#;�#�tLi��"����:�~�;+:&��/���+�V;�pc���b��:up.�0�`�}}��*�o�<�`�-4-:��];�c��#=[�jdȻr�!���:C��9��7FǷ���8���8ߢ?���8���8��˹�ɷ�bF:�A�8o�8��:��n�r��8ؗ8���{��8R�8���:���:v��8�(�6S��+��:��,:3��8ꋍ:ky:u	����:(��o����٭:W�к�w�V-9�e�9�XӺ¿麣ﺍ�Ϻ�=�:���n��8s�9kJ��P���g"���	��ګ8���:��P���ep�F�޺�S��TL��0�8�&ɹ�y:�F�:�-+���O:p��:�5�%�ɹ�	����J:���:�����N����:L}B:Tƹ}݁:���:E�c��Lw�`ă:r7�S�X��c۹���V�N:5�c:^�=�ϣ<�� ������;�y�;��+�43�;Fے;^���>h ��e;.`�;"N�;�[=��I�~j�;g}�;l��"�;�4�;X��<��];���;d�	��BO��௼Ģ:'r�<"��;�O�<�,~;a�n��A�!��:��:�d��,�:�PR:E�y�m�V���Q;���%��:<�׻m��E�:��O����:�v�:�ؚ<�ea<l�:���g�ϻ,������<��}<��:
k�:�u�:&�ܹw��� :|��9(�_��:��
:N�úlM$�X��:@S�9��9
d�:��L����9��:]�"���:�:X��:��:���9p^��L]�{wº��:$��:���9�
;^8:x.&�;�ɹ��zwA�I��?德�ܹ3�_���׹*�p:�'��Hp��,;��g��t��"�7��ι�f�`�����:)�:z-��I����F\��t�mt�9�J}9����U�u�+Ȅ���9�*�d��:�>	;ꙫ�F��:#2;Gт��l*����f�:B;�/��xѺ��
;c��:��&��;�;t���%;�1���Qܺ��Y�bW���+�:���:lڂ�¢��� 9��8��B	;��;�M���t�:[�*;�k���78�����I�:��;�r��4�:};p��:-_4�2�;_H;�t��v���;�������G��ъ���	;;��:P����;����_�;��;�ܰ���;/�;@��W ���*�<�+�;q��;J����b�E�$< ��;>n��yη;��
<-�����7��q�;1����������(4f<����+�;~Ɯ�F�%����8-P��;Z:�fh:2͹�{�(:ʋB:��7����Bv���):F`:��V�����9:��:�&��)]:{8:sV��_���*:������F08�3[�rP ���E:U��<F�J<5l��)_��Ny�<#d�<W�i�Z�=E�U=�m�1Ԯ���߽���<4� =�w"�Rқ��%=G�<�����ñ<PD/=��̺���Ne�<U����ļ��j���Ļ<T<���<~�39��5��e:`$;�O{�`���^�p;m�j�j���V�$;��;�I�:'�\������;��k;{��@�e�mw;�v��v�����;=U:;g��m)�:^�p;c�:�bW:�M�lk��m6;L�X:�1�z ƹ��������0�ש'���&���c�1�ֹ��y:!l��c��P�';H!�8e���׸m�̹����O��qI�:_�;"���l.����8�т�]��9�~]9�eb�X�<�J<wZ��%�F��Hv�	�!��^ȼ{ڼ���ڼ��{���=fO����b� g�=�:;��U���L�ܺ��-���:ʻT=D�<)�Ի���:vܻ�z=��=��绚���z�<�^<G����/�fӃ<s�<��*�	�g<o�<�A������5��'p�<̗�<X�
�r�����<�L<����!	�<�"�<��7�#쵻��<�����S������ʼg�<��j<�s�_U��;�9n�3���;��;��$�:�";��� o3�i@��!�:/&	;��	�uߺ��;��:M�/��;S�;���Ck��P;����X뺀�u��_���	 ;�C�:�늻Ǒj;9��:�YY:�z��̻=#;_�8���5���;8Iv\:ːE=���-��:������;�aH�}���Ժ���<��S>;-3���u�yP�:/-;TB������u�����'Z<�|<4(-�� K��);X�:\:b:2";��;�>P���\��@üڻ.�ի�:��<Xe��F��:H�.;c�X��9���:X�<c9=U?�%K�����Z^�)�I<�8K<R�:�v<)�3<� <��V�v�<���<w#���x�<�V�<= ���V�B����v�<ǋ�<.h,�c,��ŋ�<9�<�X����<]�<��q+;A
�<� >��;��`󩼲�Լ�ҵ<���<���9mV�6�38��n��,d9�^9.R(9��8������9��o�T��E�	9i�g9Z̃8j�8����'|�8~���ɇG9jh����K�b�v:�r�����18?��9]q7�_k��4tH9$T;�-M:8k0��:�W^���<%�&벹]7>�ڹfqz����,��:zj^���θ��;yc������3�8&������j���Y�:���:њ����
���d��`�:5S`9� ���n}�մ��Kq�8��3��;�;q���<�:�_%;�����q3�É ��&�::1	;%(
�}��п;w��:a�/�sS;};����*��;;�������2~�Yc��BL;���:��T<�J<�i�Fuu�K��<���<�bһ�k�<�=	��Q)y���ۘ�<�"�<����)�����<�ϭ<5n|�V��<5�=���t�<��<� Z�!ީ������<���=�R�<�Ѻ4���H6�Ny��y׺���8H"���M	<��z���3.���>� }�Fmͺ�#=Ѐ�9�+�>����1��S��v8^���4=Z޽�d�G���u�'k/�F^���F=��<̃��y�%�T�����<��<���Y����ژ<9ʼA3=�"�<��;ّ�=}������$�Cʾ;\�仅����x(<f7��-�9�kܼ����	�.:�;M<<ل<��O��I������a<�O<�G�!R��V�<�;�<lˊ�=d�<x��<�ļ�MP��k �ل�<��<�������	��<���<�4Q�P4�<X�=��W�Ȓ�;��<FxA��㙼�Լ�1�p��<���<���:�k:�!�������9yab9\%�7�9k�)9kx���~��$;�'#9 |9��;�����,9�4�9k8칔t�97^J9Y�;St;i�&9�p���O����� ;uO:��9��g<��;d`���b����;��;�s��r��;5��;&���ݧ�v(b�|6��;dY�<����W;��;�ڨ���B;l�;p<<���<��<:�����������]<�$�<�͊;����ڔ�i�G::�;��ͻh�ջ���;P�Ż�C����;]��;�R;5��[kϻ�hf;�P�;$q�H^û�6�;�ֻ��l�;5�u:������<;�x�;�ޞ;��;��j��kû��<�	<��i�C�&�^]�� �ϻK��M�?�%�
�s�ٷ������=ܚջ�����'=UѺ�唻���2��<Ļ�����^=P�f<����L��+Dj�����G�?=���<��I��Z�;��;�����T9b-�:��l�:h����6����"9B@;di��l��1d�;�ɻ:tb0�&�ں�9���('�myM;��m;�D'�����k�:����w�:���:���
"M���t����8K���:�7�:ӆ���:D��:q�6��������u�:�\�:�1ۺĻ���p�:���:��	�� �:�%�:������+�:�`����ח��Ђ�lT�:+G�:��<0�#<D"�;�q.<$�ۼ���w؈;�丼!�'��췻��<yl�<{�i�ܼ���=��:<�"�Cϭ�s<���́�J�g=˔�=�7��%J<=�6<Mܤ��DY=	��:y=Ӽ�^=��=u�ۼj�缘�P<'�;��d$�<����JO��r������䕆;��";l�C=���o�TC�<,��ٶ|;���;@=��=�B�w1���[��rI�B|L=0$3<��a<!����*��l	;��;��=�;��<?|���U�;f/<��໷�G��
��n�;\��;>6��@��b�<���;�2���;π#<A����g���&<|"�:�G����}���H<��;�T���؆�5y�9C��V$;�i6;����2�;]�s;�qͺH�Ӹ ��;**;t� j�"�_;�o	;�
���0;%�^;�d����G�[;���8���8����œ���j;x=;ޏ{������:�8�:1��V;Y�;~�����:�	&;�x��J�0�^:�"��:s�	;�L
���ߺ0;��:N-�P�;v�;�l���{;�����P캳�t��0��PN;�>�:6_�FSp����8���;�:� ;Zg��N�:�:;�c���0�"��|�:9=�:�����ƺ)�;zB�:�Z�7<�:	C;��Ϻ��/z;��l���Ѻ�]�_dj��6�:3��:�K}9?B9F�!�m=Ǻ"9�:���:-��̈́�:���:c�q�յк�̈��$8:<��:.V4�g�ܺ�'�:���:�ɺa�:�=�:�\\��%���:4����-溩Z����Ϻ}Lm;gm:m�������i�$9�f@���;�4 ;v˿���:�},;��_�?���!����:`5;�$�ۃ�̸;0�:��;��;� ;��*�x�;l����)_��+���M�:L;ky:~��9Y�)96n�r{�:���:�gO�ｪ:�s�:p��l��_�&��N�:O�:�'%�H'��G�:-��:1����:���:����
��9iq�:}�T�&������ѿ���9��::��<'�<��+@�A6Ϻ�XJ�u�;<{j�9�»6Y=�}�f��<�������-=m�Ļ��˻���:8NV��J���Pp��@=��.=���R��������<:X�<�o�;�5V��������Y�9�y���]; np;�g4���F;~�;�%酺&�C��
D;p3c;a���.�C�+S�;�=;j2����j;b%�;d���8ܲ�E��;����,J����K\Ӻ>&�;��M;�_r�Z�o��<u�=�ӽqj�c���;���E��A8_����=1?�=2Z������O��,�@=#�����Oq�=��h<�U�����<�YU���%�.�<��=z]�$�j=ýZ���SP{��Ո8�479k�����:7��:�g��ٺ:R��:qzr�K5�l�5�,h�:<�::&��������:�t�:}���h��:���:Q����~~����:R,��X��K{-��ƺ�9 j�:�]i�ϯz��}�8��%����:Z5;�ު�/��:��;X���V%���캕�:#6�: �*3к��;��:L"���;*�;.hںջ��	;�9y�e�ۺ�f�:pv�ȯ�:�:�:�S<�p�*�}Z>=Cv=����ew�Kx#=�������v�"=�ҁ=�$H>��?��Y������=���B8��=��=.���U����=O(W���S��	8=#RW=h�K=0�[;�v���p����U�/��M�7>��8^}u��n[��9;C�Q���3��D=�8�p)�1j���v�y��7.z���R/�o��85�R�#�<�S4���.��8�F��8�p7"�Ź�<ڹK�:TUo��99Ey79��498\칗��:#{�:U�@��e�:i2�:[?�mU����/��>�:��:�|�1��A"�:���:��湂�:��:c��_B99��:��(�����㏸�%ƺ���8��:jRw����549�5+���;Z�;Ac����:��;��6���)��c*����:��;����uѺ�;Ϧ�:f:'���;[�;9���hպ1�;����J�ۺE�W���o�:�S�:��������!�9�S�&}1;�CE;X	�	r;�(�;�ߺ�W�cE=���;��7;�|��W�"�j	r;�C;���-?;Nq;��{��-��8%n;q��8�*��ɺ+�� Ȁ;{�!;�Y�(�k�C(�8�-��&�:���: Ꞻ�Y�:�;�>��Q��3�ٺu_�:���:���I(��7�;'�:�����:�0;%�ɺ������:�f�w�˺h�W��d����:�l�:s�9y�9@9�&��o�:<�:�;�♝:�=�:�MG�������l�:[��:XK*�����8�:�:�~�V��:��:e^z���9�w�:(?��ޓ��T�
L����8h��:o�9�E���UJ=��=�Ɯ��A��߼�
���g㗽S�=�֙=2oN>O"��{���z=Md�=fN�F7��@��=�,`�����7�<�=@��$��{m=��=	;Q=���r%���0��� �<��*:��v=X_A=Ee���vT�i4=���1�az=RF=�3�=����;E���>yhP=E����lp���F=�[Լ�jY��U}=���<�B��	#=�H)=��+='E�<:��|_����8tE99��I95�ܽ�:��;~^�о�:ۙ�:_p����fF��m�:���:�ܖ��Ҫ�`C�:���:���
�:��:�ɶ� ��8���:TsA��˳�Kx��޺�-x9g~�:ݶ����̓5:?����7-;��P;��Һ|3;*)�;m��Ĺ�1��?;�.:;Z��2H��v�;��	;�ْ�5]C;���;��^��ߚ�;8�,:7|'�u�D��
�)��;��;o�Z��k�¢H8M��8���7+��8!�V��8g��9,CS8�8"�3�e�R8u�48qL�������k9e*8uh�8��8��~98`�?�͹M~9!!�8�'�z��8�
﹚�99Z2w7/�g<�;g#��)r���q;�v;/KF�O��;�\�;A%�;�*n���-�9;SEv;��<<}������;�l�;�Bv�͂d;@L�;�7<l˦<Bs�;��&���ǻ �	;���;�U�;�u;�ߟ��W��<[�=T�?�VS���;f/x��!�,�=-&=���=N�i���B�a��<|�L=H�ڼ�\�2y=G*�|3�n��<XL�̄��=�=��'��;��`����Q�;}�:n����:��a�s�r�
�U;=�P��K��0�I;��:��;�K���f����;'~\;a���� J��m�:R�n�������;1��;�.��4�L9��d;��4;��w:ޮ����S��������19�:���;��;���%��:"�+;Ƽ��c:����O�:0j;Jy��P麅�;��:3z6��~;MO;���?��;&ꐹ�����xz��ݑ�-�;Be�:����]�* ;���:���䌺�(�:�.���u���[;�@�:(���V�Pu���������:�C��"������:z{���M���W 9hߺ�.��<��:���:Qx�:wg㺏:��|�����T�;�'�*�:�w:Tc���͹|K�9g!�L����2��uv{:��3����*��� �&�� M:_D��0�s�~:�ܹ&�Ĺ%�캉/ɻ1�ɂ:��J:������ 5:b�	���;�u�;>�/�J:O����;+�j;��$9	us;�O�;��6�sh]���4���4���e;��.�h��+��;�~r;O#c��/;���;(ɔ�`T�<�(;z\��4�<AH�,�4;kW?<OGY;1�S<M�<L�<}��J<K<�3��f<3<0�ü�����Is;iI<�6�<�Z|�d$<�Z<C%����;
7<N;�'=�1�;J/�~h��dɼ�8<:c�<�
<���;GnD<����v ��m=�3=*)$�n'=	c[=O�s����9���%=��=��q���'*<=�H =���B�8=ȋY=bDc�@v;�l9=KӞ����&�J�>��3�p=p)=	S�Q� <����p��q=V�+=�K���<>f`=�"���&�7�x`(=�F=��ս�V���I=���<�ʑ���.=|�T=(���]����C=ue�� 	��J�	��V��.m.=��=1;�r�:� 96Ո:N���@Ѻ�;y4��4C�C#�:+�:���:R���kźے�:kxz:�S �䪺�d�:~Ժ_��*X�:�;��Ǐ5:D�l:7�:*�C:�P���D��K��;�$<�/������_�<�8=uY��x�< t;=sE �[	��\��.��<F�<��y�B��M�=��<ù��D�<��&= ��X/=��=�ː� F�(�F�0�[�"�i=7��<�c:`9�9+h8�
�8j��9z޵9���8d�9* 9A�:O��8!���|j9�y�9h�9F/��J��8w�8D>�8���9�19��80��:iyn8��7�����:��� O����9�M�t@7��η��
9�����J��AɊ9m\x��_
���X:DJ9�%:�d��S��8�Z:,��9=�ܹ+}y�S?	9QU��i��?H:>e:����?7�0�9�a:��:�7����j���_<�!)<O���Q#�{�:<l�<:�`�"�0<�N<r׼(�)����V�;�l0<�.�<����Ĉ5<`�/<�l*�m�<'�Q<���;(�0=���;��;���p�޼2�<8M�<�'<�Im�����/�9��q:Aɿ�T������:�r��L'�N��:>�z:�F��lj�MȮ�i��9��s:V~��|��w:�����x(�:&:�g9M:�b?:E�x:/��:ɞ��!���ﺹ�i���eK;��*�������;(<��U�N��;,~t<�s�����^��<u9�;�6���'J<� �;���(<Uz_<���ln�h[<����Hb��%	b������s�<$�;���9�"$<�z��#��b
=g$=��ݻ��<t>=E��{����"�"	"=�=L���[@=���<�P����!=�R=⟽%G3�e�;=�p��۬�����M�Eg5={Q�<�+���f:�I:��C9�S�9¡�8�W8I�n9���:v�:p� �v�9�y9L� �^�:c��9:5�v:X�9p�9\���713�@��9kh	:�4:�V;���i���/99�y���!�48��7J��R�i���Q8�	�VL��j:~�7�.�9�C�r���Z:��p9����6�T�7��z�9�@�x��9�:��DC�b�ַ$Y�9Z�:���9x�6�5`���:���9�{/8�>9��9��.9�ӂ9���������0\:�yD9�\�����8�+/9� �9,Z�8>g��H���j=9m9뎘��s8�^�:�q�]��8��8��_:@�!��b���W�8zI�;��F<S��w[���*=�=c�ֻ�<*;,=|�Ӽ�����FѽhE�<�@=-�Ƽ,�4�=t��<�P����=�h(=\]*�{=�0=�.��#g��vr#����b�'=oF�<�VX<5c/<�i�H1�K�-<6 <r�W�R�%<%�D<�ͼ)#��޼l�;y�!<��<�]��n"<z�%<؊$��B	<5�T<_E�;3�E=�1�;�������ݼ�E<�2�<&i<`�� <-{4���5��4�<�f�<d|�g��<�)=�����>�</!�y��<u�<M�ͽ�ؼ��
=(�<w�?��I�<?$=v&��N�:���=�b#�R���V����Q�б=٘�<�5:�έ9��8���8c�E94�K9 �9Ҟ)8�u�و(:��8b�m����8PR9aK:���7����޵�*�8)/9R�#��gP9�P�:�zøg=�7R	�7��-:���j���69W%������9��e:���P��p�d:9`'�o�Q�*c;��n:{����sʹI��;�v�Ъ�:��2��E6��Sl:�3	��=O���f��2��1�3���,:	��:�e�:,H����3���^��������;y�;�饻;H��g��;~᫻�y���Z<C��;^�"����������lN�|�;����e	��A�;D����F���ѻ ��;�_�����;��;c�<
A�W6�_���%��:<�F:��W9�/n:�ڹ1��/��:��+�<��e8:�p:ދ���s� �¹):��#:�8ٹ	%<�t�h:��ҹy}ȹX`�9�>����qU�9�� :��v:�Q��݂�����	<�f�;�uʺ�y��o*;2�;�cs:�H;p��;����x�5��yV�8�;�w��
����Ta;��;��'V ;���;�s���;r^�: ���3����j)�$2;�x$<-��:�i�el|�v˹8���7d���K��q�:8�m���P��B����7Na9�l��k��H�9���9qm��������7�i�$��� �8=��9���É9�v(�)�P8+ٶ7�Ըb�����8� ��Á9�ǹt
�����9%�����@�H��:�p�9a3�:�Nܹ0m޹c�:�G�9����M����|9�:�K�1�Fu�:�a�:�PI����8�A:�S�:rɝ:T�������P:��9#�
9��5:5!��u�%�C:�'�MfN���9�C8:�5:����o�4�:�5<:�Z8�J�-�X47:���VG���:���:��1�U:�n6:%�8WF�9�F:9���%�?9^��"	�8���y(�ӗW��Å7���簹x�mlL���9�6�Zw<�]e7:
a9a����\�8����R�ΰ�����9��������g��Z9�� ���M9C��9Ia��:@</�z��|��i�<��
=�O�7|�<��3=P^��)��VB�8�=2~�<֟$�-y���=�Ӿ<�����	=��=Ip��c��g=�}e���׼����F�@�!=U�<��;�8*:�E:�1��;@L	;�f�;��:o��=�>���3�6�P:���<�^	;
�:��<;�=ִ�9�F��;�'=R�+���;�I@=1��9oʉ<���a�C:�0��ߠ�:	)�<Z�{<�:�g���y;2��;��i;Of�:V��=(e-����`������<r^�;V��lXX<3}�<ℹ�����;��%=h�C<�\n�I۰<�:���<#�x��;��s��A;�k;H#�;�JQ<ԉ;�;kl�;�T;���;�'7;k�"<��q;�Z1���8;̞�;�Oh����9�-;Ӷl;��;	};S+;<��H���D;(��;}�8_N�;�昼 樻V�;��y�"T:<E<��<t�e=�UW=i+����U=�A:s�����<#W�����<�� =�j���  � �9!0>=���<�A=��m9��ʺ�V
�;��:9�:<�@8���y9�E	:Σ0=��;��):�E:�1��;@L	;��;��:q�<=���3�6�P:���<�^	;
�:�W�<7J=ִ�9�F��;g==R�+��m�;	��<1��9�;�<��k�C:�0��ߠ�:u��<2�{<�:�g���y;2��;3���Of�:da�<!e-�����J=L3�<r^�;����ȿ5<�?4<ℹ�����;K3�<�R�<����eX =�:*�<j�x��û��s��A;�;�9��JQ<ԉ;�;ll�;�T;���;�'7;�"<��q;�B���8;̞�;�Q_����9�-;Ӷl;��;	};S+;�.e���D;(��;}�8�M�;/�2� 樻V�;��y�"T:�~N<>��<�af=�G=F~���Ih=�A:s��`��<#W���,�<�<=�j���  ���9qH2=��<wK=��m9��ʺ�����:�O9<�@8���y9�E	:�r1=
       �X�?��F?D��?�-L?�D�?;�?���?��G?#��?h��?�      �t=�6�<��мø�<��=�%c<�X�< R���.<�����]��FҼX�j�T��Yg�[��mû~1)�WƩ����5�!<��<����./���}�;A����
����=��)=��m����� ���e>�e�<l��<��<�/�;�:=�'�<��<�r<�v����=p�<�<߽�X"=�A?=}�<aۿ<�1����;0�	�������Ĭ���<��<%K=�=���; b�:��W�����v�\r=[�k����=��=�谽�G=�ԑ=�9>t��<C���)�d;�7+���=��/<;���n�=��=%�,>�;*��7��=��<��=�S�<jjT��*�;U�ü��>E�L=�_ս�(������ ;;��j�����Ҽ^�\J=K)�<��A�@h�:�Y�;+�3=_�ļRw���笼���*�<@���N���m�.�پDo�=�	�=�}I=뷄=$=L�=��=�ۺ�zw<C�=P���^��4T���<b���|��<'\<�D><4Y:�w5<��1<җ{<��#<z��e��<Ȝ��<1 ��2
�\;��T�T��@3�d3W��� ����<�U�j�Z<6<�'���/<s�;!�}��0\�ٮ�<$l�;i@��@��<,<�R
<�p�<�腼/ݟ��/+<'��<�n<tɼL�^�����yE9�2��;c=��w;%�<ٟ[�Rq!<�ꔻ��Z�����m<H�<<e5<�Լlm��!߻��(<��<�6�K�.<���;�Pż�v�;TbM�!��;C���� ����ń<\K=:��<T���<,��;|[����;ChA<J�~<��<��K<�<YQ�<��=LFi���=q �<E�<x� =W��=7�=x��;�!<`	A<�L�:p�;��;=��X3ܼ��ݾ�P�<��<�B�Κ�=�0�=ay<��ڽ�36<�?꼳��<ߑ�=��=�2ϻ�]����=4F=�
w<�=�d�;�:���7üě9)s�<�+U=s��=?&=����Za<�Ƭ�`<�a��O;�܇;�����YC�O����͇��B�<Lu:sC��5�;����㒛����6�{l��x��fP��ƍ�kә<s�]���u=�8�x<0��a=}�ּM���Ӧ;�Ѽ հ<�p�=�U�S�<�c=C7�<6�ս7����[/���Ӿ�$���8<�+����ֽ�d�S�̺���;
�<�g�<���Ɇ̽�˻�dK=�Ė:<�1=�>=�=	̑=@���)�='P�=��<w�ؽ���+���<i��<z�;;1�8=�L�<l����6��5m�h=��\�CG�<�._=�0�=�U�=���=H
	>;�=ԟ=U{=�+>ag�>�� ?��;	uu��m�<��<��;���<�e�:y�ݹ?r'<�Ca;�%�:B��<��)�Qȓ<KVB<�m =_U�;�U�<������<%r�<,g<�>�<Gl�<
Jz<W������ߒ<�l;��D<�K�:ȴ���;.3����=�~�<=����<j=Y��;�K=�^<��<ӕ�<e!�<���<b�)<�@;���<+><AH�<+d;���;E8e<mҽ<��;�!=�*h<��	S�:���p�<ɦ�<z=�s=ۮ�<��,<�pt;��{;O޿<���;�=�z�<w]�����<R���_=`
�<395=�0`�Wv�<�ݶ;��F<��R=��=�1=���>巻�aW;�8<>I�<���;�)�<(h�<�C�;�ä<Y�T<x3><c�=���:��A<Ώ�<��C����=r��c(�<5K�<��;5ԍ<D~(=w�=�G/=�՞����<�iW<g��<���=-B�=�C\=�b=�g�=t��=�=$�;�����m��)��:1��A�J"N���!�����*�M�S�3�ʻl�����M���a=��<�An�fvݼ���;��=���h��:5����<[�M=��X�6�u��D���`�:6�̷�����=7�ԀؼiG����<8�,=�L��Nн������:<�ֽ8�?=�f��
닼��=L���^�=�a��p�<�y�=���=���=uL�<�'���D�<k�<Œ�;?&p6�=��<E�o<�Wj�Q��<����.�����%�E�{�u�=�k���3ҽ��0��-���<�b�=3�ؽ ^����<66���o�����	����P�$�*�m� ��8�}��
lμ��.��n?�N�=S�=Jh|=�B�=���=�wu=��1��s1���T�ws����V���˃׽������
��[I�*a5�>;+�s�4�$�$��*���4=;,�(�0߽�X<9+ۼA-��?�<Y��=���:��ࣛ���< �=���c���(���Ȼ\��<J�K�_'Z�'�#=3Y�=�4N=&` ��s�<*aa�
���$���9���<�白� <֍o<f���ـ����� �4����I�;A���Vy���u<~3-=������c��߮<y`f=} ="1��8Ġ�����
E#=R@=��=�A�=��:=�`���ئ�z�׺A逽�YC�1Rv<�у�2�׽w���O�ټ�o|;�t�=�٤9��s��x�<_G�;�F���6����-����<�w=+�=yq==s���7=N�W=W�=���<c�?�J�=����c������[�ճ�=:�&=����s��U�(��ѽ�譽����:z��((�/��=��>ח^=�QU=ྦ=M�<m��;�Pu=4>��>%}��`���cQ<n^=�^==b� >Ж�=@<<#ɕ�~�����$�_��4	�p�н�2��kA�8���ؽϪ�o�켍��<��<0IR�c�2����=��=��=$��;�t�<Г�=c1�<�d���+���<��:����6�)���ƽ���;���M�׬.�L�J��^�<�V�� �5=c��=3눽]&潪|ٽ3�`=�_�<П(������<\�>=c�=X�V�$����z�<���;��;@�=Cv�=g��<j��<#�׼��>��я�CH=�D��~�_��ܞ�:�����"���Ƣܽ�\M�j˔�,�/�@Q>��dH��͟��Q���c���|=J^j=�{ܺ�F�Pbb<%Ɯ<WU�Mؽ2W���#�������A�N�I���%S=9s<烑=��=����;^"="M=��=����<@*?�^���R�;���g1�(��1��hC<��O;��z��~�<6���ûO�@��[��5���g��h�;��t	�\�����S`�;	Ha�Dot<�g�; ^�<5��<W�;��һ�2'�z�<���<�Y�={4<?�<�=k'J=��<`1}=��=�H%=m�\=�$D=�=x�=�<�3�<��N<甤����P_/=�01=�h�!K���U�"U�<`ڱ<`@	�O�S�.��}J�<����i�=�<��[<�>H<�ww=��=� �<��<VW�:��ü#�S=����ߨ<��@<F�;� .=ࡍ���0<d:<j '<i�v</<��;�����<[C={���<<lI�<K�=�'~=?^���#�;��;���;���;A�O�i�ۼ��9C�<�2����Y��L�*<��S��B�<��ƩO��v;޵<*��<���<�;�	����q����"��4����:���K�(���xo�=��">��Ӽ\���g��=J@P=������= �4=f�`� O��cJ˽<iʽ,��f�|�����5=�aZ���M��,W����<ӷ�����MSn=E`L<���9�=�[���H���<���=`�e=�f��;�;��;#Y.<�T�t�&���Z�/���Ak$=y=�PȽ�h0=@��=%�����a<n3�*����MӽP���Ŝ����R=��\<ǎ|��r.=���=��޼�\���c�.1=C��������� ;F�=`�=ꃁ�^.�:s��==�8���<�4�:���f�r=n;=K!�I[�;�n=��=ce�<Ӓ���H�=�E�=�.=>z���dC��c�<%���ܬ=��7�餽��ǽ�8�E�&=�X|��z��Wi,������<g=f�<֣�;��w=��2<�=Rg�"�ӽ�P�;�/y�36�m�q���1�HpϾg�Ѿ��@>���=��-��2�<.�\=�6�_�I=h===�w<�u	;�LP;�ʽ���s�h��}ܽ�2_�R�
�r=<�����J���Y��w���&����0������l=�(=EǢ=�K)=�qq�d�;��g�`E�<m$�=[�Ƚӽ��Hν����<�<`�F��y�����=�!������=�ڝ=��1��IĽ5#���I�<<ѧ��������=����=	S�`����(��>��ρ����G=�l�CQ��=��<��;շ>�8=�܅�d�=��h;2t�=zʭ;|N�:�~�<��=�o�=��]=Ý���u=�=�=�%�=K��<�Lؼ���=j?=J��<�e����<��b><@ڐ:d�=������3��u���A˼�E �Էܽݳ<y���l"<�C�=��<��<����l��<!�F�v�W��7y<�r��;D;W�%��"t���5��>>,>3��C ��<���f�G#  ����x�N ��B  �?����R���>	$  1�d����� �˶^�a�I�X��>Ev>�_���+6) ��0 �c>�>3��:Xg�
       Բ>>â��� ?��i����?�M#?1�L?����?S�?�       �&���o�<��>�Ē�N;�$|s>p�=apH>����2�=�mX�<%0J�?�>����׽��{<#;�=9P(=udl�ا�<�!��|7м$�+=XU�=#����d��/�;��(��w��2�V�`�5>�Pf�?�C�D�f=B}���j3�&";50�=���Nh7�s's;mGa>`>�(�=�>�.��d��<�"��?rf��%=
 �=̟M�����QؼT�Ľ$nν��=�E�<�M=RM>�z�=�7R�	�@>A�\���#�<c�X>�����>��|>:(�_[�T�%>��p��D�2=ȼ�=7�K�y��=M�R�0�>�������j� �=<�-�Jz>&��^˕>Ͽ�=��=�.�g`#�~�>M�>�τ>��=!�A>w@>�;��1�U�w��Y7�54�"Ž��@<\����D=Z�&�)�X=�=�<T�=���=űE��ui>� ><&���!���1�h�뭒=vT>[�;��)�8��i=�r��oi=���=(��=s�s�D�]�Q)����)uH>�<���ӽ*q���d>�'^>=�"=J�H�T��;���y(�%+,>��K��z>�5��S���@>�C�
       �>�
%�<ю���h;g��Q���=	h=N(u= ���
       ���Aa?�@>�B�ϺAx��A�ABv�BF:}AdJAV5�A
       �#�L�;�D>��<9�>�= k&=��'<���=���<