��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq c__main__
Net
qX$   D:\Desk\DTU\DeepLearn\DLProj\lstm.pyqXS  class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 30, num_layers = 2, bias=True, dropout=0.5)
        
        self.l_out = nn.Linear(in_features=30,
                            out_features=output_size,
                            bias=False)
        
    def forward(self, x):
        
        x = x.permute(2,0,1)     
        x, (h, c) = self.lstm(x)
        x = h[1].view(-1, 30)
        x = relu(x)

        x = self.l_out(x)
        
        x = torch.sigmoid(x)
        return x
qtqQ)�q}q(X   _backendqctorch.nn.backends.thnn
_get_thnn_function_backend
q)Rq	X   _parametersq
ccollections
OrderedDict
q)RqX   _buffersqh)RqX   _backward_hooksqh)RqX   _forward_hooksqh)RqX   _forward_pre_hooksqh)RqX   _state_dict_hooksqh)RqX   _load_state_dict_pre_hooksqh)RqX   _modulesqh)Rq(X   lstmq(h ctorch.nn.modules.rnn
LSTM
qXI   C:\Program Files (x86)\Anaconda\lib\site-packages\torch\nn\modules\rnn.pyqX'$  class LSTM(RNNBase):
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
qtqQ)�q }q!(hh	h
h)Rq"(X   weight_ih_l0q#ctorch._utils
_rebuild_parameter
q$ctorch._utils
_rebuild_tensor_v2
q%((X   storageq&ctorch
FloatStorage
q'X   2109526031904q(X   cuda:0q)M�,Ntq*QK KxK�q+KK�q,�h)Rq-tq.Rq/�h)Rq0�q1Rq2X   weight_hh_l0q3h$h%((h&h'X   2109526031904q4X   cuda:0q5M�,Ntq6QKxKxK�q7KK�q8�h)Rq9tq:Rq;�h)Rq<�q=Rq>X
   bias_ih_l0q?h$h%((h&h'X   2109526031904q@X   cuda:0qAM�,NtqBQM�*Kx�qCK�qD�h)RqEtqFRqG�h)RqH�qIRqJX
   bias_hh_l0qKh$h%((h&h'X   2109526031904qLX   cuda:0qMM�,NtqNQM +Kx�qOK�qP�h)RqQtqRRqS�h)RqT�qURqVX   weight_ih_l1qWh$h%((h&h'X   2109526031904qXX   cuda:0qYM�,NtqZQM�KxK�q[KK�q\�h)Rq]tq^Rq_�h)Rq`�qaRqbX   weight_hh_l1qch$h%((h&h'X   2109526031904qdX   cuda:0qeM�,NtqfQM�KxK�qgKK�qh�h)RqitqjRqk�h)Rql�qmRqnX
   bias_ih_l1qoh$h%((h&h'X   2109526031904qpX   cuda:0qqM�,NtqrQM�+Kx�qsK�qt�h)RqutqvRqw�h)Rqx�qyRqzX
   bias_hh_l1q{h$h%((h&h'X   2109526031904q|X   cuda:0q}M�,Ntq~QM,Kx�qK�q��h)Rq�tq�Rq��h)Rq��q�Rq�uhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�X   trainingq��X   modeq�X   LSTMq�X
   input_sizeq�KX   hidden_sizeq�KX
   num_layersq�KX   biasq��X   batch_firstq��X   dropoutq�G?�      X   bidirectionalq��X   _all_weightsq�]q�(]q�(h#h3h?hKe]q�(hWhchoh{eeubX   l_outq�(h ctorch.nn.modules.linear
Linear
q�XL   C:\Program Files (x86)\Anaconda\lib\site-packages\torch\nn\modules\linear.pyq�X�	  class Linear(Module):
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
q�tq�Q)�q�}q�(hh	h
h)Rq�(X   weightq�h$h%((h&h'X   2109526025856q�X   cuda:0q�KNtq�QK KK�q�KK�q��h)Rq�tq�Rq��h)Rq��q�Rq�h�Nuhh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�hh)Rq�h��X   in_featuresq�KX   out_featuresq�Kubuh��ub.�]q (X   2109526025856qX   2109526031904qe.        �@�|�@�j}G?�.M��&�nw?�50?��W�To<��I��^># ���'��I�>Wm�{�b>V+C�+v3>�1�\?�=��D?�E��C?� �Y�L?[y5���>H�?�I�?%>�>�,      Q�Ͻ_�=7r=��=�~8=��S=L��<h&н׮�<�Z-=��>9o>�=�I�<u�2=^�x;��=�:S��9=0��=5�t=A�W<�{�;��<�=�ӽy���2#=AY"�d�#=�$���{�=x�H>R�X=��=X��=W�=�4����=ޮ�=�$#����&�>B��=�P�=Z>�S�=�	�=���=+�>�?�=��=�>/��=K =׷����->���=A8�=�ö=ׯ�eaＩ��<A���3sP=2��<�J<�$+�=�V�������t_�=5+=�m?���Q=�l=hV=���*L�~����g=M�#���]�O�e=k�>��x�=�{����Y���=(�c=��b>э
>ͷ->�ɽ�C=G�M>_�L>.'_>�$N>z�>M�W>�aY>C���^�ٽ��׽s4>(|ӽ�M>}��=�	>�Uz=�������=4Q>��>�c>�~�=\�ؽ�"�='�׽ZXt�0���r>}X��{�����>˚���>�	�e󅾻K��x>�P�>p_�� ��>3X>U҆>����.��=�Rr����t�������>�Ҕ�2�>���=驐��%>�:�>�:���%��w?:=-��<�v9L�;~4���3:�ֺ�}���ͺg�:5�
�ۍ˹�ӡ9֧��(�9G�\�p�d�Jһs|37aԹ�ƺ��:Z�h:F�1:�ʺ��ݹb������9�ʌ<�5<����<#M�;Ty��6a<;~��4a<+��;��w<Ix���,�H�<U���Ad�Ҙ���J�;*\���|��т<�bX<~�+<��\�����������<<^�<n��;Y����'׻��p;e<�l8�Kb;�;<�Y�*��;�u��;d�^�[�]<�=����ǻ��;fb�2B�;�!(��H��P��rWe:���^�l���w<r�J<�i�;���������۹OU�;x��)�ۺqN;�+��L	:�;�S���;��2�����@;�;����;B�:8';R�:B��:8��:O�Y���>;��:��;a���ST��Ͳ:U;�J��X���H:>�x��1\97L&:o�%�WxN:bn%��F��}"���!:�iT:��X��V:@wk:�WY:��:�@�f&3�n��9�>3��S� �%:I�:`�J:Be��U�N�:dmW:�}��a���.�:'k�����9�Ν:�ٝ���:�*��6������y�:��:LL��۔�:F�:ib�:M=F:�`�(��q��9�����ޕ��H�:�d�:��:��X�k�����n:��:�Sw�����O{>�����&�>H���N�>��-3����Se>_.�>$2���]�>Bq>�σ>���+>��s��V������W>�$���/�>5��={���j)>c)�>yLO�	�@���t:�mP���9gO:��N�WjO:EDN�1�I�̣G� �F:��P:
7Q�L�O:�i:��P:=��9���m�d�_2�9�/���D�W�M:E:��O:����FP��\%:�tP:
`�;@L^���k���9ؔȻM*���<�����<�č��"
<'"
��"<fH�;�����@;�����,��&<"b}<C5��b�;�|<vg����x����<��;��s<�ڈ��p��m��aT,;	�$��Of:�yy;��z��';�{���7F[��f[;��!;B#�r_;�#6;�e ;�9X�$�x���d:9���a�f{{;��Y:z!;�<.����J�;��#;���x��jO(;	4���[:Rm;Ւn���;��n�����{P� �P;�E#;ˑ"�c`;Y�?;Uo!;�*\:� �3^���c:����DW�\o;��:�H;�*(�x3���;H�#;<�:�ǔl;����~#:7Ph:�m<0�ֻ�;�}ٻbш;.���H�;˵W�}�Ѻ�ڼ:_	����:���㑻D�;E�h�j�V������;g�;�f3;)C���'Ϻ��g8�d�:�H��F���2;d�躧� :Ϸ`;��V��F;ޓW��H�K�P��P;*�%;�6���J;��4;`�A;]%6���M���<�P':5:P��T��IW;��W;v;G;^T�֊I�vM6;� @;0C&���:� �(<ٝ � ޓ;�|'<�Z'�<_&<b�'�|�:���$���$<Gb<6v!��G8<�x0<�$'<u�׹�v5�	15�nU�;��(��],��(<*/8<,&< d7���3��E<I�%<ȭ>UF��s�=�l��=TZ��t�A��=���q�=�AG��Z�=�˽ �lL>��!�P�A����A�^=O�(������?��+>e|�=�����E��CS�}ռ��>y�=�H��goӻU�仾)�;��Ż`$;�5�;I`׻7,�;��׻���XmԻ6��;*��;�1λɄ�;�E�;+�;�Iȹ�C�}�ܻj�H;��y�ۻbF�;�Q�;���;j�滫��?2�;S��;�@=j&��)�Z=���<���z�%��=JL8���=�.��e|=_��N�ûY�3=W40����;3����<A���jR=2*=eT=�6x���G<���<��==��f=��2��ٝ8qι%��9"��6{�7 �:�,��崸:`�)`͹K��S�:KM��ڣ�7I̬���d:X@��f �z5����:�ߚù�&Ṩ�:��s9���BR���f78��9��_��|���P!;�����5IO�9m��;����W�X9e����;9X��LZ;�QS:4���"9z�:ތ9�v�7��+�2�o:ޘ�9zJ�%�u��m�;>�;\�9��7�i�>��X��^I9]���9뮺q>�:N~���Ü9��:0�%��:����1�Ѻ�Z��.��:�Cp:E뙺���:]�:��:�@�:e\t�|W纮��9���p���ӛ:_o:�k�:�h�i^��OU:�j�:R4�;���9��Ӻ��;�Ӻ�_L�%�O;�Å�L;�:i!Z;�Y��8D�^�;R6����
��=���P�9���:b�'��\�;��";T�L�r溉����/:���;�J�:�R��?��;	��
�;Q�;r�7��,��H:e�໦HI:&��:,�:5��U<c x;T�m��x�!r��=���ĺ��_:�Z9����|����P�3N�;O)件�ź�rr;�Uh;"~}�渺q⣺��:�ռ�U�9���:n
��p��:G��ᨺ���+�:�Q�:�μ��Ի:=Z�:샼:�_\:Cł�ɻ����:����,C���G�:ϳ�:y/�:��{��ѻ����:B��::'~��ِ�OR�<�t�Gb3;#x_<c}X��&�<!�Y�|���ۅF��/F<P�^<x z���{<M׎<��t<�[���6�^eż��b;)/k���Y�2Z<��;�ˁ<�A�J�z���M<�y<�|u��ւ� �|>�C��}s׼�}>������>�.��@��By�+c>\�>���Dh�>�w>z��>�h���>�Ny��i���R~�� ���>)����!�>W�=!����!>��>���:UZ;<�<<�d_;e��P���?<�1��t=<�u}���<J�gy?�#Z';1\��al<fx�y�<'�=T;��λ\ G:�1W<�:�v�<p,ۺ�s�;>�;a����+�����9��� <L<ֻ���;���;`>��&�;����0t"�R���7��;�v�;�"�'<�<���;Τ��� (�ܔ�Q1�;

�"��&��;�l<���;�&��!�h'5<���;ֽ�;����8=LY��7�V!���E�<�)һ���<D�"�i)�<�u���k�<�U�;�c����	��޺��S;͊<��F�^$���z�;��<�����$��ۻ���<�,�;�F<����7R��>�I�=P:��2�k͹�4�:������~:����5�v�~��"|:�cC:bEU��: �o:��G:S3��k� �+��)˹1�G��y�����:DS:} �:�h6��4=�[\W:�9:���o�����=����QC<Y��=��۽A�>��׽65���޽/��=	�==<!��T >�W>��>�_��9=-!ӽ���;v �!ս~1�=K.���>@ܯ;�H�j��=W�>�m�;�3弇V�<[.���<��<'P�y��е%�:�C<�(,�*�<FK ���9<��K��=���%�/�~:Z@%�Ó���q�;G�-;�@�#,<q�=��*���,<;�¼2�(�Խ�����=�߾��?=O��=��ؽ���=�/ٽc���|)׽���=ˡ=��Ƚ�B�=�=!z�=F5�/HȻb�2���=�ݖ�S������=�u=%��=�߼SSϽ�Y=���=��<��}��9}=�D��kܒ�#����c�<�_�����<�_�.�<�ꦼ/��d�9<��>��⻂1�9Q;6+=BF�'~?�a���D]e<�ᦼJ�v��썼�1=WÆ<���<�#(������[u�M�;ꨪ����:�ܚ;O\��N
�;՚�ɦ��Q����;w��;O��K#�;=É;Ĩ;��_;O��݇�/��:s]��I���;"Ç;0%�;��V�����T�H;ҭ�;b�>�+�0;��Ϲ�~�j��:�޴;f�*���3;t8
�s�;$���B�;ٰC:GJ(���.;���:��%;D⮺�G���˿:�R:@�!��	��<!��;_[?;�Xû� Z�� 9KF(;�2�a	��1;5O�1�0:)l?;pa<���2;��B�����2�C2;��+;�7���/;�;;C�5;i �:���������E:U��%X0���A;,�;��2;�����6�7�:�5;�w�	ͽc��=�ͽϜ�;s=�=���	>����Ƚ,]���=M��=��"�|d!>9�>j>O����R=Z�ڽ��Q:C��ٽJ%�=I�:�^H>yN"<B# ���=��>A���hܽ�[�:�t��S�9�*;����:����������XC;���:_�����:���:o��:�v�:󲸺�aպ��9'�ٺm��A;J�:���:������]��:gV�:��"���\��{��*�Ł<�	=)y��z<�g�O�p<)����=A�<�Q�2��; ���"<����[��m�V=�<�%x� ��U=:=��<�~��5����<W�<i���)'(�G�<�A��<@�E=�{K��a�<��H���-���5�mv8=�<'Ҽ���<��==W��<n�����+�g�<�뮼i�?�5�G=�n�ċ�<|���{�ͼb��<h��<��a��}� D�<h)9;t�<�	E=!tM��g< K��%#��6��":=��;�ŧ�%)�<gZ=)ƚ<�����"»���uI�<�B����A��K=���#t�<P|��x�w�<�<���<��<����Ey<�n<Ƽ�<�l���ɼ5��tơ<�k׻�D�;�����S�<�����X_<��ü]�\���@���;	(�<`��<���D2<��@<�Fʼm�F�t��<g`=,5��iL޺����':�	�:� ;�tZ;��5���:�C7����J'�*;���9.�P�ƭ;D_���:�]��7g��sú�4
;�����1�}y4;��";�s�:�_�$A�:�;���:7��:f�j�TN$��Ӄ;�X�;���!��9ʹߺ$!79��g����8�NS��"F���2;=�:C�#�I��C&��$Ż]���x#&<C�w�n2_�*��h�;�ܺ�M�����k��;8 �g����ɺ����=�i!�5�=˨=����^�=�J
q;�#��Sg�=D5h=b������=��=}��=�+p�?n���¼f���c�\�'���W�=l�c=�S�=ׯ�����F�<▴=`�A�շ����;�ϣ9�#�;䭡;������8;�����G��%��l)�;��u:���m;�;���:�0;��5�һ�-c�$��;���"����I�;M8�;P�<;xɻ\��в;�+#;;��:v����s=��7�6	x<��<�.�a�$����"���л�K�;@& =�5�f]�;�&2=-�;a�t�#c�}D��=8S=�pĻir-�5y<<4	�iqպU���gdH;��=�w;�75����p 0:N(b���49�,:��-��U9:dP-���	�;"���!:��G:>�@�G�>:�Ti:
�A:���9V��z��?J9���̗���-:��
:��5:m���<�3m:�@?:v,��7=��^�r`Ƽ}�=8l=�Pf���$=�Wp�F"=�~m��l=�ۑ��R0=+�C��� =p��:��y��U&<H��<��=��%u���q=�\�=M&=7�@��"��k�	 =���[��y��<6��h��;�ɟ<!���
��<���'	�����ء<HX�<����R��<a��<<��<5������֮����;>�,h���K�<�`�<��<T�����3��<ӫ�<R��<P�<������ =�ػ�9���ܯ<{�#;�<%ϐ<�1�<�O��<�Լ־�<r�༥ļ"��S�$�tH]<|��<��O��q�< 3�<yW������%���i<���< 8i��]�N�;��2���;�7�:	��;�0<�P0��҉��;0���a;o� ��V!<9Н<�:�~��񪻾t���#:��1����:�F}:�`�ףe��0<(=<���� �0�][[:���;�F����N������P;�]�"5D:g�I;��I�,P;�MI���$��?�\�>;ْR;0�T��N;��O;�US;��:����!;��-X:��8��1<��I;>�/;VO;���5�O��g;�*S;�?�<��<n�V<z�><�G��{ ��-<���f��;�)<�:<AEA���A�g��<����Y�<�P�����;�M�<�W�˻ֻ#;�<,_;~�g�\>�)ļ� <K}�<�:bz˼���Phʽ�G�=rǽ�L<I�=E�����>]��9ǽ0��T��=���=d� ���>==">�>�\����I==���;�;��:��Ns�=��>�XQ>&��;�!��٫=1[ >/�E�U^�P�;?��7V:���;̣��$<;qԣ�7�κ���Í�;��;f�6��!;x�:˛/;�:>��8麼��8E��I1����;��D;�*F;]D��-���:��0;>�y;%�7J��J�;�n�;n1�X,);Fu�(�;54)���:��������;���:/���K]��������9�߭"< �:�%�:���dn�:1?~��:��j��:R��;�zq�{��\FF�wЂ=�@�|�;U�:<нQ�!��;��C�$Rֻ[�3�;�2<��<%Lӻ��<U��H��;s=<֨;������;V���vs�s4<�X��\�;>H'�����!K4<�
�;۵ں�d'��&�:n��Z��8��;�e��L�:l���e �۟���;�:Yg�:*����;/��:�,�:�b�e /�P��j 0:��S�B��ќ;��;O�:E�0� &��\ZF;��:��9ދ�9���.X�95(3� ���9�k����9&y�9�I�9��ι�����9Î���ṃ#�����9���9v��9�uL��3�9F\�9���*&A9�.����9֨�9�:ϹɊ����N<�==��K?=���:��;b���g�k��wY��:>�t�XrP:aA¼�Ե<�޻��_�����XK���0>`� �`�@<��9;I��;��=�:g�|�D��4�<49�üYs=D��]W�=@h ���:�l�����<�Wf�&�=N߷�Ǭ5= c>�7u����!=`�
��@:>]��y��k��=�z���g���W���Q���`
�- ̼i�n���=�j�<~�<y�6���=v=슽;� �@�<W2:�7=J�ƽpK�<Z�e='��<-߼���/n�=���]��H�%N�;)���}�<��<��=��=�P��:�"=ohʽJR=�}�=��F+�qI��ͤ(:�e��|���j��:8	�;N���p�;3J��O܅�@"����;�b�;�W��3�r;g��:���;]e�:�Z��Ѩ�9Ia\:bl�;s޻�o�;J�;T6�;k��V���%:��;[�����2�Z��;�sһ����i< �����<8�ƻ����}��3�;�d�<>6ȼ��=-g�=���<z՗�A�:4iܻ��s�妗����o��;<�o:��<�K:�n0�$�/��<�����=;�8��W<~�h;}ü��i<j{�;��a� ��;��ڻ���;�0�;I�"��d�(�/];�H��q�ӻ|��;�r�ޔb�+a�;3K�;�ڋ:��;�ʻs,^<UP\��V�;�M�VPl��<C9\�8�h�8�\�9tv��?XM8-����\��sp��r9�)�8�F��=8���88.8�Tܸ�`o�7}I�P��8~rF��w��ݦ�9�����8_�j�k�f�q6:9v+b8�f�:�u:�F���4�:H�������X�:�T��u^�:7|:��:��кf���ð:7%�� ,������ D�4�:0��:੓�c��:_��:sS�aӡ���1�:g?�:�T�K���3����=��U��z�=rj5=x�<%ﾼJ��<+�߼�5<=����T�<���<lĻ fɼz�&���;�4<�D�N��=��<�'�=�j�<�/�<�U�����<u) ��!��#�"+�;#9;!��:�a��7;l������;���:�;[��:5d;�����|9;�g�>��X��S|�\	�:͊;���a2
;��;'	�o����X��:�';�����!
���Ժ@	;ȶ�=M:�p;����	;y��z޺��=?�:;����;�>;�;��:�����(���^:*,����y�;�7�:$�
;S��C�����:P�;�oa=*+q��&�<ߒo�!�ռhtl<�B�<�zY����<�3��k��<�g߼���G =�.��W�>��G��`w��|�<�?�u�1<�{ ="Ą<m2ʼ�p��,S�<��<��1=-�=>c<��':NŰ;S��"�\;m��8ҵм�*�; �l��
�0�;_��8zv08�9��;!Ն�A�Ҽ���!�W�oru����;�����:;Ǳ;��9�1溉π�dm0���9CK0�g��s�;�r9���`<�k�<t�;�l*<j˯�����k��d8�Y����˂;q���5�;{z<K��<�'7�ܴ��e�/<���j�<���:�� ��mV;��H;R*�����<5W�9���;6�2�9��=lļv�<\�7=4�O=�}� �e<�[޽Ź�<7m�o��<��޼Oa"�Y�=�������2���P:�U;����;�=Z=_�=�~p<o¯�g��xy�h��\��=)9�<T%����ϻ�ݻ��<�gU;p�;���<��-��?�;�D��,���I���Œ�;���lfŻd<MC�<3C�;�=n���ȻK&��@<����I껁]�;�4�;��;�Ż6���A;�F�;�w���<�q8=T락"dX<�k��Z��֩u=՝Լ�=P��P�=��<F�;l=�}I�G�Q=}z=q����P=����!4A����x=σ=��k=\���F���z��jR=Q$�=J=�Ƨ<��<LB���!��/#=vh~���=&]:�6O=�xX�q�9��{G=Ux	�7��<�_;��l=��b�G5<��C�	��i�`<+7�bp�<28~�i�=-0=�R1�CsI��@i���A=K�x�� �<6��<M��<�J����D<�����&>\���"K�<շ�;��`�� �j1���8�;��;kI��k[=<<p��<�^�;���<$X�<��O<.`��ى��8Ҽ�Ρ;�8
=O�j���h���<�ċ;�<���\�<̅
�2�<'���|�<�����9��$=�F(�/�"�"�K�ؼF2�;����<�+1=�$�<�$���bB������;>5&=�
����#���ڼ`�ǻ��л����;AE<K�D���<��c������&��d��<j�;ӼV�<���;���<b	�<����M��� ��.vû��<
�n<:�A<S��<��+١����b��<�¼�x8<#�߼��s��g;H���;���<�};�l9Ky��,��:�th��Z\��3,<u)X��><w*�::Mf��5:���u�8X��<�qM���8���<����m-�Ǔ��h<TJ��p#˺:&�:��	����9��;Hv���l�:Sp��4˺�\�X�:��:5d���:E��:;H;�#z:뾧�X�府P:-޺�o�ǩ�:��:5��:��&���|G�:N ;h� �=�.�KX�#1���\;�S���d>x���=kU��}C<a
3>��K��!M>Li$�x�M>,���&��I�=
�{�8�t��<5f;�=h�_>b�D�P�O���;�(N>��Źv�����:zY���]89�	:�����9	����������Z�9���9.i͹I��9KZ�9!��90l�A������E9a�ɹ�Q�3�:����9��㹸й���9>5�9��=X<;�=BHd;���G��=���W=��:9�96=7b;��=��Lf�<N&���<,R��\0=�_f=9u<zR����׼r;o$�$ �<����<�*�<-��WK�M�ϻ��;��/�~}i;����k�.�Յ�<��;۝�:��<�&�#�-:(x�;:.���r5����;�m��^j ;jI�;�M������;�U:Q�����;�ɱ:�b	<�#���>;�dD=CG��a�K�7�=<�gF<�p%=嬼Ǒ=���;�o��2��;��m~�<���<x�F��p_>�-ܼ�[�:L�; !=�
=��<�8�����Q0»�:��t�;��9��ϼ�h����:W�t��#����
;�N�9$����:B'���:s�m��9`Nٹ^��H�:.�:�ɺ���M.�����d���N;�킻�Cw8s�v�!:}��8�ۺc�ʹ4<�:J6�)�6��լ�J6-:y�I��7+8L��:,��3v2:}���b:��ˆ��Y�:���9vԹ�F�9r4+9�.�9&G)�����nv��;�դ������m�:uv*:4�8:�ع��`ֹ��:��:t�a=	�4���3=��3;)&7���"��gJ=��_�G�<=|M���.=��1� u��6`=��c��9q�^�Z�0����=��F�`Y[:7jK=
%=�S;�ft� �`��i=%gT=���Y�B�&;���a=F�ܼ�x�:'O�rq�<2-�5�7<��ڼ�9�;���l�O��;�;��	� t�%����;�Y<��e��V;y�F�|��;��+��2���w-��g�;'��;>U�ݹ������so)��R&>�u����<�^B>�@���>��E�Ă�[�S���T>tj�>>M��=�>��5>e��>�H9��>�<c#�2`�<���3��nH>�qf��#�>�q�������A>j �>3�=���=ư���=g����݁��-{=���I�u=L�=6�|=�v���ɽ�²=+��Ձ��(��l0!=hD9� ��=���<�h�=�0x=G�w���=�����bT<Q��=-Z���$������pJ��
�;D���:�#�;Χ����;q�����b�kُ���;�^�;	y���i�;;h�;��;�L;�1P�Ֆ��2A�:�����Џ��֚;ݼn;�R�;ސV�K����Ԃ;�s�;:u�%������;�y����:q�;��8��;�仍/���ջ�`�;�q�;������;���;�K�;�8;�~���B��Ww�:ӻr�һL0�;�a�;���;S���^�Y
�;���;zӆ8�	`�F�¸��%:�dV��0:����l��[U��:��ƺ�fj�9�|�N��9����+��D�ٹ��r��m��@�������@�|��:�:���� �b��d����9ͪs:(���n���G��ב;����-�:��;Y���+h�;I���p�]�`���_�;Y[�;Q=���q�;4�;BE�;F�;��Y�ƀ�;:c���ї�"Ѧ;azr;_ϖ;��b�D9��ǽg;���;��=|Y��=�<�G�;7if��*1�� R=�'��BUB=��p��8=
 ;� ]�}݆=�����*F�+%��޴C�Ʋ=�).�# ;.l_=M�+=B�?�!^|��ㆽx� =��|=��;�1��?ϥ���K�K�;z>���A�:��;g�ɻ���;�ʻ�Wj������;��;C���?��;2
�;9£;{��:�N��Q���0�:q���b��JH�;���;B�;�U������X�q;�<�;7���{�E�x;^���ࠄ:䋝;�>����;�w��sS4��׍����;_4�;�����y;MW�;)\�;�N�:�R�p�[�
�:�-q��o����;��a;�,�;f�Z�BJ|���K;��;fߕ�~dA��T>�댾-��<:�*>\�9�ei�>-m<��O��Z@�+�<>���>A'��� �>��>"(�>$�����=�P�jq
�'3���0�7�1>�rp�>{�>�<�ɥ���>�U�>�̬��K���q>�A���쬼9t*>��&���>05*�������2�?
9>���>������>`�d>�[�>�j�7n�=��v�����Uf����"�Ls%>���AK�>P�f=ֻ��f�9>�F�>�󫾯�����p>����6��� >���18�>�~ ������-���->l��>4N��Z"�>Se>�>���Oj�=��t����h暾l���>�d��݅�>=��<����:>�m�>�m�;���;��ʻH�;�*�;�Һf��:I���aH�:�̪;@;����4��4�;Sz�����KB�����:vg;^�;�7�;�,4;ו��Y����~���;�φ;K��|뎻_p���?����x>�鞾��h�e�>>�1�� �>+*��⃾�-���,>@��>2Q���l�>IQ>��>v���Q�=.�h�����r8��d�%���>�Б����>	7=�����5>7�>��<N�R� oǻƃ<��;�<���=<�8���<<[���(F<�2C�:�`��$�<���yGܻ�2��\�E���:��;�#7<��[<�3<6�=� 2׼i���B9(;�f�<K��;}a����==�<�����<�{��]�?���L=�B���>=�ɿ;U)1='R3������X={�a���Ľ�S��.=h3=��<�i���*=�'1=.W;�-ּ��B�F�E=.�D=4Š�@!Q�
��<̗���B=iŻ	���ʼ^�<ۈ��ox�<�7
���<1�Ӽ/�u��߽<�Zм��-��O���q�;��<�n9��;H�<��<���&H)�b�����<���<8�=��C��z�=>>Z1��#�=ݱN�qV���ž=�νҽ�=.%>�ʺ=�H���F��T��=��ѽ�e����ƽ��=�x��� >{��;�ء=���=[N��ieB>��׽o�O���=Hɽ�׽�܏�ؒq��Ea>���!�ɼ� > ���>r1��}e�L����>>�1�����>��b>�m�>�����S=��Z�Vv����v\�� >�`����>���=F��@2>�ћ>�ic�S���Dx<l�����;�ݴ<� ���c<���������%��	�<�=W<1�g�j�]<���<�_<=x���L��=�x��;�:?��U���Ű<T����k<��Y�o�d��o(<%la<o�û`낻v��;{�׻i�:��;�Gʻ��;�Jʻ7��:,��Z��;��;iɻ�4�;z�;0a�;�(;��,��Ѧ�:i��������;�t�;9��;�񍻸�Ļ��;F�;w0�<o������=�O��*�U� �r����<f߼h
�<�;��L�<~�������$�<M!��
q����ع��y�<�������j�<@��<,�������'�<���<MG�<YM�ѽ�; ߺS��:�q^�NI�8�d;��4�:��G�Ⱥ��̾�:^e�:����1�:+*�9ѷ�:ր���ߺSE������T躶���j�;�?�:��:�e��*�:�@�:W=�q<=�'C�	v�<ރ�|~���=E��l�<�͍<6��<3��cu\�w�=��&�����͆�I+�<C=�<�n=�s�8>$=o�<�V���@H�����<�@=���Qe�䬾ɨ��u�s>������t�X�,>
�(�z�>H.��ӄ�i`5��;>3��>�༾\+�>�%]>�o�>������=hmq������L���u&�lw,> ���7�>�eF=dἾ{�5>x�>�=ql�����?��<�J�%ۼ19�<���.��<*�׻XE�<E�ؼ8u�<�.=�[���T�$#�.�l�讀<�b�<���;'@='��<t�ۼ��4�����<��=� �;����Я��ρ��;z>X]��k���n?/>y�.��(�>5&��y��f;���<>��>匽��}�>��[>,1�>g�5��=��v�>���ӝ��Y#���">l���t�>�;=�Ľ�˛/>]`�>�M�<�ӧ�k�N<��=�W=���<�i��i��g�	���W2����;��o�[��<V���[S_<�Ӑ�6��Є� n1���=��<�H�����<����|�%��0��|�<hi*<��ü�'�<〼�j<�=�!=���<Ƭ߼I"��#;(q��@->�+H|;Gƒ��&=�~��_ �<Aü�j�s���������=�4�<f@���Z�;�ҋ��QV�*.�ˋ�<��;x׼�¤�Ql�2lp=�"��n�c=dd�=�-��$�<դ��ҟ�����	C�=��<�)��<��=���<u����N��a%[�e^W=Z�μx��s]�=`j�]�<�t���~˼���=�ɂ<Ȗ���`���E=8�����	=�y=�w�M�<ġd�L���Y�� E=QШ<W�K�>)�<�UP=��<<a���I��Uo��,=�K��X�U���b=S���~�<�qY��	��0�I=Ig<�A=>tJ�0�;51=w�=!T�<$����W�^�;5mJ�΃�ҙ����^=�>˼�<�Zݼ'���(j�́�ym�=��<,sr���-;�km�f��Z�#�Ұ=���;1��D��n=��T=k���?�
=��m=<xh�y�<}jR���6���;�!l=�ds<qw�@Ui<-X=Kn<��0?W�2�O��|�<YMe�?Q���o=�f	���<l�:���~��\=nt<�Ē�>*Y���=�ߗ��.F=$z�=����D$�<�/������݁��t={��<�M��P�<.y�=�*d<y]��>���bXe��P>=c�p�{1�����=_5.�^qS<7ܕ�Skf���=\B�<7����'�'�=uc;��!=iX=�oV��q:���5�#�q/�w�!=0�E<	�;��B<E�'=E�<������/�_�2��=�h+�>h7��2+=@�H��Ra<�nb����=ޫ(;M8�����1=Oa�>e�<S�@=iW�
=s'�����4��W�=��B=�+9��m1=��=��N=E�&��٩�p��d�<�5���H���c=�{�<�(5=e�K[��I=%�&=KUw=���ma�݈=��\=5��;嘓���u��
<�hz��r�:�~3��+$��]�=c�c��
�;|YW�üc��w�����]>#��=�G�"w%�<ҧ��N*���;��~=��;%�i��풽͑����<�Ȥ�~���+�)������=#~�Z��	<����<y5�=��v���n=ԭ���T�=L0����#���l��%����c�o�<���<�J�=����і�M?D=�y�=���o�x~�<*2��a3<�+�<�EƼ+��;��� �R��:��2-R<��;�<;Ye�;��<�4�; �㼺J���t��Xj<�0�m$|��C�<�{���%<C<��D��X2T<1�;�#�<���i�%<P|�<x�=�b�<b� �D�˼�5Q���ɼ��~���L<��z��<#�S�r�<�}�o���u�������=���<�׼ʡM<8�)�����L?�H�<�|N<!���)��h�b�C=/�N��F*=>j.=}1�6<�5�����"�Q�P�Q=B,N<��1 �;�T=�6<��R��	cT�7�v=v����DC�E�R=��ս#~;dSe�	Y4�cVu=���;i)
�7mC����<�CS��b<����>�� ��<���;)y6�:�ں�l�JA�==�׼ρ�<�K�=�d=ol%��1=�(#:R+�<�Y���B��eA=(*�<��<0�G�X <��=Kڻ�$޼�s�<�.�kx=hS$=��B�!�~;�p���"��`!=i <_^ǻ�8<��%=%��;�7�a������n��<D،�b ���=}o���I�;�&���һ� 
=,��;Jb=[�0�uS�;υ)=�	1=ӻ�<qKۼ��Kʣ;��l��5��z�;�/��/f=*3ּ�$E<J ��L�+�#�ϼ�1	�>�=R|	=�y����;O@���<��
�<���=xu�;Ђ��﷼1w��:=ʘ��UY�<+g=t��8�<�E&�)�(�Ǽ�_"=��=�s���vk<p�@=���<�	��Fj�l*�!.<A����,���0=m���~�< ��j2��1=w}<ҙ�=I�=�q޽�&�=[�����>�|Ͻ>�Q=��>�q��T���]�=KRԽ������ý�o�=��=�j�=�[*���=!%>b-�S�˻�4���i�=~�=�����Խ��*�g��P����C�4E�<��=�]�����<+�=�#�< W#��==�=�.5����</1<�w=>ݼk	>��T�;��>=��%�E>��Q=56���'�<��8���^�p~�<ɐ=����l�l��ǆ=G���@�n=�F�=q뿽׋�<�箽��������@0�=�l�<n4A����<0Y�=�j�<�򡽁'���o� �j=�nؼ����I�==g��2��<:���u�¼��=�և<X�+=�r(�ˡ�:<�L=+�8=��K<�k����<��;���+޻;����M����;=�����<6(�A�4�3��Qg���=1�9=��V�/7�:���ٖѼ�"<�$5=?;��$�x	���F��5�=�R���Ia= ��=����-��<dt������듽gʕ=�ǫ<43�퀼<eO�=`?�<���򁦽��x��U=�ܼ�ۤ�|��=��z�<ȫ�XҼ�ۯ=@��<@8�����c�<"!��h�~<��<�[����;2�ݼī��1<����<h��;k�5:���;�m�<g�;=����¼����Д<-�*�����q�<=�R���$<�l��C��0�<O��;� r���I���=1���gG=�W�=^h����d<A⋽��{��{��b^=)�<��E�%�;<8t�=i�<�-x�j"��J�f��.=�6�'D���q�=�/G�מ-<�c���g���=�=s<�=��r�;E�$=�)%=j�<��{y�x��;����Mf����:�О��d=�oռ&2b<�����u���ŵ
�.��=��	=�V��u�;������-G2�H�=�!
<�����Ģ���T��Q=�᡼�0=�s�=Bٔ�¡�<�~�����6\z�|ly=���<��%����<�z=�-�<�i�ja|���[�g<=�|̼u��%E�=[9q�d$b<�P��Nb�����=y��<4=��f��j�;�1=�w&=�n�<�_���� �jR;.d�������к貼�=�	ἂyQ<�.�����N����>��=0�=
u�?O;U"���}����.�1�="��;��Oy���A�� L=Cx���>R=��=�b���Y�<xg��_Ԁ��z��=�&h<������<巋=)�z<���m�����G��s=��	%���f�=Yˌ���<i���y֚�+-�=2�<<:<<��Ŋ<dX%����<�&)<���,���-N�;��8����<E#m��GB������=鈠<E������8�⼹�Z=L "�;@ͼ~[<P����㖼���eGD��B�<��><�`k���<Q3��i1@<��B<'��tdA��\����):Xd�<r�0��'�d0L���<V(�;9�ż�;px���m<;'~;��G=�&�:�5���=\���<�N3<�o=�^ܿ:�@��%S|:�2��5�<=���<��<͏��O��Q��/�3;��<s���]9ʍ�I��;ܷ�;y����oQ;�uL��^v�h/��0�=��x�a�P;��A���=��%<<�G� / ;a¼��j;��O=6�D�m��; �O=V�<��<���#d�+�̓���D�c?�<��K� <D=��4����<^h
��2��x�;<󰩻�
;��=��Ӽ�<�a=��X�_*t��R6=��:;����G��۪�<YMռ��O��f <v��<t���;#�<$���&=b�˼^Y<�jY;9�#�&��<�����E�<
��;Z�ܻ��<,A�<���u5��;w<V<=ͷ�<����Ŷм�߅���~<�Ҙ�x�:�:����<9m-<+�e:��L��xu�a�����<�/���	�k�����;�2�;�f��R)C;t�I��m�:j�ǻ�=���-ǡ�����u =�$<m�6���:����@x;(�k�_Ĥ�3^�<��:�-{�<�*"<���%I�90��t���>�<Z߻��f�������<��<d/I;0�U;�޼��r<�Ã��D���"=!��<�<�:�����S��<g�;�m<���<.@a�xj�<�	�<�V�<����G����O��Z�=������<M�,�Ϝ<��+��=�<�EZ��������j~�<�z3<���;0p��{<�zJ=jꪼ�Ƕ�]�x<�F�������ջ��<��R��H�;�B<���;"� �n൹ң4�.�Q<O�7�9m»�E���Y;HG<G���'<vDh��X�>p�:���<��������;��]=��A<,o��È�8�Q�?R�;���v�j�&��=�9�d�_=SF
=!	����%=$"�����9x��X]=��=7��ϛ=0(=�N7=����I� ߔ�o�=��)�A�g�+��=��9>%*=,p��n���_=R�<�ϯ���к���� ;ae<;�;oD�s�v;�%';��ۺ���w^`�+�<�r6;�5<Az��,<d�i�A�<zcA��0m=	d��!���8�Qe=���<����W��1��3�
<�.���e㽔(�=Y��Y��<R*�=/н�0;n��ua����ݽ0p�=��<ygԼ��<��>	=�O ���=T=ݽ͕�=�J���dȽq'�=IY6���<��=¼���=\��<��ݻ�e���I�<ϰ���+�<Ŵ�<5q�C��;֞�Ð���{�}�<.�<nL���#<X�<n%<�]����)s�Bá<��x��uǼu��<(�ݺ��a<���![��ҟ<���;���:a����ƃ;���;�kk;p�;+o�J���ӵi��K�;x���>K/<�h+����;?(�;?(&<��M;���	Κ����:�r~<B;�N;5���,<��4=��;�?B��2;��";̨[;W=��N���<~�=��T=Bp�<xv����C����}[a�������<1^��Q;=�8C�}��=8.�N ۼ��*��c���}=c��<�WݼC�<v�ý�n�oX��V,U=n1�<I��ְ��Ɏ�.��<�����~=l3�=������=Q~ֽ^��:���ɽ�=U��=�ʹ�[ٴ=Ɠ�=���=�k����p��>7�*7�=W���:
����=D��=�ʸ=�Vn��|���9=��=p���?���|]=� �I]P=��0=��{�Z�<�;f�Lw��n�=o=������<�c�=D�=a���氽���4=Ƀ�1�m�4؆=:�����<??����׼��=�P=o�{�%�<�h����;��8<j<���G�f�uc���CF<u�H�n�ỪY�ww�;��-<��M��<c\h���=���);<'=�m�X;N6��Ir=�pY<`?���~��{$�!�;�N��F�"8V=�D��+=^#L=ԌP����</˖��	2�8fX�a֌=��`=E{�$=�>�=�v= ay�'�����s�5�<��-�W�g����=�6�%bR=�j��;��4ʑ=;w�<�@$�R	�r�[=��ϽJE�=]�=>;�'� A>8�2�����#��&>ެ�=t���<>��=�}>���@ߟ�A!b��q�=D���f4���$>��=]�">�I��5�\��=�>�,.��zt<����m޻b��;�ᨺ�邽l�=!i-��+=�g�]G=v໿��	�=.�<O�(=�x>�]�)��<�ۭ=j4Ҽ�-8��q�=X̼D1=^"L�p/"�`[�<a�<o�b=�h��_N�<�4q=C<�4�<ٳ��Tn����Z�'w]�V�
=8*�h�b=��?��.=`������n��<�H@�_��;Y=2=���*L2<Ke/<g�g�NJ��JP=�Q<�x'�����%H$;�_��Z�;�4N<[T�N�y�ߺ��r:IZ<}y+���ފ��߶�;)�<֓�#@�;
�C��9��Dۺ h;=���n �9e�v�`6=p�d<@�h�3:�8����!�;�[=�N�B+�<U�O=ΆS<f��<QL�;�a�)����
Ҽe�N��O=�B��TI=Y�-��4=�f�Bc���4<s��:9�;�='̻���<<��<έX��J��5=��<q��%���6��k=U1����<m'=i�K���<$�/�ͤм� ��Z=�G*<�O��n&< �+=�\*< >��+&����<�a�� �;�'=��W�X��<��8� c~��� =o��;��/<Ƹ��*~e<k5�<"!= �<@l������\LѼ`�;�4ڼ@�<	�B),<lՀ�E`=�6�/:���������c�<�hV<ݜ���:�<�<b����o����<�D<G���"��	aO<m�;A��;�_X��y��s�9t�7���<�ǉ��~a�*��9�o;?!<�= �R&�;\eV���{�M�����=�OV��e�;󗽻�G=WFN<�H����1s��h��;�(�<qhs<��׻�|�<A<�<{��h��7J�&#<�S�Ս�;�D@�����Ӊ�<򗹼����od���<�"�:?
�<T�<�st<��[�(�Y�}�x�3���=	?��.�<r���tm��5r9:\)#:�����+<;<�RȻ�"&�����[|�8D�<O�̻ހ��7պ��<z±;k���h8$;��>����;+T$���2=���:�ԇ�=�9=B�,<v'�I/�:ކ��k�;[�*=��	=�ȼ�*?=D1:=��i�x�;e�)��_<��><�Vٺ+ �f����=|��٧Ǻ�楼�i�<5�۝=�eK=A��<���;ٍ?�X���)�.�Ƽ�=[������~-��'����5=d���_��;�w=N�B�ك�=��a�_�6�Cn��S$z=�X�=Ɋ���Χ=*A�=�=1�w=����$�_��=�2��T=���<�=} �]'����$=5�=������w��A��{���}r�A�;�K��O�=D��;��8�8��^<A%t=!���&0�=��8��=چ �ꁫ�HG��A�I�DK����c<4#��.�<ܚ�=�W�&�ƽ�#A��S�=�����0��۹w���;l�t<�s�o�=		����<|�;ʧ5=��=�M߽��	>��';���=Z�<����0���$�>�ڽ�-Ӽ[>��,�����=s%�;�G�4n�<�O�=$N>�2�0�8���`>�Y <�!<
�:�.�[���bl�t�4��c:���h�^>Z	W�Di<�kY��ݼ6|���X}<;R�<�8R>��;��<y���Y�J��4f>��<��M�y��D�6a;^��gѸ�wwɻ�������=Z&������+���<7��=�/޽W�>zL�9I	�= ";ؑ�;�n�<C���^	߽�9@���O<n�<<��=�s�����uƻ���=��ٽ S���4k9L5�L�߻���;����=�n��9�)�E�����y<D��=>������="� <]�=��;�߼�)���<7QŽ��;�;���/k�=U�ɻ1���"�;��=��=.����<�p>�=1=���<8|9<+Z��;�kּdX�;�xi�<��;>Y���/�����(ټ�����G��,=w��=�W���;�����窼d�>�K�<����>�@G<?�<B�<>�FB={�����:�u"�>�T��O�;���:r��*=��A'>O�&�����6�nJL��J���w;e`�=.�>E^��Z�׻D�R�����31>�5�<&,�9��Jʼv�����&��2�:���'���>�R���!l��D�1=�t�=���9�>��<�A>��?��m7���M<���[��/��?'K��#<5w>�(��^��E���5>�=���͍�<���=?{<yI=���8��{�5�ٻ-���9�;8E������ƛ=9�������Ӛ���K�8�[���1�<�}�=qc��V�;X�^��`���#�=13
=|��Z̽�xȼ��u���XD���a�P�,�ͩ�=�2�;F��;]�:�r <Vx�=����߽ >��<m��=H��<A�?�I��9c���4ͽ.�_<.���<`��=(t_�bܽ�!�;h��=mĞ=�	D���;ܑ�=�A�Rt�<pmu�)��l��p4���'�M<ɡm���]=�냽�hG=����s�:*!G=�%� �I<�n=����A_<�[<�0���I�=T|=� K<*����ں�p�<�*������}� ��I=���;T=,��<�=��� ��nR.��nb:`���9��t=}&=Bx!=�>��K)<���<���N!�<"���4,=��<~Yڼ�!
<�	�`Ǽ��R<����Ž<ME�������=�͓�ձ����m�<v�=]H �
�
>}�P;�d�=���;yd���<�@�<6J��ڼ��.(<+��qe>���^��؏:
�>�>y�弴�u��{>^�;T-5��3�C�����"Ͽ���<�Y����콅�!>2��|l�;j����:�H��	�ĺ��?;<�>S�;�G)<X4�;)m��b���+>�����Z��=��&��p����=?��<~.0=^���_W����<�/�<<�=ӗ[��s=!�3��_>��(�Sݽ���<R]�=n��=Ƚ�=ʐ�� �-�Q�*u�����:�=��;[<��{,<��_=֖�uz�<��<��<p�f�����Lů��eR=.0<�a=ɪ���wk;*�c� �ʼ]hI���BBV�GlH<!��4��<0��f��<"�=�и�4���<��̻VL�^���;����;����O#�G-��u��<>ǉʻR�<���T<���=u<��6>���<�� >��C��ᾼ�{Y��N7�%����vX��k<�<"�=��:<���"��԰>_�=���;|��;��=31<J6�0HX���-�ryw�]Z�<��L�+���:�`�V�*=D;Ǽ�_R;�:�d�~�2h�:��غ�%=�A%=�]��m�;P#�<��ԼY ���A=�.<���<�=B�����=e�m<�S)<;긻����ĖO��)�<�:\����n���=�]���=��Ҽį��0>g���M<xNZ=�.m=�*�oSz�V� �H�o�4!K=���Z�ȼM��=�p�����<gB|=j[޺��.�����>$�;�%��0�;H�;���ږ=��u���=�f����<e�ʺ;B��W,=�ܥ=��W�<hj=|�ڽ����3p"<v�=��<�=��5�C>QJɻlm�;�Nm>��G<��9<�ϼ�L��볼I~Y��y0���W<�D��W>��Q��r<ɩN����;�^Q;ї����H<D0K>�����<�`м|�S�J%��/�^>�S:�uL�,罤٥�i���g�������k��S��A��=����5<"�߼'�<�=��˽�>4;��	�=�N���4��H���@��G�ս�n�:'�;K�;<�]�=x1��f\�&�;S��=',N>�?7��n��X�d>y��<xz�<Z;�l�]�|�Gj껈��;A���(�F�b>��V�Q��<ւ[�
��6�=��EJ���<U>�k<�[J�ǩ\�LS���h>
�<�PN�b�<;��=�����>:��!"?��;_=�G׻mLD=(n�<�|9=�������|;��o�>��0��{x�o�q=�4=�6=�t�w�T<�+=�0�$C=�I�E{4=6X<N�ur;��=��<]�"�k(>��f="%�<*�!;B��7E���<��8�R�߻��oQ>s�W.9q���eC�1)5���Ļ�$P=�|>	ʻ�|�;�x����'Y����>��<YF��������P<�
�%N�:��=�Ċ���=��I��Լ����=���=<��c\>��<�'�=B��;N+<+�:���;�Iݽaî��V��z�;���=�i:�����;��=X#>kۼ2ɏ<f�<>�S�<d =5���+�z�n(<��	��3ȻhF�I�4>�1������?7���<�f=�8��<o�4>C�k<�_���a���3�'8�G2F>���WY3�*L���^�ƻ������Y\�;0*��JX�=V��Oǅ�_.!<e�<uc�=�uǽo�=�o�<,��=E3ػ�n�����8�׺-ѽh�;�&�-��<M��=8����0/<���=W%>�Q=��<�I>�P�<+&���5��b}4��(��M<�=Q<�����s!�%,<>E=.���2<�!G�8�伃�;e^�<=�
=��">�9�;..}��6
�A$+�M|<N>N>C'q<œ<�7�����,=�֎��3����Ľ�o���O�;Ѭ�<��;8,=��B���:���<��%��,�<�����=��d>c�_=��<����B���<�3�;���;�o�<ß=��/��7H�Rm=J1=f!����@<�aL=��=v��<��;�<��y����#��P����<��~�T=/$D�a�X<U@F�R�7�t����_Y�!�">��=۷@�7�=�;��������j�=,�
=�]��Q�=�澼�Cf<�=d6�=<g�<�;�����/����Q��Q���<�1ؼb~7=��"�5�}<&��)�M���G�7�>��a=�C�%��<�c��x�>���5�n=.}=�|��ҕ�B�G`�=�ߋ��T�=`��=��߽W��<tҽ�%���ù�5ٴ=���<��,���<�=�Vw<uʽOȽ"6��$h�=���-)׽f�=�<���*�<��ɽ/���R��=C��<J]O=1�9�`�~;�
v=
��=S�w<Vd׼��c�%>���;o}��2�<R��~=)�e��y3<��o�/A��ݜ��hC�ӫD>��=��D���=~�q��<9��˒�<7�=��<�f��!6=�̼��<q=�x=�9�<��B���W��)	���(�����<{w���^"=���d�<7:��C��Ҽ�P���>~E=@�3�7��</_üO�ϼ�cs��R=н=)��⼶������=������`=P�=�J����<����㠎��ዽ �=��<���|Ч<�=�Ț<�#_�ȗ��n��0PS=��u�ʽ�5�=
��4�<�e��E젼�Ƨ=�}�<�%���1}�F��=�U��A�~=�ؾ=^����O<����(���V����=*�<�]e��<�<���=��q<�V����������b=U�A�4/�����=�3$��)1<⑪�y�j���=$i�<��:=�Zz�� �;�R=���=mLj<˹�F�H�t˂��������`U�<=$�dFi=�9T���^<�M�p�K�� ļE_D���L>��=
r?����<YpL�[��k���ώ=B*=_���@ �׬��O$�=��1��W=֯`=�|�4��<z�I9ɽ�~���v=>&=�'�%�<�:�=]>�<�]6�������4n�=g�������ӧ=;+���` =I�.�hg����=�� ='Ȳ<m�%��	=��<tJ�=9�%=|�<�X������~����d
=<����<�żQ_=eM��]$�n�'��S����>m=ˇe��o0==��!�q�f���H�=�;J=�h9~|7<3���֘}=PT<���=}��=g�潿8l��{ܽ�	���ܽ�%�=�4;Lѵ;����n��=4�໬5 ��-E�N8ǽ�>`�;<�ɽ���=����&#;�雽!<9��=�p��p�7�2z<��Z�~�����;�\�b$I�_1;�κ��<j1u���9�Z;YO���[ <�e<s *<i��F���m�	<B�&����;T�1��D�<��1<����_ϼ���M�{!�;���<�궼N�7<Y"=[v=�I�<|������v���)�����sD�<���w<#=��<�<��ب�t4缭X8���>?�>=�Q4�t��<���������|��AP=�`�<�u����H�`M���}=2����{=�K�=�6h���ƺ��x�W0��c�s���^=��<f��W&���_=���:�$2��Ys�i�[��9v={�%�������{=����͹�b���� X:�E�=�P;"'��s��,X=A?@���D=<>�=u(���s$=�����"�����9�=��]=��F���-=�TI=�R#=�(h��PE���\�e��=�e	��ǥ�?I�=�����%=3=����\T�=�#=x�W��|��@?=q Z��:,='�*=9B)��9�<��L��:�N�N�
�a=�&�<
�[�sG�<kC�=��<�[��S����z?��MJ=�����x�9=�N��G�<	Y�;���K�'=Xz{<�=��Ѽ_�K<��=$�=lo�<e;����C(�� ��� ��4��<�E���.=����&�<-�"�=�6?	�m�G�F/(>��U=��F�.	=����м#Ӑ�u`]= U =�����/�^���U�=zҼ��L=h��=������<Ƚ��p�틛�C�=W%=��Ӽ$�=9C�=�I�<Ũ������*J��\�;=-�9/�� �=�=M�=�C����?��=���<�ٽZ�e��Yd=n�ʽ��j=s'�=�Ʀ��!�=~"彌���_F�1�=Ӂ�=�ٽ8��=|��=w��=�^)����d�0�}=�����gн���=�}�;m+�=����<�Ž�ܓ=4g�=d4��I���>=a��R�=���=�ǿ�ƃ$=�'���=3�����a�=S�W=�XN�-p@=q;�<]�%=�<���׵�q[D����=U3�>н�6�=G�r�ݵ/=J�Լ��r�=�N=�)���ɲ�q�=�҄�΀�= ��=W���<�ٽ@Ľ5
���=�;�<��5�|<���=g�m<��ӽxb��-u��dU�=�u��_ܽ<��=|���r�c<�vʽd���2��=Ԛ�<��=d������<�G=�Ř=@?�<���c��ʛ޼m ����-`�<4�,=d��y��<k��O0��)��9w�h:>@V=i�N��e!=�4�+�缤��l\=��0=�"�>,��.���Q��=�
���c�=���=���)��<a-ͽ�j������
B�=���<B�4��Г<W�=^�<\�ȽŁŽ+
��:]�=/���E�ѽ��=�#�����<�ǽ ��Y�=`��<KT�;~<y����;�O(;���:�x��3����;���< �.;3gh�b����;�(;�&�<��;��"<�c��J��:	�;S4]�<5i<e�8���=.T�;Ql��U ;����(F�ڍ��1���?��=���׌=o��=Kܾ��<i������"6��u��=�#�< �P��J<���=%)<1���G��W���w+t=ݔ�p�н�&�=3�W���#<���������=��<33�<a�ּ�R{<{.	=�l�=<��<����rt��2���O���ļb��<k���!=O��R�<{�	��[+�v�����L��y>�kE=�13��o�<���7C¼GO��R�R=��=�~�����Uo��/�=�����O=l��=T��^ܭ<t��5���ڌ��}=Z�<��V��E�<�͓=��<-����C��v�y��0V=���bӥ�ˢ=��~�wg<é�������א=��<="=׎߼�?�<D�0=�ߔ=^�<G[
�?2��i���ƫ�j�����<Q���E�>=�y2���<�2*�!�,����Ys��X>��m=��G����<���	��+��Q\o=�+%=� c�)=��P�Ԍr=�`��t=��=.����g<�ꢽ>����(����=�T�<�����\<i��=��N<E���ǥ���Y��y�=�xj����Cz�=�є�#��;�&������Ϣ=�4<�+���1����=�\Ž� �<a�=&���@�=ٛ��W���.��";�=x��=E���p��=�T�=jY�=��ٺ����g����!�<�Ȩ�����u�=�o=��=@"������l�=5��=R�"=�N$=���<��=�� =[=��=��!=�d�<6�S=[���*����=��<��<����3=e<��=nY4=���<�<=�>�<���\�+=�Q)=P��<�n0=�-�<�VR�ۼ9=W7=H��<�S/=�3=ou&=��.=��5==�n=�r�����?)=��<.H=Zם�L�H=��i<�i>v�g=�i =T=�=�<?X��g#F=�c>=YX�<��G= j�<�Dc�7'�<�<a&=�)�<> �<tv=å=���<f�;�{�<xu��ϪA� ��<A_=���;^�����<I�U��
�<�d�<��>=�	�<%=��B���=R%�<��<I��<46�<�'Z�N��<���<r�;���<���<3P;<�+�;-C�<hPr����<��i��,��:�<_Ix<��:�d�1����<�s�!u�<��:��<�<�<ty�;8���5<���<�Bq;���<#(;�,���0=�_.=&��<]�'=�*=�=�R(=�V.=�F=&�e=Ph�"��0�=���<��%=�m���@=�=�<�x>�Ri=�E�<�SK=��<#s���==?J6=X �<N�>=�7�<��a�(ϵ<���<��[<Z�<7�<�&2<_�E<,�<Kf9;j�<r ��Z�ռ�"�<qD<��x9h'����<
>�<V�c<H-�ƯX<jU�<"#m<�7Լ��<���<��;�Ӿ<��;��ռ;�<�F�<R��<C��<�v�<Y>�<�
�<s��<�������<çf��$��D�<v��<.�%�"�Ҽ�<Od��s
<e	���=��<&�<��&��$�<�<��}<���<QL�<C<�"�=i=fs<Wp=�n	=p�<�Ī<�Q=q�<W�&=��X��g���Y=���<��<͜��8=,�����y=���<o��<��=_�e<~�<4�<J=�0<�=�<�a
�<k�<�<L^���<�گ<�IƼ�_����<E輗	�<ER����
Y�< 覻������<.�*���;]�F�K�@��<��"ּ�ɺ���<�_{�~S�<^�������"Z=6W=��>=ڜ>=ReU=�Io==OFN=á1=Q��=+�������~/@=ȝ-=��u=�ج�7�r=ρ/=M�d>�ާ=	NL=�y�=��9=۲6��ԕ=ܹf=��H=�(q=�F=@?��m�:=8�=2!��ݵ�=`2=]����i����=��C��_\=�0�<�-�@_6=i_����귪<Y��=h�\=��̽$5����Ž��q=ZP��<��h����b~=ن��k=w;ƽɿ�;�S�;��;��;M��;���;�� ;Pn�:f��;*�<OC��:�&���;@�<`ﺆ�F�_3 <�=S�1$�<�ٺ~�<�<�O�;��.�0^j;�h�;1��9q� <���:���/�%=��=T��<~�!=��=V>=s =O~ =�L.=8HO=Ж� r���=�!�<g�/=9s���w-=���<���=x5_=��<)66=��<VS���==Rj#=μ<�Z+=�.�<�F���<���<�xB=�c�<fB�<�,*=�yE=��<��4�<�kd���7����<L�D<7��<�,l�D��<\`�b�=���<�37=��<	�B='�7�t�F=� �< ��<M��<"�=z�B����0au�_a�R���\h��f���ࢼ&�Ӽ>�㻤|��f�o�N^���<G�	��Ό������f"���&=q���V�ԼJ�,���F�v+c�|����滓�<�h�F�$=}�����<��<*�K�V��<%�<#樻���ӹ�<Cs�<6��<������j�<�ớ��<GdϺ��<��#<�^<�k0��������<d�'�vg��?r�2��<�T�����<�.�� �Ӽ9B=��>=��!=�3=�<=��M=w{[=�D;=i'=iڀ=�������-=��=��!=���VT=@A{<�!>J�{=�N*=Wa=��=�L���v=��G=Az=ʥR=�8=[�u���U<�##<�K�;��<,v�;����چ˺s�T<���<$H<�Y��I���Y]W;�0<�#�<� �;�W+<�<����î;ʀ<��><ZZ�;�����i�;^�"<8θ]P<;��e��=;�<)f{�%փ��ܬ<�6����=F[=so����S>)�2<`����/�=$��w���Js>��=�c��A4@>��:�
>�=�w���r<ڟL9���=P����ꍼ��<�_v<t��;� =�軄?@�������l�k����������<�}�����<�6��B<�����U;u���'=h�;�-p���=�H:ai���ݼ}��3���D/���[�<;�����*��x�
;��;���<qt�<H>U=�G�<���<��R=��==iR�<R�;�G�<�a���sK��1�<�&=gj-<�3����<Tv��s�<�O�<*e=���<)R=&yM�
�Q=;�<G��<�L�<�K=Ou^�z}H=�8G=l}&=��5=�6E=�R=��f=�:A=�*=�͇=0� �_0��^�5=ߘ=�}5=vi��\= Ĺ<�<0>��=�	3=�!m=�D#=�tw��=�Q=�)'=�]\=��=���Ӓ�<Z��<�$=^��<�@�<�f=�n=M��<ۏX;��<�����B����<�;=-��;\j�z��<J�p�H��<��<Su==9\�<�"=2ED���=C�<D�<�$�<���<9�^�q��;�m�;��<��;���;%}�;RT�;׻�;��㻳<Y�μ��Y�ז�;�@<�~[:~�\����;_�]��< s;�B<�C�;dA<��_�ė�;�i�;V�=;{�;S�;if���<#¸<�/�<�ѽ<���<R��<�	�<���<gH�����<=7R��t�є�<���<8��/�ȼp`�<�q���s�;k�C;7�=H[�<L��<���u �<~��<���<�ڽ<���<O�&�?=�S<=a�=.�3=ܛ9=��?=��K=;=�^=�U{=���?���{,=fU�<g�#=�����P=���<O>��y=
=�9]=4�==��e=X-F=��=+�O=y =|n�k��<n6�<���<Ҋ=���<���<���<�=vC��w=i�u���"�h�<�j�<%J�����Z@=�ɟ��d�</�|;���<TA�<���<W$��˱<9Q�<��p<<�=��w<YS>��7=@�6=ش=n�)=@�3=N�4=��A=�`3=d�=dTr=zM������'=,��<M�!=Y���*�I=�)�<�)>!#w=��=T�U=��=�À���X=��>=5p=�!H=Q�<'�o���<��<��=8l=>j=��=�9�<� =��H��$=����y�2�-=-w�<K�M��5��\�<�ϼǩ<�گ;!=`�<�L=J�3�,C =�P�<x�<C��<�4�<az\����_꙼��<t6��J(��|ڕ<SBN<\���	'L���� �:��;����@�=�[���B�����0ɼ/(����:T�<9������<��;�rh<�*���9�<w��3��<�%�<��=�m
=�]�Ë
=H:�<!-ϼ����=�)�<Re=�ǻѪ)��=N/�Զ�<��u�4U&=��7���=��<�1^�h�=�Gf���&���ݼ��=׍%��R=�8/��? �=K�=*�d��Q=>Z=���������%=n�*=D�1=x�;�
�ب=�1I��M=��X��o=�>�:���=*�(=��u�Σ=Yl����n����&=�1�BR=CZC���� ��l����=@�ʁ��l�=Y�a=�D��ϝ=*��f�Y��$:=�@��ы<�p�=�T=�}�� =ZW�;�j=�Ѫ=1���S��=�N;��=��L��<F��tL	=yZ�<�Q�<AT�<��V�(��<���<���B���<,^<�/�<��z�H��74�<�ϱ�n�;Ds��C��<`���<Q��<W�Q�
I�<_��I�����CQ�<K#�ި�<�4�����=8k=@S�ơ=I6�<�2���ּ\O=0=?�=��
;xr ��=lK�F=�HŻd�!=t�5<fX�=�%=n�c�o=�SY�S��� ͼ͊=�(�>�=�0���'��8iD��$�<g}��J����<]a�<�G���A�<��{�\�1<��һ-@��$���0��<:�G=Z���v�R=�ѩ;���<E[�<̿j��Ԓ<P$��R�<-|y�J�;�2g���B;�e;����為�#=8>�;����y��<I�!=h���"7<Սܻ n�_ A���ǻ���<����`=<��=!�<�0ɼ�uż�6=#1.�pd=g@���0=�<Ż�F�<�
�a�<��<�L�<��<��E���<N��<�u�Pi�I��<J�O=�G�<�<�楻�ѱ<(��i�=Z�����<�Ē��A�=N�m= �D��Ŭ<�9S��n��񬼗Q�<�A-���<#�B�T4Ƚ:�r=�چ=�P�;n �=�c�=����l�<Hu�=yǭ���b=��⽳�y��u�=���<�1����b��j�=�H���U��Ț��v�<�Æ=�I<���3�=�v=���<
�=�gG=k����W=h�'=ω�\� ==}���_,�]�5=�2n<�p;=F-��<a�~'=:@��O�<c���@=j����+�=G��<C���0.=����Ela�����w/=��i�]�=1�w� �V�D�N=x��=/��=��=B�=�==7��=��c=���C=t�½�F��C��=`��<՟	=}��"<%=�,/=baY�W랽u��=� C=|ԙ=�Nͽ�Δ=ƥ)=q7=|X�=,z�=�r�<5b<cld<��X;T@_<ƪi<M���
�94i<n��q��<���"���$_<;s;<ޑ��λ�$z<�������<�<��<I�|<t�;;rU���wD;�i<i���q<Ĺ�:up_�1ĭ<��<`���ҳ<�e<b�;�ne�6Ң<_�=Y��<�p<�2d�S�&<�W��<�=����<k��<�h�=��=<&�k�<Ǚ��v9���Ļ@��<C���ƒ<��,���� �/T:��=Il�:�q>;dy�=�Я=Dl̺��\=��~3�e��c��;��P��=j4��ƻ^��zT�=�>��=烅��=������=���h (=ϼ;��9=E�
���+>Zj'>*����=>6�.>�x�)*��V9>;�0��@'>f������	1>ؕڽ�i���#�q�'>���=�,=Ϩ��՘��>3>�^��ɪ�Q\���>�~�u54>U��AԾ�F�^=b�c=;O��n=71a=����Eő<�B=O����T=(�ս!\��=H=�h'<�������&�7=�!��Eü�	��bJ��fe=U::�o�~��:t27=���<r�j=<y8=�ӱ�b-�<���<��D��.�<��<5�&s׼=7'=�� =�̋;�J�?c�<���i?�<
���[��<iN��W��=}JC=�H�{=�<=�Q�Cj�����j\�<Nh0��p�<C��f�%�<��<*�%<m=�*�<�H��j+�N��<����CN�<����@����<�C=XjC�ͮ9�*#�<��:0֐��?����<��<ڪL<��W�
���R[�<�$�<�V�<�T+=ď4=�+[>
B]>e�M�kKb>A`>g-d�YYͽ�c>F�ӽq�W>�@\��M��,d>	r$�ܛ�_��
[W>���<�$/=���C=���\>�N���cֽ�'L>�d*��\>[L��4⊻�Ȕ=�̑=EЕ�ik�=�j�=}�����S<;��=rU�\ԙ=٘�c(�v�=�E��4��f.<���=R��<�7Ƽ(�S����У=Ep��F���4q��N�=�h��p�=D<��Am��ּ1��z7�=Ȅ��e��h��=��=͵��e�=�<�(A(�����L�̗�<hT�=t��<�-�;�=��6��6�=�_�=����6�=]�o9�n�=uZ+���!=!�	��o=�z�<& =�!=	�v��2!=��=h׼��1�0=��<�U:=��󻴣F��T"=~4-��?�<W ��8=�Va����=>�=A�z�7�)=ڙ~�	�B�dK���'=��=��,=.�I��y7�����Ҽ3�=VG��]���?�=�Z=𕬼�Ϝ=�ݸ���`���*���<9��=�4='��1��<�R<��i=4�=3W̼SF�=|�;��=5��H�<AL˼m	=�.�<��V<�L<�l<�AK<��T<
�;�0�;��B<~%��eɃ<�K^���ʼi�<<^��<��2���\<;���y��<QIK�x��<�Vf<>-<��Ѽ��"<�cT<rAV;Տ]<}�;�g�l���f`�24=��W��i��1=er-=פ����;(籼�th���:�g��^	<Q/���J=�̱�=cO�RXt�U=63���7=�G�2R<=�蚼_��<b"v��=�<ue�<�<�B=�Y����<ӈ�<�t��r�򼯑=W%=�n=��J;w�z�<�4��x=�q��x�=�|���=��&=%�d�$=�Xb���׻�I˼Y�=�G4����<�E�����9�;�q�;I��<CU<i��;�dG=�_=��=<�?=��X;���<,M9��|;r�9<�3�<�� =�z�;�aٺ\�r<FQ3=�^=#w�;-��<�5:;A$=*��;���<���;��<��<�=�$=׀j�Ba=�=��ͼ0* ���,=�\�<� 9='X��C�8���=>�0�|*�<�໑�7=&�ۻ]S�=��=�t��&=�q���6��!��S�&=��0���=D;�x�)�Wu5<�ʳ;w�,=ӭ<]u</��=9�i=X�e<�'�<��;��"='��2<:��5=U=�4�;/x�;�ӻ���<�`2=��<�*3=�E���U�=���:*��</�4<���<M��<�[ =y	=�Di�A=�B	=M�E��Jo�IY=��g��.=3�;�`P�$,=�����po��\�V�=�����>�q8`�(&e��z�<\o���%���\�5�=ӝ}�� =�{���i=�[]�B%<:Y�kR���;J�;��콎��<�=�|ƻ&���r� �r��<��!�J��=��k�=c֮�Zݶ<Q>��߽Y��&콯�:��,��8{<��	�kԧ����
������
��Dt��_g�:咙:#��� ����<
-K=TQ<+�=^C��0&W�� S� Ӯ=3���=��a�A=���=�6���L!�+b���������"���
�K)¼����㒼k�<A�?< ߀=_+C=\!g<\%����;K��<�⤽~��<�yZ�f��*���(��=[�A��0=Ҙ<?��=V�F�	C �ۀ�=ׁ�<�`�=��Z�'�J�(<��p=��|<�²=e�:�����/�<u��g5R��@&���?� �{�<-5�=�k���w���E����M;5�|��x�=@T��@'=������;�$%>��½��4���]6�
�L�4�=T��&:ȼk�#�!��_�9�躋��
<uِ������(��Z��<G�"=X�;�%v=��V����<�i��7�=@���t��<�����I=8�=Y���u�Q;I+뽍D1�GL��j��;�����R;Z��`�Ƽtq=�4=j7 >+v=`/�<�~(�9��=q��<����G(=����!�,�R"x���<)㞼����<�@<Wr�=��j<����AK >��=\>�0��~�<U��<D��=U�6=k�>r��<�=UV<��=�P=^�=�93�h�=���<���Du=ؐ6�S頼9��;��=<���\�=�W&<)(�=8Te���ԽX��=�'=�^�==��p�;5B<SE�=�z�<���=���;�+ǼF㻜����)�Ϫ�;�bA�Jz�dض;tQ=��x<�[=�����w�Z����=\.����f<	s��X�Y=�>��½���'*�8���ߘ��^������W�!��P��k�:���;��=q��������=^�T=���t�=�(���?=PgO<2}Y��Ί=�}>�t�]������8E=�>l��=�fR����=k��<� �<+����m�=~��;DG�=B㹼�Ӽ�i�#���"����Լ��c����s�<\�=m��A�Իv���IM<���G��=������<,5ѽ�R�<�<	>H0轸䍼b��VܻFʌ�u�(<���-R���'�A����^��I��݅�=VT���k>�$�>G�=ycH�/E>�����6>���<u$��c諾�Ү=��=�D.���K>B|�;,8>UX�=	`�����=�C=�$a=���BT�<�t ��Y�~jG<�v������OΏ��厼�����Z�]z�"{��w��;�VǼ4=��<����mș���*�S7�<ߩ��$<F*��Z1�F���ĭ��슼�<�����A��OO%��ܣ��p1�G�=[oӻ�B<Ow׽0����<!�j�p����g<���<��;9�>S�k���<�����==����<�/��x͙=~�=(a�&T�'/ս�?���5�S�"<D�ʽO4�G���42�%0�;~��b�e;������:��� �u��Ȑ��+��;^����缸Nݺ	��=E����q���e�Y���6���*#��0�<-B0�e$�:kh��'a��z���d ��䒼�R=��μ�^e�t�<��'>�Yk�)=T;^">�(>�Fܼl��=����Z+=�#&��
�XJ�=�>��]:S�G��BS=�1=X88>�~>� Ӽ�Q*>��L��~>�o�|�=����24>���<�w���^ٛ=5c�������D9=e�=�M��#=�=�j��'X�"Ǳ;D���w�=!5)=^ʑ=�f��w��=���('^=���=O����=�zA<�$�=Q��ȯ=����lF=��=)��ׅ��2���6滑�1�i�Y�D9��]�;˪=�@��6,�=�޼C_ۻK�H���=�#۽�:��������a=}�=Y��0�I��r�����(Z�:��<��M�H�>�)j;��<wM�⟾<#��<�	:AR�<F0�=�<�?>5<���8l�:�3���ҽ�sF>��L=R>���_>K����[�=�	=��h<��/=���:]��=��I���3=Z\5<ǹ�<p\�<��D=��M=C�9��A�=�==�[ƼZ��<�bz=�>B2=���=�e
��k7=��\��>��0;K�-=�f>��=�7�=B����Q=�	�/��;C[d<��?=(�伉y=�m�<��Y=�G$�&�����P>� ��1�(>�$�=�;e��g�<I'�f�M=O�<T���(>o8�<�uY�,il�*�)�3�>�>ЖM>b�����D>Hl�;���=�����R%>����b>!2��asc<��*<\��=��<cmO;��-�d�<�3����Ͻɽ<�P�DD������,>�ы�nç<���;�wO=0�]�������=��;A}�=�ƃ�kػ���:+�=lA�;j��=�e:>�-�������c&��KԼ�Eq���޽��<ɉ�=4$;�� <J�|��S�<l<���=�����{�<�e�����<v>r/۽D���q�������u���;���َQ�}"���Ǽ��<��;��q=!�P=R��9�SF�N��: 1�<�՗�v�=�{d���켕�H��=�&�:o=_��<�x=��X����*�=��<>�o=wO\����2�<�R8=��<���=I»�	k�0[�Z���vW�Jy`��c�����g>��X#;�����(X=�r�<69:�0���t�6�ƞ�<�f�V�M<�'@����s�ʼe�y�2���Q�<î��8[]�� �M�l��3�X�=+��<�u�<c�=�<��<��Cz�=��<�扽N �<2 ����q�(�	�A=�"N���=QM#��l�=�^L�|� �=���<�}�=:ݼL�<���:W�=���<2+>Q��;�E+��[;�{ܽ�����}.��O�>���e�<E�=*��^=丼��</�a����=����CO<emQ��R=�_�=Ō �(���������!^�Wfk�L��S̐�T����b��]=ֻ�<Z=�DQ=&f�;����o���=^�x��u�<�9��\b���W� [�=v�-���<�w<=?5U=Z+R���%��=�M�<��=d����]aI<��=���<v�j=����bB;���e����K?;s:<�u��9�̽]�<G!�=+�;�4�<�}��*�=�5��:�='5���g =�g���9�<Hj�=��Խb�;7���׏s�w�����j<S� ���>�sͼ�<=���< �<�0=WV����>����I�<C ����N=cQ���z缂~4��K�=�{%�[�<.{<Wʇ=z�u�t���*=�)�<8�<�B`��Ƽ%��;P4�<��=�a
=�K��=����1���QȻ�������M���]W���BǼf�<8�����EEF��oм;�^���=��&=�d�=\r���s�eǿ��Ҽ�T��l�$ߔ�%�����N�t�μә��,�^=�$�=��=�|�7���=��=��<�<�<���=q	�<,��=�܁�X�*���=򳖻�>u=,6��DN�=�+C=���>��^=���f�=�˹����Q�<�z�=�,<�=�� < 
��,�=ud�=��:���=���=h��<�3�<�޴=n��<��=�:��y*����=]=����P=ĩ��=��	=�N�>��Z=e��7M�=ȴ>9M!���<�T�=�c<�=�[�;X���HF�<�A�<2�3=1��<tN�<�n,=`0=���<���;}E=�曽�i�2��<�^�<&�;Q������<���;
�<��<��H=d��<��-=� k��F=ݢ�<�^�<=�e�<Cx���= ��=�<�>�=���=}0=i&G=�ݶ=�e =(�=�Չ�3� ��ҭ=�p@<c�=UJټ��=��~=��>�'�=%��<O��=���<G��[z>=���=g��<-�=���<}�����=@M�=x^Ȼ��=m��=_�A<U��<�Ч=�G�<?,�=/&y���"�D��=Al�K�:=���d�=���<�k>$�9=��{�=�c�~��eՃ<��=���:�2�=���:�拾zu�<�>�<>�f<���<*��<�T<�7R<0��<�ƻ ��<�#���'���<MsX� Z/����\�<��;���<��ŻWYq<y��<�I_<�&��w<%ֲ<)1����<{�)�-	�b��<gp�<y=�L�<K��<�R�<M=���<i,%��>�<����%I����<gՅ<�u^�����7V�<J���$;�c@�L/&=���<��=�$K���=#��<��w<���<|��<[�����=�E�=��<}��=̽�=�=;a9=�C�=1?/=� >Y��g. ��;�=
�9<L�=�D�����=��D=��>��='Ɋ<�;�=Qx<,%��E>=�(�=�t�<�=�ͻ<\����i=�*r=�q��:�=�.p=k�H�;��;��f=�p]��[h=�?e��UM�+_o=�Ƽr��;�����Y=��9��P<:j�Y�Y�� o=�ꜺGiE���;D~`=Ɏ$;N|r=ۼf8-�����=�-�=E&����=&�=�솺��6;�S�=|�;L�=��MR���=�>��`�=!�8�H��=*��<!b>��<�w�����= Q��}�J���;V��=��P���=bᄼ����e�4=��"== =
38=:3=��<�ZP=g3=�����2=���ј��t1=�S���	=�K�8�%=��S=�?`�h��;3�M=iP-=��=�a���'�=��=	�9X�)=�ߩ�����J�?<u�f<ל��wf<�\<���%ꁼ�V�<����^<8~��A"4����<Z��;F溔S黋ċ<+���J��;Ft����"��7j<�`���ea��Z�Νj<z�}��_<ɷL�`;�ˠ�=\ϟ=�w;(O�=ƭ�=��< &�<�W�=�K/=@��=��K�I]�%ߑ=�&�L.[=h�἖ެ=w	=�y>a�g=m1:96�=ĉ;3���7�<xæ=U��; �=`S;M⌾��</�<�T=J�<H`�<�Ѕ=h�p=��<q��<}��<A?��=9�2��<5s���/=��ñ�<���<	=�yr=C�9=���<�?S=�/9�N�n=�ݽ<��=7D�<i�<�H-���=Gy�=Ty����=d�=���;g�|=Wì�t�=M���UiQ�X2�=ț���%��y�����j=�@��+�<�k9��꼾;�=$a�P�F��y�&aq=2�����=��T���"�]�'6K���&=��'��~J�L�=q�%=zd��W\�roi����}��M���<�
��I���]��!ü�Y= ��;��>=�#X��r=�)��J)S=�$k���<G]���<.	���=n��=��;�Ѭ= ��=[�<���<�մ=���<���=�怽�2�ժ=�z�D9=X��^�=ފ�<D\�>lbV=믣:c��=ר:�''���<^�=[:
<���=@�;+����0<9<V�<*�}<��<��Ȼ��<@o<SY�i'<*��@�A���	<�<y���Ul���`<xv�<ni���ټs�=�M4<6;�<��I�
�<Z�:<�<S�<�z<�Z#94U�=�T�=�R��9R�=�.�=4+��P��_?�=|��Þ=]z��^����p�=�ͽB�������=$�J=R��A�����@�=|����I���h��
�=�h�W3�=�56��-���9Z=QL=�޼�eO=Аg=)�Vя<��6=�����T=���#�c�q�y=�en�ȧǻ�/��-=j�E=��Ƽ;����-�hR=nY׼�=T��̀;h�1=�����L=�S���;1��<��<�na==��<F��<P�f=�$b=�<�{<e(�<�^��<Bt�k��<A��<S`<�c �$�<`n<��<�l=9s=��<=�Y=5�v�w[|=e+�<���<�u�<�v�<��f��U�=)h�=��u��H�=��=�p<wǬ<�V�=�<��=�Q����<�-<�=��컃�F=\�)����=�*=��>�I=�u��V=�=��� �2�H��<P��=�\�;���=���:q���D�=��=�}-=�)=��	=��$=y�)=�{=�8�;R�=.����j��?�<(¶<���;�|��?�<��_;Z��<0.�<��B=�=�(==�k�3�==�n=U1�<�{=j��<\z~�s�<��<ª��[�<�5�<]y8���ݼ���<¦R�Nǟ<�0<�j�P�zF�<"<����	����<���շ^��^$������<B�����g�伱:�<uMļ@�<l����
�J��<��<�="#�<��<4�	=u=� �<�'仑��<��z��L�Ws�<a�R<!��V��}M�<���������;�62=.�<.=��L��g/=ӓ�<:Lr<��<0d�<�����=�s�=dn	���=��=�D�<ch�<��=���<���=I�v�5�*���=��6N/=���;�=���<�[�>�K=z���C��=���9�!�*q�<k*�=��;t#�=&�;���(=�=~��<n=�3"=���<Dʻ<]}=�|[�(T$=��}��8�0�=�R�<�i)�1�'�h�=���g��<�D�:1¹<�=S-�<2�9�^�<J#=�i<;�"=W�g<�DH���=��=@���4�=�̵=��<Pm�<���=��<��=rL���~/�!6�=ͬ�"�^=2� ����=l,!=�E�>7B_=��^��~�=���0&����<"s�=Y�;�o�=�t�;�1���T=yt=d�
=�	 =3K$=�'=��
="�=��%�d�=����J��}$=X��<'���N2��i=�늼v�<�-�;��=�=T�=Q�J���==�~�<D =�e�<��d�y�<���<�:b����<��<��Z�̃� ��<G ��m�<ur����4�@��<�5x�-���A�����<��ѽ����x6��GuH����<�fg��XE��!e��Ǟ<]�f�,�<��i�|��<��>���:h�y���w<��;W�2:�a�:d�>�r5:�q�}Q�;��|;���;0V;9<����0��;��h':~'�;�Λ:S͉��(��N�:<�X<L�>3�H�2��;�ʼ�:���=�<��=&x�;E�<��K;B��=�;]V={vP=a�S=p��;��0;#��:�=�M|;���;@�9:t=�2�<���	�.<^�Q;����w�=^ۭ;y[P��W:<c��:Y���G3�CC��� ��w�;���:y<�5�9Js����<�9�
�;+����:\,�;B�ʼ��;ݢ=�a��%�<�`��^��<S�2;v;��v;G%
:�O�؞;�^
��Q?��̊:�C�@�@�(K>��l�H�;���;�;�9���;��J���;b��;t�%> +>��">���W">��Q��4?����膹�s>���<(��;��˼W)�:���)�/>�^޼�0>U8>��:.z��-x<��;W�2:�a�:�8>�r5:nq�}Q�;��|;{��;0V;9<����2��;!���&:'�;�Λ:S͉��(��N�:�X<4>�$>�0��;�ʼ�:�v�=&�<��=�>��x�;I�<�K;��=�;�Y=��O=�S=Λ�;��0;�"���֪=�M|;���;�9:d�s=�2�<���	�.<_�Q;������=R�;\[P��[:<c��:Y����3���F���W9�;���:͋<�5�9Bs�ut�<�9�
�;��G��:�;${�����;�=\�E�<�G���Ç<% ;v;l��;G%
:�gN�##;�	��Q?��̊:s�C���@��gJ>L�r�H�;���;�;�9���;�MJ���;b��;��8>޷*>o�>mː��X >��Q�S2?����=��Q~>�<(��;]�ɼW)�:�����}1>�^޼�j/>!x=N=�8�=%�x=H��<��g=g��=��p=;��<��<�<���<Lu,=\��=������)=�J=�L=���TH�<k��=�Q=���=��=�n�=.e=b*�=�=���=�t<��<u��<��=T��<flu<[v=�j=c܄<�=���<�?J>��;=$	=���=�j�=I��=���<0�=�ML>�{=V��=���<K��=vHw=�y=�Z<t�<���<1\=��g=�2;=i0���(y=��=F�V<v�<#�=�b?<���<9��=7�<)�l���n<d��=�]a=W�k��'ּjޏ<�+a�\l>��=�x���=�e�����=fQۼ��x;��K=�KS=��T</F�=�ږ=��>��=�X�=(
�=
{�=%�=�\�={׭=\F>�=��=�=�=�L�=�Q�=>q�=(��=t�>4p�= �>���=��>��'=���=~L�=���=���=���=�I�=^�=":=o9�=��x=�5�<��g=d��=صp=���<.�<����<�o,={��=w˥���)=Y=�L=n����<��=�r=���=�=�n�=�a=�/�=3�=G��=�t<���<ϻ�<k�=�-=�0r<W�=9g=_9<�ε=湼<��J>�;=>=>4�=���= ��=�$N<�/�=cJL>��m=&J�=֭�<�̏=��w=�y=��T<�m�<�_<-JS=��g=G�=�b�=���=}�<t�<=hޝ;u�>��<��?<!Q�=qh����n����:yQ[=��	>�(J��܉=�M�<�@���?�=̦=�n="�D=����ba>��<�n|r=nτ��c>
�;ߣ�=���=��>(�=E�=k�=[�=t��=�q�=^̭=�,F>)=0-�=J&�=���=~D�=�t�=��=p�>]��=�C>���=��>��&=��=kQ�=��=��=h��=�I�=