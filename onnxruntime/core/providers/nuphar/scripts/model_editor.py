import argparse
import onnx
import struct
import numpy as np
from enum import Enum

class QuantizeConfig:
    def __init__(self, param_type, reserved_bits, signed):
        assert param_type in [onnx.TensorProto.INT8, onnx.TensorProto.UINT8, onnx.TensorProto.INT16, onnx.TensorProto.UINT16]
        self.type_bits_ = 8 if param_type in [onnx.TensorProto.INT8, onnx.TensorProto.UINT8] else 16
        self.sign_bit_ = 1 if signed else 0
        self.reserved_bits_ = reserved_bits

    def signed(self):
        return self.sign_bit_ == 1

    def usable_bits(self):
        return self.type_bits_ - self.reserved_bits_

    def q_max(self):
        return float((1 << (self.usable_bits() - self.sign_bit_)) - 1)

    def q_min(self):
        return float(-(self.q_max() + 1) if self.signed()  else 0)

    def q_range(self):
        return self.q_max() + 0.5 if self.signed() else float(1 << self.usable_bits())

def num_elements_of_shape(shape):
    s = 1
    for d in shape:
        s = s * d
    return s

class NodeFactory:
    node_count_ = 0

    def __init__(self, main_graph, sub_graph=None):
        self.const_count_ = 0
        self.graph_ = sub_graph if sub_graph else main_graph
        self.main_graph_ = main_graph
        self.name_prefix_ = ''

    def set_prefix(self, prefix):
        self.name_prefix_ = prefix

    def get_initializer(self, name):
        found = [i for i in self.main_graph_.initializer if i.name == name]
        if found:
            t = found[0]
            shape = t.dims
            if t.data_type == onnx.TensorProto.FLOAT:
                value = [v for v in t.float_data] if t.float_data else struct.unpack(str(num_elements_of_shape(shape))+'f', t.raw_data)
                return np.asarray(value).reshape(shape).astype(np.float32)
            elif t.data_type == onnx.TensorProto.INT32:
                value = [v for v in t.int32_data] if t.int32_data else struct.unpack(str(num_elements_of_shape(shape))+'i', t.raw_data)
                return np.asarray(value).reshape(shape).astype(np.int32)
            elif t.data_type == onnx.TensorProto.INT16:
                value = [v for v in t.int32_data] if t.int32_data else struct.unpack(str(num_elements_of_shape(shape))+'h', t.raw_data)
                return np.asarray(value).reshape(shape).astype(np.int16)
            elif t.data_type == onnx.TensorProto.INT8:
                value = [v for v in t.int32_data] if t.int32_data else struct.unpack(str(num_elements_of_shape(shape))+'b', t.raw_data)
                return np.asarray(value).reshape(shape).astype(np.int8)
            elif t.data_type == onnx.TensorProto.UINT8:
                value = [v for v in t.int32_data] if t.int32_data else struct.unpack(str(num_elements_of_shape(shape))+'B', t.raw_data)
                return np.asarray(value).reshape(shape).astype(np.uint8)
            else:
                raise NotImplementedError("unhandled initializer type " + str(t.data_type))
        else:
            return None

    def remove_initializer(self, name):
        initializer = [i for i in self.main_graph_.initializer if i.name == name]
        assert initializer
        self.main_graph_.initializer.remove(initializer[0])
        initializer_in_input = [i for i in self.main_graph_.input if i.name == name]
        if initializer_in_input:
            self.main_graph_.input.remove(initializer_in_input[0])

    def make_attribute(self, node, attr_name, attr_value):
        attr_proto = node.attribute.add()
        attr_proto.name = attr_name
        if type(attr_value) == int:
            attr_proto.type = onnx.AttributeProto.INT
            attr_proto.i = attr_value
        elif type(attr_value) == float:
            attr_proto.type = onnx.AttributeProto.FLOAT
            attr_proto.f = attr_value
        elif type(attr_value) == list and type(attr_value[0]) == int:
            attr_proto.type = onnx.AttributeProto.INTS
            for v in attr_value:
                attr_proto.ints.append(v)
        else:
            raise NotImplementedError("unknown type")

    class ValueInfoType(Enum):
        input = 1
        output = 2
        initializer = 3

    def make_value_info(self, node_or_name, data_type, shape=None, usage=None):
        if usage == NodeFactory.ValueInfoType.input:
            value_info = self.graph_.input.add()
        elif usage == NodeFactory.ValueInfoType.output:
            value_info = self.graph_.output.add()
        elif usage == NodeFactory.ValueInfoType.initializer:
		    # initializer always stay in main_graph as input
            value_info = self.main_graph_.input.add()
        elif not usage:
            value_info = self.graph_.value_info.add()
        else:
            raise NotImplementedError("unknown usage")

        if type(node_or_name) == str:
            value_info.name = node_or_name
        else:
            assert len(node_or_name.output) == 1
            value_info.name = node_or_name.output[0]

        value_info.type.tensor_type.elem_type = data_type
        if shape:
            for d in shape:
                dim = value_info.type.tensor_type.shape.dim.add()
                if type(d) == str:
                    dim.dim_param = d
                elif type(d) == int:
                    dim.dim_value = d
                    
    def make_tensor_proto(name, ndarray, tp=None):
        if not tp:
            tp = onnx.TensorProto()

        type_map = {np.float32 : onnx.TensorProto.FLOAT,
                    np.float   : onnx.TensorProto.FLOAT,
                    np.int8    : onnx.TensorProto.INT8,
                    np.uint8   : onnx.TensorProto.UINT8,
                    np.int16   : onnx.TensorProto.INT16,
                    np.int32   : onnx.TensorProto.INT32,
                    np.int64   : onnx.TensorProto.INT64}
   
        tp.name = name
        found_dt = False
        for np_dt, onnx_dt in type_map.items():
            if ndarray.dtype == np_dt:
                tp.data_type = onnx_dt
                tp.raw_data = ndarray.tobytes()
                found_dt = True

        if not found_dt:
            raise NotImplementedError("unsupported numpy type " + str(ndarray.dtype))

        for d in ndarray.shape:
            tp.dims.append(d)
        return tp

    def make_initializer(self, ndarray, name='', add_value_info=True):
        # initializers are stored only in main graph
        new_initializer = self.main_graph_.initializer.add()
        new_name = name
        if len(new_name) == 0:
            new_name = self.name_prefix_ + '_Const_' + str(self.const_count_)
            self.const_count_ = self.const_count_ + 1
        NodeFactory.make_tensor_proto(new_name, ndarray, new_initializer)
        if add_value_info:
            self.make_value_info(new_initializer.name, new_initializer.data_type, ndarray.shape, usage=NodeFactory.ValueInfoType.initializer)
        return new_initializer

    def make_node(self, op_type, inputs, attributes = None, output_names=None, node=None):
        if type(inputs) != list:
            inputs = [inputs]
        if output_names and type(output_names) != list:
            output_names = [output_names]
        input_names = []
        for i in inputs:
            if type(i) == onnx.NodeProto:
                input_names.append(i.name)
            elif type(i) == str:
                input_names.append(i)
            elif type(i) == np.ndarray:
                new_initializer = self.make_initializer(i)
                input_names.append(new_initializer.name)

        if not node:
            node = self.graph_.node.add()

        node.name = self.name_prefix_ + op_type + '_' + str(NodeFactory.node_count_)
        NodeFactory.node_count_ = NodeFactory.node_count_ + 1
        node.op_type = op_type

        for i in input_names:
            node.input.append(i)

        if output_names:
            for o in output_names:
                node.output.append(o)
        else:
            node.output.append(node.name)

        if attributes:
            for n,v in enumerate(attributes):
                self.make_attribute(node, v, attributes[v])
        return node

    # change initializers to raw_data to make model size smaller and loading faster
    def modify_initializer_to_raw_data(self):
        aas = {}
        for initializer in self.main_graph_.initializer:
            aas[initializer.name] = self.get_initializer(initializer.name)
        while len(self.main_graph_.initializer) > 0:
            self.main_graph_.initializer.pop()
        for key,value in aas.items():
            self.make_initializer(value, key, add_value_info=False)

def convert_qmatmul_symm(node, out_main_graph, out_graph):
    nf = NodeFactory(out_main_graph, out_graph)
    param = node.input[0]
    step = node.input[1]
    X = node.input[2]
    Y = node.output[0]
    param_type = [i for i in out_main_graph.initializer if i.name == param][0].data_type
    assert node.attribute[0].name == 'ReservedBit'
    reserved_bits = node.attribute[0].i
    qcfg = QuantizeConfig(param_type, reserved_bits, signed=True)
    nf.set_prefix(node.name)

    # Add quantization for X
    qmax_range = np.asarray(qcfg.q_range()).astype(np.float32)
    qmax_range_plus_one = np.asarray(qcfg.q_range() + 1).astype(np.float32)
    abs_max_X = nf.make_node('ReduceMax', nf.make_node('Abs', X), {'axes':[-1]})
    inv_abs_max_X = nf.make_node('Div', [np.asarray(1).astype(np.float32), abs_max_X])
    scale_X = nf.make_node('Div', [abs_max_X, qmax_range])
    Q_Xf = nf.make_node('Mul', [nf.make_node('Mul', [X, inv_abs_max_X]), qmax_range])
    Q_Xf = nf.make_node('Add', [Q_Xf, qmax_range_plus_one])
    Q_X = nf.make_node('Sub', [nf.make_node('Cast', Q_Xf, {'to': int(onnx.TensorProto.INT32)}), qmax_range_plus_one.astype(np.int32)])
    Q_X = nf.make_node('Cast', Q_X, {'to':int(onnx.TensorProto.FLOAT)})
    Q_X = nf.make_node('Clip', Q_X, {'max':qcfg.q_max(), 'min':qcfg.q_min()})
    Q_X = nf.make_node('Cast', Q_X, {'to':int(onnx.TensorProto.INT16)})

    # MatMulInteger
    Q_Y = nf.make_node('MatMulInteger', [Q_X, param])
    nf.make_value_info(Q_Y, data_type=onnx.TensorProto.INT32)

    # Dequantize
    nf.make_node('Mul',
                      [nf.make_node('Mul', [step, scale_X]),
                       nf.make_node('Cast', Q_Y, {'to': int(onnx.TensorProto.FLOAT)})],
                      output_names=Y)


def convert_qmatmul_asymm(node, out_main_graph, out_graph):
    nf = NodeFactory(out_main_graph, out_graph)
    param = node.input[0]
    base = node.input[1]
    step = node.input[2]
    param_rowsum = node.input[3]
    X = node.input[4]
    Y = node.output[0]
    # initializers are stored in main_graph only
    param_type = [i for i in out_main_graph.initializer if i.name == param][0].data_type
    reserved_bits = node.attribute[0].i
    qcfg = QuantizeConfig(param_type, reserved_bits, signed=False)
    nf.set_prefix(node.name)
    X_value_info = [vi for vi in out_graph.value_info if vi.name == X]
    if not X_value_info:
        X_value_info = [vi for vi in out_graph.input if vi.name == X]
    assert len(X_value_info) == 1
    X_value_info = X_value_info[0]
    X_shape = X_value_info.type.tensor_type.shape.dim
    input_dim = X_shape[len(X_shape) - 1].dim_value

    # Add quantization for X
    reduce_max_X = nf.make_node('ReduceMax', X, {'axes':[-1]}) # keepdims = 1
    bias_X = nf.make_node('ReduceMin', X, {'axes':[-1]})
    delta_X = nf.make_node('Sub', [reduce_max_X, bias_X])
    scale_X = nf.make_node('Div', [delta_X, np.asarray(qcfg.q_range()).astype(np.float32)])
    norm_X = nf.make_node('Div', [nf.make_node('Sub', [X, bias_X]), delta_X])
    Q_Xf = nf.make_node('Mul', [norm_X, np.asarray(qcfg.q_range()).astype(np.float32)])
    Q_Xf = nf.make_node('Add', [Q_Xf, np.asarray(0.5).astype(np.float32)])
    Q_Xf = nf.make_node('Floor', Q_Xf)
    Q_Xf = nf.make_node('Clip', Q_Xf, {'max':qcfg.q_max(), 'min':qcfg.q_min()})
    Q_X = nf.make_node('Cast', Q_Xf, {'to':int(onnx.TensorProto.UINT8)})
    Q_X_sum = nf.make_node('ReduceSum', Q_Xf, {'axes':[-1]})

    # MatMulInteger
    Q_Y = nf.make_node('MatMulInteger', [Q_X, param])
    nf.make_value_info(Q_Y, data_type=onnx.TensorProto.INT32)

    # Dequantize
    o0 = nf.make_node('Mul', [nf.make_node('Mul', [step, scale_X]),
                              nf.make_node('Cast', Q_Y, {'to': int(onnx.TensorProto.FLOAT)})])
    o1 = nf.make_node('Mul', [nf.make_node('Mul', [step, bias_X]), param_rowsum])
    o2 = nf.make_node('Mul', [base, nf.make_node('Mul', [scale_X, Q_X_sum])])
    o3 = nf.make_node('Mul', [base, nf.make_node('Mul', [bias_X, np.asarray(float(input_dim)).astype(np.float32)])])

    nf.make_node('Add', [nf.make_node('Add', [nf.make_node('Add', [o3, o2]), o1]), o0], output_names=Y)

def convert_qmatmul(n, out_main_graph, out_graph):
    if n.op_type == 'QMatMulAsymmetric':
        convert_qmatmul_asymm(n, out_main_graph, out_graph)
        return True
    elif n.op_type == 'QMatMulSymmetric':
        convert_qmatmul_symm(n, out_main_graph, out_graph)
        return True
    else:
        return False

def convert_qmatmul_model(input_model, output_model):
    in_mp = onnx.ModelProto()
    ff = open(input_model, 'rb')
    ss = ff.read()
    ff.close()
    in_mp.ParseFromString(ss)

    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    for opset in out_mp.opset_import:
        if opset.domain == '' or opset.domain == 'onnx':
            opset.version = 10
    out_mp.graph.ClearField('node')
    
    nf = NodeFactory(out_mp.graph)
    nf.modify_initializer_to_raw_data()

    for n in in_mp.graph.node:
        if n.op_type == 'Scan':
            out_n = out_mp.graph.node.add()
            out_n.CopyFrom(n)
            body = [attr for attr in n.attribute if attr.name == 'body'][0]
            out_body = [attr for attr in out_n.attribute if attr.name == 'body'][0]
            out_body.g.ClearField('node')
            for sn in body.g.node:
                if not convert_qmatmul(sn, out_mp.graph, out_body.g):
                    out_sn = out_body.g.node.add()
                    out_sn.CopyFrom(sn)
        else:
            if not convert_qmatmul(n, out_mp.graph, out_mp.graph):
                out_n = out_mp.graph.node.add()
                out_n.CopyFrom(n)

    ff = open(output_model, 'wb')
    ff.write(out_mp.SerializeToString())
    ff.close()

def create_simple_mnist():
    import cntk as C
    cntk_model = C.load_model(r'X:\data\CNTK\test_simpleMNIST\model.onnx.model')
    cntk_params = cntk_model.parameters
    cntk_consts = cntk_model.constants
    mp = onnx.ModelProto()
    opset = mp.opset_import.add()
    opset.version = 9
    opset.domain = "ai.onnx"
    graph = mp.graph
    nf = NodeFactory(graph)
    nf.make_value_info("Input3", onnx.TensorProto.FLOAT, ('sequence', 'batch', 784), usage=NodeFactory.ValueInfoType.input)
    scaled_input = nf.make_node('Mul', ['Input3', cntk_consts[0].value])
    proj = nf.make_node('MatMul', [scaled_input, cntk_params[2].value])
    proj_with_bias = nf.make_node('Add', [proj, cntk_params[3].value])
    activated = nf.make_node('Relu', proj_with_bias)
    proj2 = nf.make_node('MatMul', [activated, cntk_params[0].value])
    proj2_bias = nf.make_node('Add', [proj2, cntk_params[1].value])
    softmax = nf.make_node('Softmax', [proj2_bias], {'axis':2}, 'Softmax180_Output_0')
    nf.make_value_info(softmax, onnx.TensorProto.FLOAT, ('sequence', 'batch', 10), usage=NodeFactory.ValueInfoType.output)
    ff = open(r'X:\model.onnx', 'wb')
    ff.write(mp.SerializeToString())
    ff.close()
    return None

def add_batch():
    mp = onnx.ModelProto()
    with open(r'X:\data\CNTK\test_ScanLSTM\model.onnx', 'rb') as ff:
        mp.ParseFromString(ff.read())

    main_graph = mp.graph
    nf = NodeFactory(main_graph)
    for n in main_graph.node:
        if n.op_type == 'Scan':
            nf.set_prefix(n.name)
            body = [attr for attr in n.attribute if attr.name == 'body'][0]
            for arr in [body.g.value_info, body.g.input, body.g.output]:
                for vi in arr:
                   vi.type.tensor_type.shape.dim[0].dim_param = 'Batch'

    for arr in [main_graph.value_info, main_graph.input, main_graph.output]:
        for vi in arr:
            if vi.type.tensor_type.shape:
                if len(vi.type.tensor_type.shape.dim) > 1:
                    if vi.type.tensor_type.shape.dim[0].dim_param == 'Sequence':
                        vi.type.tensor_type.shape.dim[1].dim_param = 'Batch'
                    elif ('PastValue' in vi.name or 'FutureValue' in vi.name):
                        vi.type.tensor_type.shape.dim[0].dim_param = 'Batch'

    for i in main_graph.initializer:
        if 'PastValue' in i.name or 'FutureValue' in i.name:
            print(i.name)

    with open(r'X:\model.onnx', 'wb') as ff:
        ff.write(mp.SerializeToString())

def create_rand_qmatmul():
    seq_len = 1
    input_dim = 64
    embed_dim = 128
    q_m = np.random.randint(low=0, high=16383, size=(input_dim,embed_dim), dtype=np.int16)
    q_s = np.random.rand(embed_dim).astype(np.float32)
    mp = onnx.ModelProto()
    opset = mp.opset_import.add()
    opset.version = 9
    opset.domain = "ai.onnx"
    graph = mp.graph
    nf = NodeFactory(graph)
    nf.make_value_info("Input", onnx.TensorProto.FLOAT, ('sequence', input_dim), usage=NodeFactory.ValueInfoType.input)
    qmatmul = nf.make_node('QMatMulSymmetric', [q_m, q_s, 'Input'], {'ReservedBit':1}, output_names=['Output'])
    nf.make_value_info('Output', onnx.TensorProto.FLOAT, ('sequence', embed_dim), usage=NodeFactory.ValueInfoType.output)
    
    with open(r'test_qq\model.onnx', 'wb') as ff:
      ff.write(mp.SerializeToString())
    
    with open(r'test_qq\test_data_set_0\input_0.pb', 'wb') as ff:
      ff.write(NodeFactory.make_tensor_proto('Input', np.random.rand(seq_len,input_dim).astype(np.float32)).SerializeToString())

    with open(r'test_qq\test_data_set_0\output_0.pb', 'wb') as ff:
      ff.write(NodeFactory.make_tensor_proto('Output', np.random.rand(seq_len,embed_dim).astype(np.float32)).SerializeToString())
      
def edit_LC_BLSTM(input_model, output_model, lc_nr=0):
    mp = onnx.ModelProto()
    with open(input_model, 'rb') as ff:
        mp.ParseFromString(ff.read())

    main_graph = mp.graph
    nf = NodeFactory(main_graph)
    # add additional scalar initializers
    nf.make_initializer(np.asarray([lc_nr]).astype(np.int32), 'FutureContextLength')
    nf.make_initializer(np.asarray([1]).astype(np.int32), '__OneInt32')
    nf.make_initializer(np.asarray([1]).astype(np.float32), '__OneFloat')
    nf.make_initializer(np.asarray([0]).astype(np.int32), '__ZeroInt32')

    nf.modify_initializer_to_raw_data()

    # insert node to front compute sequence length
    # note to keep node order, new nodes need to be inserted before the first forward Scan
    all_nodes = []
    has_LC_Position = False
    for node in main_graph.node:
        if not has_LC_Position and node.op_type == 'Scan' and [att.ints[0] for att in node.attribute if att.name == 'scan_input_directions'][0] == 0:
            num_scan_inputs = [att.i for att in node.attribute if att.name == 'num_scan_inputs'][0]
            assert num_scan_inputs == 1
            num_initial_states = len(node.input) - num_scan_inputs
            scan_seq_input = node.input[num_initial_states]

            seq_input_shape_node = onnx.NodeProto()
            seq_len_node = onnx.NodeProto()
            cast_seq_len_node = onnx.NodeProto()
            seq_len_sub_1_node = onnx.NodeProto()
            seq_len_sub_1_sub_fcl_node = onnx.NodeProto()
            nf.make_node('Shape', scan_seq_input, node=seq_input_shape_node)
            nf.make_node('Slice', seq_input_shape_node, {'axes':[0],'starts':[0],'ends':[1]}, node=seq_len_node)
            nf.make_node('Cast', seq_len_node, {'to':onnx.TensorProto.INT32}, node=cast_seq_len_node)
            nf.make_node('Sub', [cast_seq_len_node, '__OneInt32'], node=seq_len_sub_1_node)
            nf.make_node('Sub', [seq_len_sub_1_node, 'FutureContextLength'], output_names='LC_Position', node=seq_len_sub_1_sub_fcl_node)
            all_nodes.append(seq_input_shape_node)
            all_nodes.append(seq_len_node)
            all_nodes.append(cast_seq_len_node)
            all_nodes.append(seq_len_sub_1_node)
            all_nodes.append(seq_len_sub_1_sub_fcl_node)
            has_LC_Position = True
        all_nodes.append(node)

    if has_LC_Position:
        # replace main_graph.node with all_nodes
        while len(main_graph.node) > 0:
            main_graph.node.pop()
        main_graph.node.MergeFrom(all_nodes)

    for node in main_graph.node:
        if node.op_type == 'Scan' and [att.ints[0] for att in node.attribute if att.name == 'scan_input_directions'][0] == 0:
            num_scan_inputs = [att.i for att in node.attribute if att.name == 'num_scan_inputs'][0]
            assert num_scan_inputs == 1
            num_initial_states = len(node.input) - num_scan_inputs
            # For forward, add additional input/output_state for LC_H, LC_C, count
            scan_seq_input = node.input.pop()
            scan_seq_output = node.output.pop()
            for ii in range(num_initial_states):
                # use the same input state initializer for LC_*
                node.input.append(node.input[ii])
                # add main_graph output for LC_*
                lc_output_state_name = 'LC_' + node.output[ii]
                node.output.append(lc_output_state_name)
                new_output = main_graph.output.add()
                new_output.CopyFrom([vi for vi in main_graph.output if vi.name == node.output[ii]][0])
                new_output.name = lc_output_state_name
            # count input_state start with LC_Position
            node.input.append('LC_Position')
            node.input.append(scan_seq_input)
            # count output_state
            count_output_name = 'LC_' + node.name + '_count_Output'
            node.output.append(count_output_name)
            nf.make_value_info(count_output_name, onnx.TensorProto.INT32, (1,), usage=NodeFactory.ValueInfoType.output)
            node.output.append(scan_seq_output)
            # add input/output in subgraph
            body = [attr for attr in node.attribute if attr.name == 'body'][0]
            # pop and save the sequence input/output in the subgraph
            subgraph_seq_input = body.g.input.pop()
            subgraph_seq_output = body.g.output.pop()
            # add LC_count_subgraph input/output
            nf_sub = NodeFactory(body.g)
            nf_sub.set_prefix(node.name)
            subgraph_count_input_name = 'LC_' + node.name + '_count_subgraph'
            # add latency control code
            at_lc_position = nf_sub.make_node('Cast', [nf_sub.make_node('Equal', [subgraph_count_input_name, '__ZeroInt32'])], {'to':onnx.TensorProto.FLOAT})
            nf_sub.make_node('Sub', [subgraph_count_input_name, '__OneInt32'], output_names=count_output_name)
            for ii in range(num_initial_states):
                # add subgraph input for LC*
                subgraph_LC_input_name = 'LC_' + body.g.input[ii].name
                lc_output_state_name = 'LC_' + node.output[ii]
                vi = body.g.input.add()
                vi.CopyFrom(body.g.input[ii])
                vi.name = subgraph_LC_input_name
                vi = body.g.output.add()
                vi.CopyFrom(body.g.output[ii])
                vi.name = lc_output_state_name
                # LC_output_state = at_lc_position * output_state + (1 - at_lc_position) * LC_input_state
                nf_sub.make_node('Add',
                                 [nf_sub.make_node('Mul', [at_lc_position, body.g.output[ii].name]),
                                  nf_sub.make_node('Mul', [nf_sub.make_node('Sub', ['__OneFloat', at_lc_position]), subgraph_LC_input_name])],
                                 output_names=lc_output_state_name)

            # add subgraph input/output for count
            nf_sub.make_value_info(subgraph_count_input_name, onnx.TensorProto.INT32, shape=(1,), usage=NodeFactory.ValueInfoType.input)
            nf_sub.make_value_info(count_output_name, onnx.TensorProto.INT32, shape=(1,), usage=NodeFactory.ValueInfoType.output)
            # add sequence input/output back to subgraph's input/output
            body.g.input.add().CopyFrom(subgraph_seq_input)
            body.g.output.add().CopyFrom(subgraph_seq_output)

        # change Softmax + Log to LogSoftmax
        if node.op_type == 'Softmax':
            node_output = node.output[0]
            decendent_nodes = [n for n in main_graph.node if any([ni == node_output for ni in n.input])]
            assert len(decendent_nodes) == 1 and decendent_nodes[0].op_type == 'Log'
            new_node_output = decendent_nodes[0].output[0]
            main_graph.node.remove(decendent_nodes[0])
            node.op_type = 'LogSoftmax'
            node.output.pop()
            node.output.append(new_node_output)

    with open(output_model, 'wb') as ff:
        ff.write(mp.SerializeToString())

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', help='The modification mode', choices=['lc', 'imatmul'], default='imatmul')
  parser.add_argument('--input_model', help='The input model file', default=None)
  parser.add_argument('--output_model', help='The input model file', default=None)
  parser.add_argument('--nr', help='The input model file', default=0)
  return parser.parse_args()

if __name__ == '__main__':
    #create_simple_mnist()
    #add_batch()
    #create_rand_qmatmul()
    #edit_LC_BLSTM(r'X:\data\CNTK\test_ScanLSTM\model.onnx', r'X:\data\CNTK\test_ScanLSTM_lc\model.onnx', lc_nr=1)
    #edit_LC_BLSTM(r'X:\data\CNTK\test_LCBLSTM_8bit\model.onnx', r'X:\data\CNTK\test_LCBLSTM_8bit\model_lc.onnx', lc_nr=20)
    args = parse_arguments()
    print('input model: ' + args.input_model)
    print('output model ' + args.output_model)
    if args.mode == 'lc':
        print('Adding latency control...')
        edit_LC_BLSTM(args.input_model, args.output_model, args.nr)
    elif args.mode == 'imatmul':
        print('Convert QMatMul* to MatMulInteger...')
        convert_qmatmul_model(args.input_model, args.output_model)
    else:
        raise NotImplementedError('Unknown mode')
    print('Done!')