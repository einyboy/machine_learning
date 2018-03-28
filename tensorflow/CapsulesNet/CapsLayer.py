#通过定义类和对象的方式定义Capssule层级
class CapsLayer(object):
    ''' Capsule layer 类别参数有：
    Args:
        input: 一个4维张量
        num_outputs: 当前层的Capsule单元数量
        vec_len: 一个Capsule输出向量的长度
        layer_type: 选择'FC' 或 "CONV", 以确定是用全连接层还是卷积层
        with_routing: 当前Capsule是否从较低层级中Routing而得出输出向量

    Returns:
        一个四维张量
    '''
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        '''
        当“Layer_type”选择的是“CONV”，我们将使用 'kernel_size' 和 'stride'
        '''

        # 开始构建卷积层
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            # PrimaryCaps层没有Routing过程
            if not self.with_routing:
                # 卷积层为 PrimaryCaps 层（CapsNet第二层）, 并将第一层卷积的输出张量作为输入。
                # 输入张量的维度为： [batch_size, 20, 20, 256]
                assert input.get_shape() == [batch_size, 20, 20, 256]

                #从CapsNet输出向量的每一个分量开始执行卷积，每个分量上执行带32个卷积核的9×9标准卷积
                capsules = []
                for i in range(self.vec_len):
                    # 所有Capsule的一个分量，其维度为: [batch_size, 6, 6, 32]，即6×6×1×32
                    with tf.variable_scope('ConvUnit_' + str(i)):
                        caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                          self.kernel_size, self.stride,
                                                          padding="VALID")

                        # 将一般卷积的结果张量拉平，并为添加到列表中
                        caps_i = tf.reshape(caps_i, shape=(batch_size, -1, 1, 1))
                        capsules.append(caps_i)

                # 为将卷积后张量各个分量合并为向量做准备
                assert capsules[0].get_shape() == [batch_size, 1152, 1, 1]

                # 合并为PrimaryCaps的输出张量，即6×6×32个长度为8的向量，合并后的维度为 [batch_size, 1152, 8, 1]
                capsules = tf.concat(capsules, axis=2)
                # 将每个Capsule 向量投入非线性函数squash进行缩放与激活
                capsules = squash(capsules)
                assert capsules.get_shape() == [batch_size, 1152, 8, 1]
                return(capsules)

        if self.layer_type == 'FC':

            # DigitCaps 带有Routing过程
            if self.with_routing:
                # CapsNet 的第三层 DigitCaps 层是一个全连接网络
                # 将输入张量重建为 [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(batch_size, -1, 1, input.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    # 初始化b_IJ的值为零，且维度满足: [1, 1, num_caps_l, num_caps_l_plus_1, 1]
                    b_IJ = tf.constant(np.zeros([1, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    # 使用定义的Routing过程计算权值更新与s_j
                    capsules = routing(self.input, b_IJ)
                    #将s_j投入 squeeze 函数以得出 DigitCaps 层的输出向量
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)