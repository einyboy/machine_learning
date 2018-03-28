# 以下定义整个 CapsNet 的架构与正向传播过程
class CapsNet():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                # 获取一个批量的训练数据
                self.X, self.Y = get_batch_data()

                self.build_arch()
                self.loss()

                # t_vars = tf.trainable_variables()
                self.optimizer = tf.train.AdamOptimizer()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)  # var_list=t_vars)
            else:
                self.X = tf.placeholder(tf.float32,
                                        shape=(batch_size, 28, 28, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')

    # CapsNet 类中的build_arch方法能构建整个网络的架构
    def build_arch(self):
        # 以下构建第一个常规卷积层
        with tf.variable_scope('Conv1_layer'):
            # 第一个卷积层的输出张量为： [batch_size, 20, 20, 256]
            # 以下卷积输入图像X,采用256个9×9的卷积核，步幅为1，且不使用
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            assert conv1.get_shape() == [batch_size, 20, 20, 256]

        # 以下是原论文中PrimaryCaps层的构建过程，该层的输出维度为 [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            # 调用前面定义的CapLayer函数构建第二个卷积层，该过程相当于执行八次常规卷积，
            # 然后将各对应位置的元素组合成一个长度为8的向量，这八次常规卷积都是采用32个9×9的卷积核、步幅为2
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            assert caps1.get_shape() == [batch_size, 1152, 8, 1]

        # 以下构建 DigitCaps 层, 该层返回的张量维度为 [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            # DigitCaps是最后一层，它返回对应10个类别的向量（每个有16个元素），该层的构建带有Routing过程
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # 以下构建论文图2中的解码结构，即由16维向量重构出对应类别的整个图像
        # 除了特定的 Capsule 输出向量，我们需要蒙住其它所有的输出向量
        with tf.variable_scope('Masking'):

            #mask_with_y是否用真实标签蒙住目标Capsule
            mask_with_y=True
            if mask_with_y:
                self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)
            

        # 通过3个全连接层重构MNIST图像，这三个全连接层的神经元数分别为512、1024、784
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    # 定义 CapsNet 的损失函数，损失函数一共分为衡量 CapsNet准确度的Margin loss
    # 和衡量重构图像准确度的 Reconstruction loss
    def loss(self):
        # 以下先定义重构损失，因为DigitCaps的输出向量长度就为某类别的概率，因此可以借助计算向量长度计算损失
        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - m_minus))
        assert max_l.get_shape() == [batch_size, 10, 1, 1]

        # 将当前的维度[batch_size, 10, 1, 1] 转换为10个数字类别的one-hot编码 [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(batch_size, -1))
        max_r = tf.reshape(max_r, shape=(batch_size, -1))

        # 计算 T_c: [batch_size, 10]，其为分类的指示函数
        # 若令T_c = Y,那么对应元素相乘就是有类别相同才会有非零输出值，T_c 和 Y 都为One-hot编码
        T_c = self.Y
        # [batch_size, 10], 对应元素相乘并构建最后的Margin loss 函数
        L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 以下构建reconstruction loss函数
        # 这一过程的损失函数通过计算FC Sigmoid层的输出像素点与原始图像像素点间的欧几里德距离而构建
        orgin = tf.reshape(self.X, shape=(batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 构建总损失函数，Hinton论文将reconstruction loss乘上0.0005
        # 以使它不会主导训练过程中的Margin loss
        self.total_loss = self.margin_loss + 0.0005 * self.reconstruction_err

        # 以下输出TensorBoard
        tf.summary.scalar('margin_loss', self.margin_loss)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_err)
        tf.summary.scalar('total_loss', self.total_loss)
        recon_img = tf.reshape(self.decoded, shape=(batch_size, 28, 28, 1))
        tf.summary.image('reconstruction_img', recon_img)
        self.merged_sum = tf.summary.merge_all()