import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable
import math

# Embedding层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, emb_type):
        """
        类的初始化函数
        d_model：指词嵌入的维度
        vocab:指词表的大小
        """
        super(Embeddings, self).__init__()
        #之后就是调用nn中的预定义层Embedding，获得一个词嵌入对象self.lut
        self.lut = nn.Linear(21, d_model)
        self.emb = nn.Linear(20,d_model)
        self.atch = nn.Linear(5,d_model)
        #最后就是将d_model传入类中
        self.d_model = d_model
        self.emb_type = emb_type
    def forward(self, x):
        """
        Embedding层的前向传播逻辑
        参数x：这里代表输入给模型的单词文本通过词表映射后的one-hot向量
        将x传给self.lut并与根号下self.d_model相乘作为结果返回
        """
        if self.emb_type == 'onehot':
            embedds = self.lut(x)
        elif self.emb_type == 'BLOSUM62':
            embedds = self.emb(x) * math.sqrt(self.d_model)
        else:
            embedds = self.atch(x) * math.sqrt(self.d_model)
        return embedds


# Positioal Embedding层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=20):
        """
        位置编码器类的初始化函数

        共有三个参数，分别是
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
        # 这样计算是为了避免中间的数值计算结果超出float的范围，
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term).type(torch.FloatTensor)
        pe[:, 1::2] = torch.cos(position * div_term).type(torch.FloatTensor)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# 定义一个clones函数，来更方便的将某个结构复制若干份
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Encoder的结构：Attention+FFN
class Encoder(nn.Module):
    """
    Encoder
    The encoder is composed of a stack of N=6 identical layers.
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 调用时会将编码器层传进来，我们简单克隆N分，叠加在一起，组成完整的Encoder
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    # 在我们的任务重没有mask，可以直接将mask设置为None，Mask_Sequence也不需要
    def forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# 子层连接：Attention+Add+Norm/FFN+Add+Norm
# 对原文做了改进，增加了一个丢弃率，成了 X->Dropout(X)->Norm(Dropout(X))+X
class SublayerConnection(nn.Module):
    """
    实现子层连接结构的类
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        # 原paper的方案
        #sublayer_out = sublayer(x)
        #x_norm = self.norm(x + self.dropout(sublayer_out))

        # 稍加调整的版本
        sublayer_out = sublayer(x)
        sublayer_out = self.dropout(sublayer_out)
        x_norm = x + self.norm(sublayer_out)
        return x_norm



'''
class CNN(nn.Module):
    def __init__(self, d_model, dropout):
        # 使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(CNN, self).__init__()
        # 1 input image channel ,6 output channels,5x5 square convolution kernel
        self.d_model = d_model
        self.conv1 = nn.Conv2d(1, 2, (3, 5), stride=(1, 1), padding=(1, 2))
        self.conv2 = nn.Conv2d(2, 3, (3, 5), stride=(1, 1), padding=(1, 2))
        self.conv3 = nn.ConvTranspose2d(3, 2, (3, 5), stride=(1, 1), padding=(1, 2))
        self.conv4 = nn.ConvTranspose2d(2, 1, (3, 5), stride=(1, 1), padding=(1, 2))
        h_d = int(d_model / 2)
        self.fc1 = nn.Linear(d_model, h_d)
        self.fc2 = nn.Linear(h_d, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        # x是网络的输入，然后将x前向传播，最后得到输出
        # 修改x的维度
        x = x.view(x.size()[0], 1, 20, self.d_model)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = x.view(x.size()[0], 20, self.d_model)
        return x


'''

class CNN(nn.Module):
    def __init__(self, d_model, dropout):
        # 使用super()方法调用基类的构造器，即nn.Module.__init__(self)
        super(CNN, self).__init__()
        # 1 input image channel ,6 output channels,5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        # x是网络的输入，然后将x前向传播，最后得到输出
        # 修改x的维度
        x = x.view(x.size()[0], 1, 20, self.d_model)
        #x = self.conv2(self.conv1(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = x.view(x.size()[0], 20, self.d_model)
        # print('x.size:',x.size())
        return x


# 每个Encoder结构由N个EncoderLayer组成
# 在本文的任务重，Encoder包含1个或者2个EncoderLayer，多了没必要，还增加复杂度
class EncoderModule(nn.Module):
    "EncoderLayer is made up of two sublayer: self-attn and feed forward"
    def __init__(self, cnn, size, self_attn, feed_forward, dropout, d_model):
        super(EncoderModule, self).__init__()
        self.cnn = cnn
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size   # embedding's dimention of model, 默认512
        self.fc = nn.Linear(2*d_model, d_model)

    def forward(self, x):
        # cnn
        y = self.cnn(x)
        # attention sub layer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        # feed forward sub layer
        x = self.sublayer[1](x, self.feed_forward)
        # cnn
        x = torch.cat((y,x),-1)
        x = self.fc(x)
        return x

# 每个Encoder结构由N个EncoderLayer组成
# 在本文的任务重，Encoder包含1个或者2个EncoderLayer，多了没必要，还增加复杂度
class EncoderLayer(nn.Module):
    "EncoderLayer is made up of two sublayer: self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size   # embedding's dimention of model, 默认512

    def forward(self, x):
        # attention sub layer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        # feed forward sub layer
        z = self.sublayer[1](x, self.feed_forward)
        return z


# Attention层
def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    # 首先取query的最后一维的大小，对应词嵌入维度
    d_k = query.size(-1)
    # 按照注意力公式，将query与key的转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)


    # 对scores的最后一维进行softmax操作，使用F.softmax方法，这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim=-1)

    # 之后判断是否使用dropout进行随机置0
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后，根据公式将p_attn与value张量相乘获得最终的query注意力表示，同时返回注意力张量
    # 所以返回的是注意力表示Value_att和各个位置的注意力权重
    return torch.matmul(p_attn, value), p_attn


# Multi-Head Attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # 在类的初始化时，会传入三个参数，h代表头数，d_model代表词嵌入的维度，dropout代表进行dropout操作时置0比率，默认是0.1
        super(MultiHeadedAttention, self).__init__()
        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，这是因为我们之后要给每个头分配等量的词特征，
        # 也就是qkv_dim = embedding_dim/head个
        assert d_model % h == 0
        # 得到每个头获得的分割词向量维度d_k
        self.d_k = d_model // h
        # 传入头数h
        self.h = h

        # 创建linear层，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用，
        # 为什么是四个呢，这是因为在多头注意力中，Q,K,V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        # 前向逻辑函数，它输入参数有四个，前三个就是注意力机制需要的Q,K,V，最后一个是注意力机制中可能需要的mask掩码张量，默认是None
        '''
        if mask is not None:
            # Same mask applied to all h heads.
            #使用unsqueeze扩展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)

        '''
        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 首先利用zip将输入QKV与三个线性层组到一起，然后利用for循环，将输入QKV分别传到线性层中，做完线性变换后，开始为每个头分割输入，
        # 这里使用view方法对线性变换的结构进行维度重塑，多加了一个维度h代表头，这样就意味着每个头可以获得一部分词特征组成的句子，
        # 其中的-1代表自适应维度，计算机会根据这种变换自动计算这里的值，然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，从attention函数中可以看到，
        # 利用的是原始输入的倒数第一和第二维，这样我们就得到了每个头的输入
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # 得到每个头的输入后，接下来就是将他们传入到attention中，这里直接调用我们之前实现的attention函数，
        # 同时也将mask和dropout传入其中，x为得到的注意力表示，self.attn是注意力权重
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量x，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法。
        # contiguous方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用。
        # 下一步就是使用view重塑形状，变成和输入形状相同。
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # 最后使用线性层列表中的最后一个线性变换得到最终的多头注意力结构的输出
        return self.linears[-1](x)

# FFN 前馈网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        #初始化函数有三个输入参数分别是d_model，d_ff，和dropout=0.1，第一个是线性层的输入维度也是第二个线性层的输出维度，
        #因为我们希望输入通过前馈全连接层后输入和输出的维度不变，第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出，
        #最后一个是dropout置0比率。
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #输入参数为x，代表来自上一层的输出，首先经过第一个线性层，然后使用F中的relu函数进行激活，
        #之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# LayerNorm 层归一化
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, feature_size, eps=1e-6):
        #初始化函数有两个参数，一个是feature_size,表示词嵌入的维度,
        #另一个是eps它是一个足够小的数，在规范化公式的分母中出现,防止分母为0，默认是1e-6。
        super(LayerNorm, self).__init__()
        #根据features的形状初始化两个参数张量a2和b2，第一初始化为1张量，也就是里面的元素都是1，第二个初始化为0张量，
        #也就是里面的元素都是0，这两个张量就是规范化层的参数。因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，
        #因此就需要有参数作为调节因子，使其即能满足规范化要求，又能不改变针对目标的表征，
        #最后使用nn.parameter封装，代表他们是模型的参数
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.zeros(feature_size))
        #把eps传到类中
        self.eps = eps

    def forward(self, x):
    #输入参数x代表来自上一层的输出，在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致，
    #接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果。
    #最后对结果乘以我们的缩放参数，即a2,*号代表同型点乘，即对应位置进行乘法操作，加上位移参b2，返回即可
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 使用类Decoder来实现解码器
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        # 初始化函数的参数有两个，第一个就是解码器层layer，第二个是解码器层的个数N
        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化一个规范化层，因为数据走过了所有的解码器层后最后要做规范化处理。
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory):
        # forward函数中的参数有4个，x代表目标数据的嵌入表示，memory是编码器层的输出，
        # source_mask，target_mask代表源数据和目标数据的掩码张量，然后就是对每个层进行循环，
        # 当然这个循环就是变量x通过每一个层的处理，得出最后的结果，再进行一次规范化返回即可。
        for layer in self.layers:
            x = layer(x, memory)
        # print('————————————————————————————————————————————————————————')
        # print('Decoder输出维度:\n')
        # print('self.norm(x).size():',self.norm(x).size())
        # print('————————————————————————————————————————————————————————')
        return self.norm(x)


# 使用DecoderLayer的类实现解码器层
class DecoderModule(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, cnn, size, self_attn, src_attn, feed_forward, dropout):
        # 初始化函数的参数有5个，分别是size，代表词嵌入的维度大小，同时也代表解码器的尺寸，
        # 第二个是self_attn，多头自注意力对象，也就是说这个注意力机制需要Q=K=V，
        # 第三个是src_attn,多头注意力对象，这里Q!=K=V，第四个是前馈全连接层对象，最后就是dropout置0比率
        super(DecoderModule, self).__init__()
        self.cnn = cnn
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 按照结构图使用clones函数克隆三个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.fc = nn.Linear(2 * d_model, d_model)

    def forward(self, x, memory):
        # forward函数中的参数有2个，分别是来自上一层的输入x，来自编码器层的语义存储变量memory，将memory表示成m之后方便使用。
        "Follow Figure 1 (right) for connections."
        m = memory
        # 1、将x传入到cnn中，提取局部信息
        cx = self.cnn(x)
        # 2、将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x，
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        # 3、拼接cnn的结果与atten的结果并调整维度
        x = torch.cat((cx, x), -1)
        x = self.fc(x)
        # 4、接着进入第二个子层，这个子层中常规的注意力机制，q是输入x;k,v是编码层输出memory
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        # 5、最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果，这就是我们的解码器结构
        x = self.sublayer[2](x, self.feed_forward)

        # 连接cnn与attention的结果

        return x


# 使用DecoderLayer的类实现解码器层
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # 初始化函数的参数有5个，分别是size，代表词嵌入的维度大小，同时也代表解码器的尺寸，
        # 第二个是self_attn，多头自注意力对象，也就是说这个注意力机制需要Q=K=V，
        # 第三个是src_attn,多头注意力对象，这里Q!=K=V，第四个是前馈全连接层对象，最后就是dropout置0比率
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 按照结构图使用clones函数克隆三个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory):
        # forward函数中的参数有2个，分别是来自上一层的输入x，来自编码器层的语义存储变量memory，将memory表示成m之后方便使用。
        "Follow Figure 1 (right) for connections."
        m = memory
        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x，
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x;k,v是编码层输出memory
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))

        # 最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果，这就是我们的解码器结构
        return self.sublayer[2](x, self.feed_forward)


# 使用两个FFN和一个softmax输出最后的结果
class Generator(nn.Module):
    def __init__(self, d_model, dropout,device):
        super(Generator, self).__init__()
        # MLP
        self.d_model = d_model
        self.hidden_layer1 = nn.Linear(d_model, d_model)
        # self.relu1 = torch.nn.LeakyReLU()
        # self.hidden_layer2 = nn.Linear(d_model, 1)
        self.hidden_layer2 = nn.Linear(d_model, 1)
        # self.relu2 = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(20, 2)
        self.dropout = nn.Dropout(p=dropout)
        # self.sigmoid = torch.nn.Sigmoid()
        self.Softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x):
        # MLP Classifier

        out = torch.zeros(1, 2, device=self.device)

        for i in range(x.size()[0]):
            # x = x.type(torch.LongTensor)
            # hidden_output1 = self.dropout(self.relu1(self.hidden_layer1(x[i])))
            # hidden_output2 = self.dropout(self.relu2(self.hidden_layer2(hidden_output1)))
            hidden_output1 = self.dropout(self.hidden_layer1(x[i]))
            hidden_output2 = self.dropout(self.hidden_layer2(hidden_output1))
            mlp_output = self.output_layer(hidden_output2.t())
            # print('————————————————————————————————————————————————————————————————')
            # print('Generator参数维度:\n')
            # print('x.size[0]:',x.size()[0])
            # print('mlp_output():',mlp_output.size())
            # print('mlp_output1:',mlp_output[0,0].,'mlp_output2:',mlp_output[0,1])
            output = self.Softmax(mlp_output)
            o = torch.zeros(output.size()[0], 2, device=self.device)
            o[:, 0] = output[:, 0] / (output[:, 0] + output[:, 1])
            o[:, 1] = output[:, 1] / (output[:, 0] + output[:, 1])
            # print(output,output.size())
            out = torch.cat((out, o), 0)
            # print('o,size():',o.size())
            # print('————————————————————————————————————————————————————————————————')
        output = out[1:, :]
        # output = output.type(torch.LongTensor)
        # output =torch.unsqueeze(output,0)
        # print('Generator output: ', output.size())
        return output


# Model Architecture
# 使用Encoder-Encoder->Decoder类来实现编码器-解码器结构
class DoubleEncoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """

    def __init__(self, encoder1, encoder2, fuse, src_embed, tgt_embed, generator):
        # 初始化函数中有5个参数，分别是编码器对象，解码器对象,源数据嵌入函数，目标数据嵌入函数，以及输出部分的类别生成器对象.
        super(DoubleEncoder, self).__init__()
        self.encoder1 = encoder1  # TCR Encoder
        self.encoder2 = encoder2  # Epitope Encoder
        self.src_embed = src_embed  # input embedding module(input embedding + positional encode)
        self.tgt_embed = tgt_embed  # ouput embedding module
        self.fuse = fuse  # fuse the TCR and Epitope's feature maps,Decoder
        self.generator = generator  # output generation module

        # self.prediction = prediction  # output prediction module

    def forward(self, src, tgt):
        "Take in and process masked src and target sequences."
        # 在forward函数中，有四个参数，source代表源数据，target代表目标数据,source_mask和target_mask代表对应的掩码张量,
        # 在函数中，将source source_mask传入编码函数，得到结果后与source_mask target 和target_mask一同传给解码函数
        TCR, Epi = self.encode(src, tgt)
        res = self.decode(Epi, TCR)
        output = self.generator(res)
        #  print('————————————————————————————————————————————')
        # print('Encoder-Decoder中的维度大小:\n')
        # print('self.src_embed:',self.src_embed)
        # print('self.tgt_embed:',self.tgt_embed)
        # print('src_embedds:',self.src_embed(src).size())
        # print('target_embedds:',self.tgt_embed(tgt).size())
        # print('memory(encoder_output):',memory.size())
        # print('res(decoder_output)',res.size())
        # print('output(MLP_output)',output.size())
        # print('————————————————————————————————————————————')
        return output

    def encode(self, src, tgt):
        # 编码函数，以source和source_mask为参数,使用src_embed对source做处理，然后和source_mask一起传给self.encoder
        src_embedds = self.src_embed(src)
        # print("TCR: ",src_embedds.size())
        target_embedds = self.tgt_embed(tgt)
        return self.encoder1(src_embedds), self.encoder2(target_embedds)

    def decode(self, tcr, epi):
        # 解码函数，以memory即编码器的输出，source_mask target target_mask为参数,
        # 使用tgt_embed对target做处理，然后和source_mask,target_mask,memory一起传给self.decoder
        # print('Epitope: ', target_embedds.size())
        return self.fuse(tcr, epi)




def Model(src_vocab, tgt_vocab, emb_type, N=6, d_model=256, d_ff=1024, h=8, dropout=0.2,device = 'cpu'):
    """
    构建模型
    params:
        src_vocab:
        tgt_vocab:
        N: 编码器和解码器堆叠基础模块的个数
        d_model: 模型中embedding的size，默认512
        d_ff: FeedForward Layer层中embedding的size，默认2048
        h: MultiHeadAttention中多头的个数，必须被d_model整除
        dropout:
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    n_decoder = math.ceil(N / 2)
    model = DoubleEncoder(
        Encoder(EncoderModule(CNN(d_model, dropout), d_model, c(attn), c(ff), dropout, d_model), N),
        Encoder(EncoderModule(CNN(d_model, dropout), d_model, c(attn), c(ff), dropout, d_model), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), 1),
        nn.Sequential(Embeddings(d_model, src_vocab, emb_type), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab, emb_type), c(position)),
        Generator(d_model, dropout,device))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class TripleEncoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """

    def __init__(self, encoder1, encoder2, encoder3, fuse, src_embed, tgt_embed, generator):
        # 初始化函数中有5个参数，分别是编码器对象，解码器对象,源数据嵌入函数，目标数据嵌入函数，以及输出部分的类别生成器对象.
        super(TripleEncoder, self).__init__()
        self.encoder1 = encoder1  # TCR Encoder
        self.encoder2 = encoder2  # Epitope Encoder
        self.encoder3 = encoder3
        self.src_embed = src_embed  # input embedding module(input embedding + positional encode)
        self.tgt_embed = tgt_embed  # ouput embedding module
        self.fuse = fuse  # fuse the TCR and Epitope's feature maps,Decoder
        self.generator = generator  # output generation module

        # self.prediction = prediction  # output prediction module

    def forward(self, tcr1,tcr2,epi):
        "Take in and process masked src and target sequences."
        # 在forward函数中，有四个参数，source代表源数据，target代表目标数据,source_mask和target_mask代表对应的掩码张量,
        # 在函数中，将source source_mask传入编码函数，得到结果后与source_mask target 和target_mask一同传给解码函数
        TCR1,TCR2, Epi = self.encode(tcr1, tcr2, epi)
        res = self.decode(Epi, TCR1, TCR2, Epi)
        output = self.generator(res)
        #  print('————————————————————————————————————————————')
        # print('Encoder-Decoder中的维度大小:\n')
        # print('self.src_embed:',self.src_embed)
        # print('self.tgt_embed:',self.tgt_embed)
        # print('src_embedds:',self.src_embed(src).size())
        # print('target_embedds:',self.tgt_embed(tgt).size())
        # print('memory(encoder_output):',memory.size())
        # print('res(decoder_output)',res.size())
        # print('output(MLP_output)',output.size())
        # print('————————————————————————————————————————————')
        return output

    def encode(self, tcr1,tcr2, epi):
        # 编码函数，以source和source_mask为参数,使用src_embed对source做处理，然后和source_mask一起传给self.encoder
        tcr1_embedds = self.src_embed(tcr1)
        tcr2_embedds = self.src_embed(tcr2)
        # print("TCR: ",src_embedds.size())
        epi_embedds = self.tgt_embed(epi)
        return self.encoder1(tcr1_embedds), self.encoder2(tcr2_embedds), self.encoder3(epi_embedds)

    def decode(self, tcr1,tcr2, epi):
        # 解码函数，以memory即编码器的输出，source_mask target target_mask为参数,
        # 使用tgt_embed对target做处理，然后和source_mask,target_mask,memory一起传给self.decoder
        # print('Epitope: ', target_embedds.size())
        return self.fuse(tcr1+tcr2, epi)

def TripleModel(src_vocab, tgt_vocab, emb_type, N=6, d_model=256, d_ff=1024, h=8, dropout=0.2,device = 'cpu'):
    """
    构建模型
    params:
        src_vocab:
        tgt_vocab:
        N: 编码器和解码器堆叠基础模块的个数
        d_model: 模型中embedding的size，默认512
        d_ff: FeedForward Layer层中embedding的size，默认2048
        h: MultiHeadAttention中多头的个数，必须被d_model整除
        dropout:
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    n_decoder = math.ceil(N / 2)
    model = TripleModel(
        Encoder(EncoderModule(CNN(d_model, dropout), d_model, c(attn), c(ff), dropout, d_model), N),
        Encoder(EncoderModule(CNN(d_model, dropout), d_model, c(attn), c(ff), dropout, d_model), N),
        Encoder(EncoderModule(CNN(d_model, dropout), d_model, c(attn), c(ff), dropout, d_model), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), 1),
        nn.Sequential(Embeddings(d_model, src_vocab, emb_type), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab, emb_type), c(position)),
        Generator(d_model, dropout,device))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
