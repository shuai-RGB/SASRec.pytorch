import numpy as np
import torch

#逐点前馈神经网络
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):#指定了卷积层的输入和输出通道数。指定了丢弃层的丢弃率

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)#这些是 1D 卷积层 (torch.nn.Conv1d)，卷积核大小设置为 1，对输入数据的每个位置（逐点）进行线性变换，可用于提取特征或者融合信息等
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)#dropout1 和 dropout2：丢弃层，用于在训练过程中随机置零一些输入，起到防止过拟合的作用。
        self.relu = torch.nn.ReLU()#relu：ReLU 激活函数，分别应用于每个卷积层之后，增加非线性
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):#inputs（张量）：可能形状是(batch_size, sequence_length, hidden_units)
        #torch.nn.Conv1d需要输入形状为(batch_size, channels, length)   通过inputs.transpose(-1, -2)将其维度顺序调整为适合 1 维卷积层输入的格式
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs#残差连接
        return outputs
        # 输入张量首先通过 conv1，对序列进行 1D 卷积。
        # dropout1 应用于 conv1 的输出，以随机丢弃一些元素。
        # 结果通过 relu 激活函数增加非线性。
        # 结果再经过 conv2，进行另一轮 1D 卷积。
        # dropout2 应用于 conv2 的输出。
        # 最终输出通过转置最后两个维度恢复到原来的维度，以便适应后续模型中的输入格式。
        # 原始输入 (inputs) 然后被添加到输出上，以实现残差连接，这有助于训练更深的网络，使得训练过程更加高效和稳定。残差连接在处理深度神经网络时特别有用，它通过保持梯度流动的稳定性来防止梯度消失或爆炸。
    #原始张量--交换第二个和第三个维度-》--一维卷积-->

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

#基于自注意力机制的序列推荐模型SASRec
class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)#item_emb嵌入层，用于将物品编号（从0到item_num）映射到维度为args.hidden_units的向量空间
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)#pos_emb是位置嵌入层，用于给序列中的每个位置赋予一个特定的嵌入向量，同样维度是args.hidden_units
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)#定义了emb_dropout层，用于在嵌入层输出后以概率args.dropout_rate进行随机失活操作，防止过拟合。

        #构建模块列表  分别创建了四个ModuleList类型的列表，用于存储后续构建的多个层归一化、多头注意力以及逐点前馈网络层
        #torch.nn.ModuleList()是 PyTorch 中的一个类，它是一个存储模块（Module）的列表。在神经网络的构建中，它用于方便地管理一系列的神经网络层或其他自定义模块。
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        #还定义了last_layernorm，这是最后一层的层归一化层，用于对经过多层处理后的输出进行归一化操作，稳定训练过程和提升模型性能
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        #循环构建各层。
        for _ in range(args.num_blocks):
            #每次循环创建一个新的层归一化对象new_attn_layernorm并添加到attention_layernorms列表中，用于对多头注意力层的输入进行归一化处理。
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            #创建一个torch.nn.MultiheadAttention对象，并添加到attention_layers列表中
            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            # 创建一个新的层归一化对象new_fwd_layernorm添加到forward_layernorms列表，用于对逐点前馈网络的输入进行归一化。
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            #创建一个PointWiseFeedForward类的实例new_fwd_layer，并添加到forward_layers列表中，逐点前馈网络用于对经过注意力机制处理后的特征进行进一步的非线性变换和特征融合等操作。
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()


    #首先接收log_seqs作为输入，它应该是表示用户行为序列
    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        #                  转换为 PyTorch 的长整型张量👇  移动到指定的设备👇     self.item_emb通过self.item_emb嵌入层将物品编号转换为对应的嵌入向量，得到seqs
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5  #这可能是一种常见的初始化或者缩放操作，有助于模型训练的稳定性和效果优化
        #生成位置信息张量poss   复制从1到序列长度的数组来匹配输入序列的批量维度
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        #屏蔽无效的物品信息
        poss *= (log_seqs != 0)
        #将位置信息添加到seqs上。
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    #通过用户的历史行为序列，计算出正样本和负样本的得分（logits），并为后续的损失函数计算提供输入。
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    #根据用户的历史行为序列来预测用户对不同物品的兴趣或偏好，并输出每个物品的得分（logits）。
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        #这里从 log_feats 中提取序列的最后一时刻（即最后一个时间步）对应的特征。
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
