from utils import load_data
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math
import os
import time
from datetime import datetime
from py.predict import Predictor
from py.predict import ttypes
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import torch.nn as nn
import torch.nn.functional as F

import torch

input_size = 52
hidden_size = 128
num_layers = 2
num_classes = 21
batch_size = 256
num_epochs = 2
learning_rate = 0.01


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class PredictionHandler:
    def __init__(self):
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data()

    def ping(self):
        print('ping()')

    def pong(self, data):
        print(data)
        data.append(1.5)
        return data

    def predict(self, i):
        idx_test = torch.LongTensor(self.idx_test)

        print(datetime.now(), " Receive data successfully.")
        model = GCN(nfeat=13845, nhid=128, nclass=5, dropout=0.5)
        model.eval()
        script_dir = os.path.dirname(__file__)
        model.load_state_dict(torch.load(
            os.path.join(script_dir, 'saved/model.pkl')))

        outputs = model(self.features, self.adj)
        # loss = F.nll_loss(outputs[idx_test], labels[idx_test])
        _, predicted = torch.max(outputs[idx_test], 1)
        confidence = []
        for idx, item in enumerate(outputs[idx_test]):
            confidence.append(item[predicted[idx]])
        result = predicted[i]

        pred = ttypes.pred()
        pred.type = int(result)
        pred.confidence = float(-confidence[i])
        pred.timestamp = str(round(time.time() * 1000))
        print(pred)
        return pred


if __name__ == '__main__':
    model = GCN(nfeat=13845, nhid=128, nclass=5, dropout=0.5)
    script_dir = os.path.dirname(__file__)
    model.load_state_dict(torch.load(
        os.path.join(script_dir, 'saved/model.pkl')))
    handler = PredictionHandler()
    processor = Predictor.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    # You could do one of these for a multithreaded server
    server = TServer.TThreadedServer(
        processor, transport, tfactory, pfactory)
    server.serve()
