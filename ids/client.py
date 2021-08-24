from typing import List
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport import TSocket
from py.predict import Predictor
import time
from utils import load_data


def main():
    # Make socket
    transport = TSocket.TSocket('localhost', 9090)

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)

    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    client = Predictor.Client(protocol)

    # Connect!
    transport.open()

    client.ping()
    print('ping()')
    print(client.pong([2.0, 2.0]))

    # adj, features, labels, idx_train, idx_val, idx_test = load_data()
    # adj = list(adj)
    # for item in adj:
    #     if not type(item) == list:
    #         item = item.tolist()

    # timestamp = int(time.time())
    result = []
    for i in range(1900):
        result = client.predict(i)
        time.sleep(0.5)
    print(result)
    # Close!
    transport.close()


main()
