from QRec import QRec
from util.config import ModelConf
import time

from util.dataSplit import DataSplit
from util.io import FileIO

if __name__ == '__main__':

    print('=' * 80)
    print('  QRec:一个有效的基于python的推荐模型库。   ')
    print('=' * 80)

    print('DNNs-based Recommenders:')
    print('d1. APR           d2. CDAE          d3. NeuMF')

    print('GNNs-based Recommenders:')
    print('g1. NGCF          g2. LightGCN        g3. DHCF')

    print('=' * 80)
    num = input('请输入您要运行的模型编号::')

    s = time.time()
    # Register your model here and add the conf file into the config directory
    models = {
        'd1': 'APR', 'd2': 'CDAE', 'd3': 'NeuMF',
        'g1': 'NGCF', 'g2': 'LightGCN', 'g3': 'DHCF',

    }
    try:
        conf = ModelConf('./config/' + models[num] + '.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    recSys = QRec(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
