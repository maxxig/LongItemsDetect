from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from datetime import datetime
from fasttext import train_supervised # pip install fasttext-wheel


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    print((N, p, r))


if __name__ == "__main__":
    train_data = os.path.join(os.getenv("DATADIR", ''), 'product_description.train')
    valid_data = os.path.join(os.getenv("DATADIR", ''), 'product_description.valid')
    print(f'Start train {datetime.now()}')
    # train_supervised uses the same arguments and defaults as the fastText cli
    model = train_supervised(
        input = train_data, epoch = 50, lr = 1.0, wordNgrams = 2, bucket = 200000, dim = 300, loss = 'hs', minCount = 1 #, thread=4 #, ws = 10  # 0.854
        # 75, 81.7
    )
    print(f'Finish train {datetime.now()}')
    print_results(*model.test(valid_data))
    print(f'Finish validate {datetime.now()}')

    # model = train_supervised(
    #     input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1,
    #     loss="hs"
    # )
    # print_results(*model.test(valid_data))
    model.save_model("product_description4.bin")

    # print(f'Start quantize {datetime.now()}')
    # model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    # print(f'Test quantize {datetime.now()}')
    # print_results(*model.test(valid_data))
    # model.save_model("product_description4.ftz")
