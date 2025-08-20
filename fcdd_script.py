from run_fcdd import launch
import os
import numpy as np

if __name__ == '__main__':
    data_path = 'test_data/mnist/0/2'
    info = data_path.split('/')
    dataset = info[-2]
    seed = info[-1]


    htmaps, scores, gtmaps, labels = launch(data_path, epochs=0, batch_size=2)

    ret_path = os.path.join('results', dataset, seed)

    if not os.path.exists('results'):
        os.makedirs('results')
        os.makedirs(os.path.join('results', dataset))
        os.makedirs(ret_path)
    elif not os.path.exists(os.path.join('results', dataset)):
        os.makedirs(os.path.join('results', dataset))
        os.makedirs(ret_path)
    elif not os.path.exists(ret_path):
        os.makedirs(ret_path)

    np.save(open(os.path.join(ret_path, 'fcdd_htmaps.npy'), 'wb'), htmaps)
    np.save(open(os.path.join(ret_path, 'fcdd_scores.npy'), 'wb'), np.array(scores))

    if not os.path.exists(os.path.join(ret_path, 'gt.npy')):
        np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), gtmaps)

    if not os.path.exists(os.path.join(ret_path, 'labels.npy')):
        np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), labels)