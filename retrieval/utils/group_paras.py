import numpy as np
import faiss
import os
import argparse

def write_file(file_name, samples):
    with open(file_name, 'w') as f_out:
        for _ in samples:
            f_out.write(_)


def group_paras(I, ncentroids, split_path):
    samples = [[] for _ in range(ncentroids)]
    with open('../data/retrieve_train.txt') as f_in:
        for i, line in enumerate(f_in):
            samples[I[i][0]].append(line)
    for i, group in enumerate(samples):
        write_file(split_path + 'split_'+str(i)+'.txt', group)

def clusering(data, niter=1000, verbose=True, ncentroids=1024, max_points_per_centroid=10000000, gpu_id=0, spherical=False):
    # use one gpu
    '''
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu_id

    d = data.shape[1]
    if spherical:
        index = faiss.GpuIndexFlatIP(res, d, cfg)
    else:
        index = faiss.GpuIndexFlatL2(res, d, cfg)
    '''
    d = data.shape[1]
    if spherical:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    clus = faiss.Clustering(d, ncentroids)
    clus.verbose = True
    clus.niter = niter
    clus.max_points_per_centroid = max_points_per_centroid

    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)
    centroids = centroids.reshape(ncentroids, d)

    index.reset()
    index.add(centroids)
    D, I = index.search(data, 1)

    return D, I

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncentroids', type=int, default=10000)
    parser.add_argument('--niter', type=int, default=250)
    parser.add_argument('--max_points_per_centroid', type=int, default=1000)
    parser.add_argument('--indexpath', type=str, default=None)
    parser.add_argument('--spherical', action='store_true')
    args = parser.parse_args()


    train_para_embed_path = "encodings/train_para_embed.npy"
    split_save_path = "../data/data_splits/"
    if os.path.exists(split_save_path) and os.listdir(split_save_path):
        print(f"output directory {split_save_path} already exists and is not empty.")
    if not os.path.exists(split_save_path):
        os.makedirs(split_save_path, exist_ok=True)

    x = np.load(train_para_embed_path)
    x = np.float32(x)

    D, I = clusering(x, niter=args.niter, ncentroids=args.ncentroids, max_points_per_centroid=args.max_points_per_centroid, spherical=args.spherical)

    group_paras(I, args.ncentroids, split_path=split_save_path)
