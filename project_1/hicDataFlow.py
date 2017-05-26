import numpy as np
import argparse
import math
# import sys
import random

# class HiC_mat(object):
#
#     def __init__(self, contig_names, inter_data, prob_vec):
#         # both are arrays in this point need to ask noam if should be otherwise
#         self.contig=contig_names
#         self.data=inter_data
#         self.prob=prob_vec #upper triangular probability vector


def convert_hic(file, size):
    # load and parse file
    with open(file, 'r') as fh:
        contig_list = fh.next().rstrip("\r\n").split("\t")[1:size + 1]
        inter_data = []
        for i, line in enumerate(fh):
            if i >= size:
                break
            inter_data.append(line.rstrip("\r\n").split("\t")[1:size + 1])
        fh.close()

    contig_list = [x for x in contig_list if str(x) != 'nan']
    inter_array = np.array(inter_data)
    inter_array = inter_array.astype(np.float)
    inter_array = inter_array[~np.isnan(inter_array)]
    array_size = math.sqrt(inter_array.size)
    inter_array = np.reshape(inter_array, (int(array_size), int(array_size)))
    np.savetxt('interaction_mat_from_hic_no_nan.txt', inter_array)

    upper_inter_triu = inter_array[np.triu_indices(int(array_size))]
    upper_sum = 1.0 / (np.sum(upper_inter_triu))
    prob_triu_array = upper_sum * upper_inter_triu
    # should i convert to an array?

    return prob_triu_array, contig_list  # if i change two elements of a class can i return the hole class?


def convert_scaling_plot(file, size):
    # converts the mean_edag_vec to a prob upper triangular and the contig list
    # is just a vector from 0- (n-1)
    with open(file, 'r') as fh:
        diag_mean_list = fh.next().rstrip("\r\n").split("\t")[1:size + 1]
        diag_mean_vec = np.array(diag_mean_list)
    inter_array = np.zeros((size, size), int)
    m = size
    for i in xrange(size):  # xrange starts from 0 - n-1
        param_vec = np.full([1, m], diag_mean_vec[i])
        inter_array += np.diagflat(param_vec, i)
        m -= 1
    contig_list = np.arange(size)
    contig_list = contig_list.tolist()

    upper_inter_triu = inter_array[np.triu_indices(n)]
    upper_sum = 1.0 / (np.sum(upper_inter_triu))
    prob_triu_array = upper_sum * upper_inter_triu
    return prob_triu_array, contig_list


def create_virtual(probability_vec, n_samples):
    N = int(math.sqrt(np.size(probability_vec)))
    # upper_sum = 1.0 / (np.sum(probability_mat[np.triu_indices(N)]))
    # prob_triu = upper_sum * np.array(probability_mat)
    multi_vec = np.random.multinomial(n_samples, probability_vec, 1)
    # currently we will return a upper triangular vec but if needed a mat we can uncomment the flowing
    new_mat = np.zeros((N, N))
    new_mat[np.triu_indices(N)] = multi_vec
    t_new_mat = np.transpose(new_mat)
    new_mat[np.tril_indices(N)] = t_new_mat[np.tril_indices(N)]
    virtual_hic = new_mat
    return virtual_hic


def down_sample(virtual_hic, contig_list, percentage, min_distance):
    contig_array = np.array(contig_list)
    mat_size = np.size(virtual_hic)
    samples = int(math.ceil(float(mat_size*float(percentage/100))))
    if percentage != 0:
        # sub_mat=np.zeroes(samples, samples)
        hist_sample = np.zeros(samples)
        # choosing the samples
        for i in xrange(samples):
            rand = random.randint(1, samples)
            while hist_sample[rand] == 1 and rand-j < 1 and rand+j > mat_size:
                rand = random.randint(1, samples)
            hist_sample[rand] = 1
            for j in xrange(min_distance, 0, -1):  # gives min_distance till 1
                virtual_hic[rand-j, :mat_size] = 0
                virtual_hic[:mat_size, rand-j] = 0
                hist_sample[rand-j] = 1
                contig_array[rand-j] = 0
                virtual_hic[rand + j, :mat_size] = 0
                virtual_hic[:mat_size, rand + j] = 0
                hist_sample[rand+j] = 1
                contig_array[rand-j] = 0

        # zeroing all rows and columns that weren't generated in the sampling
        for i in hist_sample:
            if hist_sample[i] == 0:
                virtual_hic[i, :mat_size] = 0
                virtual_hic[:mat_size, i] = 0
                contig_array[i] = 0

    elif percentage == 0 and min_distance != 0:
        for i in xrange(1, mat_size+1, min_distance+1):  # min dist 1 means we skip every other contig
            virtual_hic[i, :mat_size] = 0
            virtual_hic[:mat_size, i] = 0
            contig_array[i] = 0

    else:  # if percentage and min_distance == 0 then we don't need to down sample
        return virtual_hic, contig_array
    # deleting all zeroed columns and rows of virtual_mat
    virtual_hic = virtual_hic[~np.all([virtual_hic == 0], axis=1)]
    virtual_hic = virtual_hic[~np.all([virtual_hic == 0], axis=0)]
    sub_mat = virtual_hic
    return sub_mat, contig_array


def shuffle(sub_matrix):
    size = np.size(sub_matrix)
    shuffled_vec = np.arange(size)
    np.random.shuffle(shuffled_vec)
    shuffled_mat = np.zeros(size, size)
    # arrange the first contig to be the contig thats number appears in the first cell of the vec...
    for i in shuffled_vec:
        shuffled_mat[i, :size] = sub_matrix[shuffled_vec[i], :size]
        shuffled_mat[:size, i] = sub_matrix[:size, shuffled_vec[i]]
    return shuffled_mat, shuffled_vec


# def Evaluation(reconstructed_matrix):
    # Distance_evaluation
    # Order_evaluation
    # Orientation_evaluation

def main():
        parser = argparse.ArgumentParser(description='HiC matrix or scaling_plot')
        parser.add_argument('-f', '--file', help='tab delimited input file, 1 header col with contig name,'
                            ' 1 header row with contig name', dest='infile', type=str, required=True)
        parser.add_argument("source_type", type=str, choices=['HiC', 'ScalingVec'])
        parser.add_argument("-s", "--size", type=int, default=50)
        parser.add_argument("-m", "--m_samples", type=int, default=2000)
        parser.add_argument("-p", "--percent", type=int, default=0)
        parser.add_argument("-d", "--distance", type=int, default=0)
        args = parser.parse_args()
        if args.suorce_type == 'HiC':
            (prob_mat, contig_list) = convert_hic(args.infile, args.size)
        else:  # args.source_type=='ScalingVec':
            (prob_mat, contig_list) = convert_scaling_plot(args.infile, args.size)

        virtual_hic = create_virtual(prob_mat, args.m_samples)
        (sub_mat, contig_array) = down_sample(virtual_hic, contig_list, args.precent, args.distance)
        (shuffled_mat, shuffled_vec) = shuffle(sub_mat)
        np.savetxt('shuffled_vec.txt', shuffled_vec)
        np.savetxt('shuffled_mat.txt', shuffled_mat)
        # reconstructed_mat=scaffolding(shuffled_mat)
        # score=evaluation(reconstructed_vec, contig_array)

        # still need to do input correctness check

if __name__ == "__main__":
        main()
