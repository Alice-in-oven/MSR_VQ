import random


def main_triplet_selection(sampling_type,
                           sampling_num,
                           knn,
                           distance_list,
                           anchor_pos,
                           anchor_length,
                           length_list,
                           epoch,
                           pos_begin_pos=0,
                           pos_end_pos=200,
                           neg_begin_pos=0,
                           neg_end_pos=200):
    if sampling_type == "distance_sampling1":
        positive_sampling_index_list, negative_sampling_index_list = [], []

        tem_positive_sampling_index_list, tem_negative_sampling_index_list = distance_sampling1(
            knn,
            sampling_num,
            distance_list,
            pos_begin_pos=pos_begin_pos,
            pos_end_pos=pos_end_pos,
            neg_begin_pos=neg_begin_pos,
            neg_end_pos=neg_end_pos,
        )
        positive_sampling_index_list.extend(tem_positive_sampling_index_list)
        negative_sampling_index_list.extend(tem_negative_sampling_index_list)


        
    else:
        raise ValueError("Sampling type is not supported!")
    return positive_sampling_index_list, negative_sampling_index_list


def distance_sampling1(knn, sampling_num, distance_list, pos_begin_pos = 0,
                                                         pos_end_pos = 200,
                                                         neg_begin_pos = 0,
                                                         neg_end_pos = 200):
    max_index = max(0, len(knn) - 1)
    pos_begin_pos = min(max(0, pos_begin_pos), max_index)
    pos_end_pos = min(max(0, pos_end_pos), max_index)
    neg_begin_pos = min(max(0, neg_begin_pos), max_index)
    neg_end_pos = min(max(0, neg_end_pos), max_index)
    positive_sample_index = []
    negative_sample_index = []
    positive_knn_sample   = []
    negative_knn_sample   = []
    for i in range(sampling_num):
        sampling_begin_pos = pos_begin_pos
        sampling_end_pos   = pos_end_pos
        first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        while (distance_list[knn[first_num]] < 0.0001 and sampling_end_pos > sampling_begin_pos):
            # print(first_num, distance_list[knn[first_num]])
            first_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        sampling_begin_pos = neg_begin_pos
        sampling_end_pos   = neg_end_pos
        second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        while (distance_list[knn[second_num]] < 0.0001 and sampling_end_pos > sampling_begin_pos):
            second_num  = random.randint(sampling_begin_pos, sampling_end_pos)
        while(second_num == first_num and sampling_end_pos > sampling_begin_pos):
            second_num = random.randint(sampling_begin_pos, sampling_end_pos)
        if first_num > second_num:
            first_num, second_num = second_num, first_num
        positive_sample_index.append(knn[first_num])
        negative_sample_index.append(knn[second_num])
        positive_knn_sample.append(first_num)
        negative_knn_sample.append(second_num)
    return positive_sample_index, negative_sample_index

