import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.read_write_data import write_txt


def calculate_cos(image_embedding, text_embedding):
    image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)
    text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + 1e-8)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    return similarity

def calculate_cos_part(numpart,image_embedding, text_embedding):
    image_embedding = torch.cat([image_embedding[i] for i in range(numpart)], dim=1)
    text_embedding = torch.cat([text_embedding[i] for i in range(numpart)], dim=1)
    image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True)+ 1e-8)
    text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True)+ 1e-8)
    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    return similarity
def calculate_similarity(image_feature_local, text_feature_local):

    with torch.no_grad():

        similarity_local_part_i = calculate_cos(image_feature_local, text_feature_local)
        similarity = similarity_local_part_i

    return similarity.cpu()

def calculate_similarity_part(numpart,image_feature_local, text_feature_local):

    with torch.no_grad():

        similarity_local_part_i = calculate_cos_part(numpart,image_feature_local, text_feature_local)
        similarity = similarity_local_part_i

    return similarity.cpu()

def calculate_ap(similarity, label_query, label_gallery):
    """
        calculate the similarity, and rank the distance, according to the distance, calculate the ap, cmc
    :param label_query: the id of query [1]
    :param label_gallery:the id of gallery [N]
    :return: ap, cmc
    """
    index = np.argsort(similarity)[::-1]  # the index of the similarity from huge to small
    good_index = np.argwhere(label_gallery == label_query)  # the index of the same label in gallery
    cmc = np.zeros(index.shape)
    mask = np.in1d(index, good_index)  # get the flag the if index[i] is in the good_index
    precision_result = np.argwhere(mask == True)  # get the situation of the good_index in the index
    precision_result = precision_result.reshape(precision_result.shape[0])
    if precision_result.shape[0] != 0:
        cmc[int(precision_result[0]):] = 1  # get the cmc
        d_recall = 1.0 / len(precision_result)
        ap = 0
        for i in range(len(precision_result)):  # ap is to calculate the PR area
            precision = (i + 1) * 1.0 / (precision_result[i] + 1)
            if precision_result[i] != 0:
                old_precision = i * 1.0 / precision_result[i]
            else:
                old_precision = 1.0
            ap += d_recall * (old_precision + precision) / 2
        return ap, cmc
    else:
        return None, None


def evaluate(similarity, label_query, label_gallery):
    similarity = similarity.numpy()
    label_query = label_query.numpy()
    label_gallery = label_gallery.numpy()

    cmc = np.zeros(label_gallery.shape)
    ap = 0
    for i in range(len(label_query)):
        ap_i, cmc_i = calculate_ap(similarity[i, :], label_query[i], label_gallery)
        cmc += cmc_i
        ap += ap_i
    """
    cmc_i is the vector [0,0,...1,1,..1], the first 1 is the first right prediction n,
    rank-n and the rank-k after it all add one right prediction, therefore all of them's index mark 1
    Through the  add all the vector and then divive the n_query, we can get the rank-k accuracy cmc
    cmc[k-1] is the rank-k accuracy   
    """
    cmc = cmc / len(label_query)
    map = ap / len(label_query)  # map = sum(ap) / n_query

    return cmc, map


def evaluate_without_matching_image(similarity, label_query, label_gallery, txt_img_index):
    similarity = similarity.numpy()
    label_query = label_query.numpy()
    label_gallery = label_gallery.numpy()

    cmc = np.zeros(label_gallery.shape[0] - 1)
    ap = 0
    count = 0
    for i in range(len(label_query)):

        similarity_i = similarity[i, :]
        similarity_i = np.delete(similarity_i, txt_img_index[i])
        label_gallery_i = np.delete(label_gallery, txt_img_index[i])
        ap_i, cmc_i = calculate_ap(similarity_i, label_query[i], label_gallery_i)
        if ap_i is not None:
            cmc += cmc_i
            ap += ap_i
        else:
            count += 1
    """
    cmc_i is the vector [0,0,...1,1,..1], the first 1 is the first right prediction n,
    rank-n and the rank-k after it all add one right prediction, therefore all of them's index mark 1
    Through the  add all the vector and then divive the n_query, we can get the rank-k accuracy cmc
    cmc[k-1] is the rank-k accuracy   
    """
    cmc = cmc / (len(label_query) - count)
    map = ap / (len(label_query) - count)  # map = sum(ap) / n_query

    return cmc, map


def load_checkpoint(model_root, model_name):
    filename = os.path.join(model_root, 'model', model_name)
    state = torch.load(filename, map_location='cpu')

    return state


def write_result(similarity, img_labels, txt_labels, txt_img_index, name, txt_root, best_txt_root, epoch, best, iteration):
    write_txt(name, txt_root)
    print(name)
    t2i_cmc_wm, t2i_map_wm = evaluate_without_matching_image(similarity.t(), txt_labels, img_labels, txt_img_index)
    t2i_cmc, t2i_map = evaluate(similarity.t(), txt_labels, img_labels)
    str = "t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, map: {:.4}".format(t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_map)
    write_txt(str, txt_root)
    print(str)
    str = "t2i_wm: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, map: {:.4}".format(t2i_cmc_wm[0], t2i_cmc_wm[4],
                                                                           t2i_cmc_wm[9], t2i_map_wm)
    write_txt(str, txt_root)
    print(str)

    if t2i_cmc[0] > best:
        str = "Testing Epoch: {} Iteration:{}".format(epoch, iteration)
        write_txt(str, best_txt_root)
        write_txt(name, best_txt_root)
        str = "t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, map: {:.4}".format(t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_map)
        write_txt(str, best_txt_root)
        str = "t2i_wm: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, map: {:.4}".format(t2i_cmc_wm[0], t2i_cmc_wm[4],
                                                                               t2i_cmc_wm[9], t2i_map_wm)
        write_txt(str, best_txt_root)

        return t2i_cmc[0]
    else:
        return best


def test(opt, epoch, iteration, network, img_dataloader, txt_dataloader, best):
    txt_root = os.path.join(opt.save_path, 'log', 'test_separate.log')
    best_txt_root = os.path.join(opt.save_path, 'log', 'best_test.log')

    str = "Testing Epoch: {} Iteration:{}".format(epoch, iteration)
    write_txt(str, txt_root)
    print(str)

    image_feature = torch.FloatTensor().to(opt.device)
    img_labels = torch.LongTensor().to(opt.device)

    for times, [image, label] in enumerate(img_dataloader):
        image = image.to(opt.device)
        label = label.to(opt.device)

        with torch.no_grad():
            image_feature_i = network.img_embedding(image)
        image_feature = torch.cat([image_feature, image_feature_i], 0)
        img_labels = torch.cat([img_labels, label.view(-1)], 0)

    text_feature = torch.FloatTensor().to(opt.device)
    txt_labels = torch.LongTensor().to(opt.device)
    txt_img_index = []

    for times, [label, caption_code, caption_length, caption_mask, caption_matching_img_index] in enumerate(txt_dataloader):
        label = label.to(opt.device)
        caption_code = caption_code.to(opt.device).long()
        caption_mask = caption_mask.to(opt.device)
        txt_img_index.append(caption_matching_img_index)
        with torch.no_grad():
            text_feature_i = network.txt_embedding(caption_code, caption_mask)

        text_feature = torch.cat([text_feature, text_feature_i], 0)
        txt_labels = torch.cat([txt_labels, label.view(-1)], 0)
    txt_img_index = torch.cat(txt_img_index, 0)
    similarity = calculate_similarity(image_feature, text_feature)

    img_labels = img_labels.cpu()
    txt_labels = txt_labels.cpu()

    best = write_result(similarity, img_labels, txt_labels, txt_img_index, 'similarity_all:',
                        txt_root, best_txt_root, epoch, best, iteration)

    return best


def test_part(opt, epoch, iteration, network, img_dataloader, txt_dataloader, best):
    txt_root = os.path.join(opt.save_path, 'log', 'test_separate.log')
    best_txt_root = os.path.join(opt.save_path, 'log', 'best_test.log')

    str = "Testing Epoch: {} Iteration:{}".format(epoch, iteration)
    write_txt(str, txt_root)
    print(str)

    image_feature = torch.FloatTensor().to(opt.device)
    img_labels = torch.LongTensor().to(opt.device)

    for times, [image, label] in enumerate(img_dataloader):
        image = image.to(opt.device)
        label = label.to(opt.device)

        with torch.no_grad():
            _,image_feature_i = network.img_embedding(image)

        image_feature = torch.cat([image_feature, image_feature_i], 1)
        img_labels = torch.cat([img_labels, label.view(-1)], 0)

    text_feature = torch.FloatTensor().to(opt.device)
    txt_labels = torch.LongTensor().to(opt.device)
    txt_img_index = []

    for times, [label, caption_code, caption_length, caption_mask, caption_matching_img_index] in enumerate(txt_dataloader):
        label = label.to(opt.device)
        caption_code = caption_code.to(opt.device).long()
        # caption_length = caption_length.to(opt.device)
        caption_mask = caption_mask.to(opt.device)
        txt_img_index.append(caption_matching_img_index)
        with torch.no_grad():
            _ , text_feature_i = network.txt_embedding(caption_code, caption_mask)

        text_feature = torch.cat([text_feature, text_feature_i], 1)
        txt_labels = torch.cat([txt_labels, label.view(-1)], 0)
    txt_img_index = torch.cat(txt_img_index, 0)
    similarity = calculate_similarity_part(opt.num_query,image_feature, text_feature)

    img_labels = img_labels.cpu()
    txt_labels = txt_labels.cpu()

    best = write_result(similarity, img_labels, txt_labels, txt_img_index, 'similarity_all:',
                        txt_root, best_txt_root, epoch, best, iteration)
    return best

def test_part_MPN(opt, epoch, iteration, network, img_dataloader, txt_dataloader, best):
    txt_root = os.path.join(opt.save_path, 'log', 'test_separate.log')
    best_txt_root = os.path.join(opt.save_path, 'log', 'best_test.log')

    str = "Testing Epoch: {} Iteration:{}".format(epoch, iteration)
    write_txt(str, txt_root)
    print(str)

    image_feature = torch.FloatTensor().to(opt.device)
    img_labels = torch.LongTensor().to(opt.device)

    for times, [image, label] in enumerate(img_dataloader):
        image = image.to(opt.device)
        label = label.to(opt.device)

        with torch.no_grad():
            image_feature_i , _= network.img_embedding(image)

        image_feature = torch.cat([image_feature, image_feature_i], 1)
        img_labels = torch.cat([img_labels, label.view(-1)], 0)

    text_feature = torch.FloatTensor().to(opt.device)
    txt_labels = torch.LongTensor().to(opt.device)
    txt_img_index = []

    for times, [label, caption_code, caption_length, caption_matching_img_index] in enumerate(txt_dataloader):
        label = label.to(opt.device)
        caption_code = caption_code.to(opt.device).long()
        # caption_length = caption_length.to(opt.device)
        txt_img_index.append(caption_matching_img_index)
        with torch.no_grad():
            text_feature_i = network.txt_embedding(caption_code, caption_length)

        text_feature = torch.cat([text_feature, text_feature_i], 1)
        txt_labels = torch.cat([txt_labels, label.view(-1)], 0)
    txt_img_index = torch.cat(txt_img_index, 0)
    similarity = calculate_similarity_part(opt.num_query,image_feature, text_feature)

    img_labels = img_labels.cpu()
    txt_labels = txt_labels.cpu()

    best = write_result(similarity, img_labels, txt_labels, txt_img_index, 'similarity_all:',
                        txt_root, best_txt_root, epoch, best, iteration)
    return best
