from option.options import options, config
from data.dataloader import get_dataloader
import torch
import random
from model.model import TextImgPersonReidNet, TextImgPersonReidNet_Res50_fusetrans_2moudel
from loss.Id_loss import Id_Loss , Id_Loss_2, Id_Loss_3
from loss.RankingLoss import RankingLoss
from torch import optim
import logging
import os
from test_during_train import test , test_2module , test_TOP50,test_TOP50_2 , test_TOP50_test
from torch.autograd import Variable
import time
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_checkpoint(state, opt):

    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    torch.save(state, filename)


def load_checkpoint(opt):
    filename = os.path.join(opt.save_path, 'model/best.pth.tar')
    state = torch.load(filename)

    return state


def calculate_similarity(image_embedding, text_embedding):
    image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    return similarity


if __name__ == '__main__':
    opt = options().opt
    opt.GPU_id = '1'
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))
    # opt.GPU_id1 = '1'
    # opt.device1 = torch.device('cuda:{}'.format(opt.GPU_id1))
    opt.data_augment = False
    opt.lr = 0.001
    opt.margin = 0.2

    opt.feature_length = 1024

    opt.dataset = 'CUHK-PEDES'

    if opt.dataset == 'MSMT-PEDES':
        opt.pkl_root = '/data1/zhiying/text-image/MSMT-PEDES/3-1/'
        opt.class_num = 3102
        opt.vocab_size = 2500
        # opt.class_num = 2802
        # opt.vocab_size = 2300
    elif opt.dataset == 'CUHK-PEDES':
        opt.pkl_root = '/data1/zhiying/text-image/CUHK-PEDES_/' # same_id_new_
        opt.class_num = 11000
        opt.vocab_size = 5000

    model_name = 'Decoder_theta1_batchsize32'.format(opt.lr)
    # model_name = 'test'
    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + model_name
    opt.arf = 0.1
    opt.dim = 1024
    opt.depth = 4
    opt.heads = 8
    opt.mlp_dim = 1024
    opt.dim_head = 64
    opt.channels = 2048
    opt.epoch = 70
    opt.epoch_decay = [20, 40, 50]

    opt.batch_size = 32
    opt.start_epoch = 0
    opt.trained = False

    config(opt)
    opt.epoch_decay = [i - opt.start_epoch for i in opt.epoch_decay]

    train_dataloader = get_dataloader(opt)
    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    opt.mode = 'train'

    id_loss_fusion = Id_Loss_3(opt).to(opt.device)
    id_loss_fun = Id_Loss_2(opt).to(opt.device)
    ranking_loss_fun = RankingLoss(opt)
    ranking_loss_fun1 = RankingLoss(opt)
    network = TextImgPersonReidNet_Res50_fusetrans_2moudel(opt).to(opt.device)

    cnn_params = list(map(id, network.ImageExtract.parameters()))
    trans_params = list(map(id, network.Decoder.parameters()))
    other_params = filter(lambda p: id(p) not in cnn_params + trans_params, network.parameters())
    other_params = list(other_params)
    other_params.extend(list(id_loss_fun.parameters()))
    other_params.extend(list(id_loss_fusion.parameters()))
    param_groups = [{'params': other_params, 'lr': opt.lr},
                    {'params': network.Decoder.parameters(), 'lr': opt.lr * 0.1},
                    {'params': network.ImageExtract.parameters(), 'lr': opt.lr*0.1}]
    optimizer = optim.Adam(param_groups, betas=(opt.adam_alpha, opt.adam_beta))

    test_best = 0
    test_front_best = 0
    test_history = 0
    if opt.trained:
        state = load_checkpoint(opt)
        network.load_state_dict(state['network'])
        test_best = state['test_best']
        test_history = test_best
        id_loss_fun.load_state_dict(state['W'])
        print('load the {} epoch param successfully'.format(state['epoch']))
    """
    network.eval()
    test_best = test(opt, 0, 0, network,
                     test_img_dataloader, test_txt_dataloader, test_best)
    network.train()
    exit(0)
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.epoch_decay)

    for epoch in range(opt.start_epoch, opt.epoch):
        id_image_loss_sum = 0
        id_text_loss_sum = 0
        ranking_loss_sum = 0
        id_fusion_loss_sum = 0
        rank_fusion_sum = 0
        pred_i2t_local_sum = 0
        pred_t2i_local_sum = 0
        pred_fuse_sum = 0
        scheduler.step()
        for param in optimizer.param_groups:
            logging.info('lr:{}'.format(param['lr']))

        for times, [image, label, caption_code, caption_length] in enumerate(train_dataloader):

            image = Variable(image.to(opt.device))
            label = Variable(label.to(opt.device))
            caption_code = Variable(caption_code.to(opt.device).long())
            optimizer.zero_grad()
            # caption_length = caption_length.to(opt.device)
            feature_id , score_mat , image_embedding , text_embedding = network(image, caption_code, caption_length)
            # image_embedding, text_embedding = network(image, caption_code, caption_length)
            ##########compute 2module loss
            id_image_loss, id_text_loss, pred_i2t_local, pred_t2i_local = id_loss_fun(image_embedding, text_embedding, label)
            # id_loss, pred_i2t_local, pred_t2i_local = [0, 0, 0]
            similarity = calculate_similarity(image_embedding, text_embedding)
            ranking_loss = ranking_loss_fun(similarity, label)
            ##########compute fusion loss
            id_fusion_loss, pred_fuse ,_ = id_loss_fusion(feature_id, label)
            # id_loss, pred_i2t_local, pred_t2i_local = [0, 0, 0]
            similarity_fusion = score_mat.squeeze(2)
            ranking_loss_fusion = ranking_loss_fun1(similarity_fusion, label)

            # theta = 0
            if epoch < 20 :
                theta = 0.05*epoch
                theta = 1
                loss = (id_image_loss + id_text_loss + ranking_loss + theta * id_fusion_loss + theta * ranking_loss_fusion)
            else:
                loss = (id_image_loss + id_text_loss + ranking_loss + id_fusion_loss + ranking_loss_fusion)
            loss.backward()
            optimizer.step()

            if (times + 1) % 50 == 0:
                logging.info(
                    "Epoch: %d/%d Setp: %d, id_image_loss: %.2f, id_text_loss: %.2f,ranking_loss: %.2f , ranking_loss_fusion: %.2f, id_fusion_loss: %.2f "
                    "pred_fuse: %.3f "
                    "pred_i2t: %.3f pred_t2i %.3f"
                    % (epoch + 1, opt.epoch, times + 1, id_image_loss, id_text_loss, ranking_loss, ranking_loss_fusion,
                       id_fusion_loss, pred_fuse, pred_i2t_local, pred_t2i_local))

            ranking_loss_sum += ranking_loss
            id_fusion_loss_sum += id_fusion_loss
            id_image_loss_sum += id_image_loss
            id_text_loss_sum += id_text_loss
            rank_fusion_sum += ranking_loss_fusion
            pred_fuse_sum += pred_fuse
            pred_i2t_local_sum += pred_i2t_local
            pred_t2i_local_sum += pred_t2i_local

        ranking_loss_avg = ranking_loss_sum / (times + 1)
        id_fusion_loss_avg = id_fusion_loss_sum / (times + 1)
        pred_i2t_local_avg = pred_i2t_local_sum / (times + 1)
        pred_t2i_local_avg = pred_t2i_local_sum / (times + 1)
        id_image_loss_avg = id_image_loss_sum / (times + 1)
        id_text_loss_avg = id_text_loss_sum / (times + 1)
        rank_fusion_avg = rank_fusion_sum / (times + 1)
        pred_fuse_avg = pred_fuse_sum / (times + 1)

        logging.info("Epoch: %d/%d , id_image_loss: %.2f, id_text_loss: %.2f,ranking_loss: %.2f , ranking_loss_fusion: %.2f, id_fusion_loss: %.2f "
                    "pred_fuse: %.3f "
                    "pred_i2t: %.3f pred_t2i %.3f"
                     % (epoch + 1, opt.epoch, id_image_loss_avg, id_text_loss_avg, ranking_loss_avg, rank_fusion_avg,
                        id_fusion_loss_avg, pred_fuse_avg, pred_i2t_local_avg, pred_t2i_local_avg))

        print(model_name)
        network.eval()
        start_time = time.time()
        test_best , test_best_front= test_TOP50_2(opt, epoch + 1, times + 1, network,
                               test_img_dataloader, test_txt_dataloader, test_best, test_front_best)
        end_time = time.time()
        test_time = end_time - start_time
        print('Test complete in {:.0f}m {:.0f}s'.format(
            test_time // 60, test_time % 60))
        network.train()
        if test_best > test_history:
            state = {
                'test_best': test_best,
                'network': network.cpu().state_dict(),
                'optimizer': optimizer.state_dict(),
                'W': id_loss_fun.cpu().state_dict(),
                'W1': id_loss_fusion.cpu().state_dict(),
                'epoch': epoch + 1}

            save_checkpoint(state, opt)
            network.to(opt.device)
            id_loss_fun.to(opt.device)
            id_loss_fusion.to(opt.device)
            test_history = test_best

    logging.info('Training Done')





