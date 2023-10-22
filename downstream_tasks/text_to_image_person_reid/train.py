from option.options import options, config
from data.dataloader import get_dataloader
import torch
import random
from loss.Id_loss import Id_Loss
from loss.RankingLoss import RankingLoss
from torch import optim
import logging
import os
from test_during_train import test
from torch.autograd import Variable
from model.DETR_model import TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3_bert

seed_num = 233
torch.manual_seed(seed_num)
random.seed(seed_num)

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

def calculate_similarity_part(numpart,image_embedding, text_embedding):
    image_embedding = torch.cat([image_embedding[i] for i in range(numpart)],dim=1)
    text_embedding = torch.cat([text_embedding[i] for i in range(numpart)], dim=1)
    image_embedding_norm = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    text_embedding_norm = text_embedding / text_embedding.norm(dim=1, keepdim=True)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    return similarity

def calculate_part_id(id_loss_fun,num_query,image_embedding,text_embedding):
    id_loss_ = []
    pred_i2t_ = []
    pred_t2i_ = []
    for i in range(num_query):
        id_loss, pred_i2t_local, pred_t2i_local = id_loss_fun[i](image_embedding[i], text_embedding[i], label)
        id_loss_.append(id_loss)
        pred_i2t_.append(pred_i2t_local)
        pred_t2i_.append(pred_t2i_local)
    id_loss_ = torch.stack(id_loss_)
    id_loss = torch.mean(id_loss_)
    pred_i2t_ = torch.stack(pred_i2t_)
    pred_i2t_local = torch.mean(pred_i2t_)
    pred_t2i_ = torch.stack(pred_t2i_)
    pred_t2i_local = torch.mean(pred_t2i_)

    return id_loss , pred_i2t_local, pred_t2i_local

if __name__ == '__main__':
    opt = options().opt
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))
    opt.data_augment = False
    opt.lr = 0.001
    opt.margin = 0.2

    opt.feature_length = 384

    if opt.dataset == 'ICFG-PEDES':
        opt.pkl_root = 'processed_data_singledata_ICFG/'
        opt.class_num = 3102
        opt.vocab_size = 2500
    elif opt.dataset == 'CUHK-PEDES':
        opt.pkl_root = 'processed_data_singledata_CUHK/'  # same_id_new_
        opt.class_num = 11000
        opt.vocab_size = 5000
    elif opt.dataset == 'RSTPReid':
        opt.pkl_root = 'processed_data_singledata_RSTP/'  # same_id_new_
        opt.class_num = 4100
        opt.vocab_size = 5000

    opt.d_model = 1024
    opt.nhead = 4
    opt.dim_feedforward = 2048
    opt.normalize_before = False
    opt.num_encoder_layers = 3
    opt.num_decoder_layers = 3
    opt.num_query = 6
    opt.detr_lr = 0.0001
    opt.txt_detr_lr = 0.0001
    opt.txt_lstm_lr = 0.001
    opt.res_y = False
    opt.noself = False
    opt.post_norm = False
    opt.n_heads = 4
    opt.n_layers = 2
    opt.share_query = True
    opt.ViT_layer = 8
    opt.wordtype = 'bert'
    model_name = 'posepretrain_Bertlstm_vitB16_bothmaxpool_my'
    opt.save_path = '../checkpoints/{}/'.format(opt.dataset) + model_name

    opt.epoch = 60
    opt.epoch_decay = [20, 40, 50]

    opt.batch_size = 64
    opt.start_epoch = 0
    opt.trained = False

    config(opt)
    opt.epoch_decay = [i - opt.start_epoch for i in opt.epoch_decay]

    train_dataloader = get_dataloader(opt)
    opt.mode = 'test'
    test_img_dataloader, test_txt_dataloader = get_dataloader(opt)
    opt.mode = 'train'
    id_loss_fun = Id_Loss(opt).to(opt.device)
    ranking_loss_fun = RankingLoss(opt)
    network = TextImgPersonReidNet_mydecoder_pixelVit_transTXT_3_bert(opt).to(opt.device)
    logging.info("Model_size: {:.5f}M".format(sum(p.numel() for p in network.parameters()) / 1000000.0))
    ignored_params = (list(map(id, network.ImageExtract.parameters()))
                        + list(map(id, network.TextExtract.parameters()))
                        + list(map(id, network.conv_1X1_2.parameters()))
                    )
    DETR_params = filter(lambda p: id(p) not in ignored_params, network.parameters())
    DETR_params = list(DETR_params)
    param_groups = [{'params': DETR_params, 'lr': opt.detr_lr},
                    {'params': network.ImageExtract.parameters(), 'lr': opt.lr * 0.1},
                    {'params': network.TextExtract.parameters(), 'lr': opt.lr},
                    {'params': network.conv_1X1_2.parameters(), 'lr': opt.lr},
                    {'params': id_loss_fun.parameters(), 'lr': opt.lr}
                    ]

    optimizer = optim.Adam(param_groups, betas=(opt.adam_alpha, opt.adam_beta))

    test_best = 0
    test_history = 0
    if opt.trained:
        state = load_checkpoint(opt)
        network.load_state_dict(state['network'])
        test_best = state['test_best']
        test_history = test_best
        id_loss_fun.load_state_dict(state['W'])
        print('load the {} epoch param successfully'.format(state['epoch']))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.epoch_decay)

    for epoch in range(opt.start_epoch, opt.epoch):

        id_loss_sum = 0
        ranking_loss_sum = 0
        pred_i2t_local_sum = 0
        pred_t2i_local_sum = 0

        scheduler.step()
        for param in optimizer.param_groups:
            logging.info('lr:{}'.format(param['lr']))

        for times, [image, label, caption_code, caption_length, caption_mask] in enumerate(train_dataloader):
            image = Variable(image.to(opt.device))
            label = Variable(label.to(opt.device))
            caption_code = Variable(caption_code.to(opt.device).long())
            caption_mask = Variable(caption_mask.to(opt.device))

            image_embedding,text_embedding = network(image, caption_code, caption_mask)

            id_loss, pred_i2t, pred_t2i = id_loss_fun(image_embedding, text_embedding, label)
            similarity = calculate_similarity(image_embedding, text_embedding)
            ranking_loss = ranking_loss_fun(similarity, label)

            optimizer.zero_grad()

            loss = (id_loss + ranking_loss)
            loss.backward()
            optimizer.step()
            ranking_loss_dict = 0
            id_loss_dict = 0
            ranking_loss_dict_text = 0
            ranking_loss_dict_image = 0
            if (times + 1) % 50 == 0:
                logging.info("Epoch: %d/%d Setp: %d, ranking_loss: %.2f, id_loss: %.2f, ranking_loss_dict: %.2f, id_loss_dict: %.2f,ranking_loss_dict_text: %.2f, ranking_loss_dict_image: %.2f,"
                             "pred_i2t_local: %.3f pred_t2i_local %.3f"
                      % (epoch+1, opt.epoch, times+1, ranking_loss, id_loss, ranking_loss_dict,id_loss_dict,ranking_loss_dict_text,ranking_loss_dict_image,pred_i2t, pred_t2i))

            ranking_loss_sum += ranking_loss
            id_loss_sum += id_loss
            pred_i2t_local_sum += pred_i2t
            pred_t2i_local_sum += pred_t2i

        ranking_loss_avg = ranking_loss_sum / (times + 1)
        id_loss_avg = id_loss_sum / (times + 1)
        pred_i2t_local_avg = pred_i2t_local_sum / (times + 1)
        pred_t2i_local_avg = pred_t2i_local_sum / (times + 1)

        logging.info("Epoch: %d/%d , ranking_loss: %.2f, id_loss: %.2f,"
                     " pred_i2t_local: %.3f, pred_t2i_local %.3f "
                     % (epoch+1, opt.epoch, ranking_loss_avg, id_loss_avg, pred_i2t_local_avg, pred_t2i_local_avg))

        print(model_name)
        network.eval()
        test_best = test(opt, epoch + 1, times + 1, network,
                         test_img_dataloader, test_txt_dataloader, test_best)
        network.train()
        if test_best > test_history:
            state = {
                'test_best': test_best,
                'network': network.cpu().state_dict(),
                'optimizer': optimizer.state_dict(),
                'W': id_loss_fun.cpu().state_dict(),
                'epoch': epoch + 1}

            save_checkpoint(state, opt)
            network.to(opt.device)
            id_loss_fun.to(opt.device)

            test_history = test_best

    logging.info('Training Done')





