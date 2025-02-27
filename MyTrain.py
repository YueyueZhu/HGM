import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
from datetime import datetime
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.dataloader import test_dataset, EvalDataset
import torch.nn.functional as F
import imageio
import numpy as np
from lib.pvt import HGM

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, device, opt):
    global Average, Best_dice, Best_iou, Best_acc, Best_recall, Best_precision, Best_f2, Best_mae
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).to(device)
            gts = Variable(gts).to(device)
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            prediction= model(images)
            # ---- loss function ----
            loss_prediction = structure_loss(prediction, gts)
            loss = loss_prediction
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], ' '[loss: {:.4f} ]'.format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
    save_path = opt.train_save
    
    if (epoch+1) % 5 == 0:
        model.eval()
        average, avg_dice, avg_iou, avg_acc, avg_recall, avg_precision, avg_f2, avg_mae = validation(model, device, opt.results_save_place, opt.test_path)
        if avg_dice > Best_dice:
            Average = average
            Best_dice = avg_dice
            Best_iou = avg_iou
            Best_acc = avg_acc
            Best_recall = avg_recall
            Best_precision = avg_precision
            Best_f2 = avg_f2
            Best_mae = avg_mae
            torch.save(model.state_dict(), save_path)
            print('[Saving parameter:]', save_path)
        model.train()


def get_metrics(pred, mask):
    pred = (pred > 0.5).float()
    pred_positives = pred.sum(dim=(1, 2))
    mask_positives = mask.sum(dim=(1, 2))
    inter = (pred * mask).sum(dim=(1, 2))
    union = pred_positives + mask_positives
    dice = (2 * inter) / (union + 1e-6)
    iou = inter / (union - inter + 1e-6)
    acc = (pred == mask).float().mean(dim=(1, 2))
    recall = inter / (mask_positives + 1e-6)
    precision = inter / (pred_positives + 1e-6)
    f2 = (5 * inter) / (4 * mask_positives + pred_positives + 1e-6)
    mae = (torch.abs(pred - mask)).mean(dim=(1, 2))

    return dice, iou, acc, recall, precision, f2, mae


def validation(model, device, results_save_place, test_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    data_path = test_path + '/images/'
    save_path = results_save_place
    opt = parser.parse_args()

    os.makedirs(save_path, exist_ok=True)
    test_loader = test_dataset(data_path,  opt.testsize)

    for i in range(test_loader.size):
        image, size, name = test_loader.load_data()
        
        H, W = size
        image = image.to(device)
        

        #-----------------------------------------------------------------
        prediction = model(image)
        #-----------------------------------------------------------------
        

        
        res = F.upsample(prediction, size=(W, H), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res_binary = (res > 0.5).astype(np.uint8) 
        res_uint8 = res_binary * 255
        imageio.imwrite(save_path+name, res_uint8)
    #---------------------------------------------------------------
    gt_dir = test_path + "/masks"
    pred_dir = save_path
    all_dice = []
    all_iou = []
    all_acc = []
    all_recall = []
    all_precision = []
    all_f2 = []
    all_mae = []
    loader = EvalDataset(pred_dir,gt_dir)

    with torch.no_grad():
        for pred, gt in loader:

            dice, iou, acc, recall, precision, f2, mae = get_metrics(pred, gt)
            all_dice.append(dice.item())
            all_iou.append(iou.item())
            all_acc.append(acc.item())
            all_recall.append(recall.item())
            all_precision.append(precision.item())
            all_f2.append(f2.item())
            all_mae.append(mae.item())
        
        avg_dice = np.mean(all_dice)
        avg_iou = np.mean(all_iou)
        avg_acc = np.mean(all_acc)
        avg_recall = np.mean(all_recall)
        avg_precision = np.mean(all_precision)
        avg_f2 = np.mean(all_f2)
        avg_mae = np.mean(all_mae)
        average = (avg_dice + avg_iou + avg_acc + avg_recall + avg_precision + avg_f2) / 6
        print(f"##### Average Dice #####: {avg_dice:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Acc: {avg_acc:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average F2: {avg_f2:.4f}")
        print(f"Average Mae: {avg_mae:.4f}")
        print(f"Average: {average:.4f}")
        return average,avg_dice, avg_iou, avg_acc, avg_recall, avg_precision, avg_f2, avg_mae

Average = 0
Best_dice = 0
Best_iou = 0
Best_acc = 0
Best_recall = 0
Best_precision = 0
Best_f2 = 0
Best_mae = 0

# "CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB", "test"
# test_dataset_name = "CVC-300"
# test_dataset_name = "CVC-ColonDB"
# test_dataset_name = "Kvasir"
# test_dataset_name = "CVC-ColonDB"
# test_dataset_name = "ETIS-LaribPolypDB"
test_dataset_name = "test"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=3e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset/' + test_dataset_name, help='path to train dataset')
    # Preparameter 
    # best_parameter_HGM.pth
    parser.add_argument('--pre_parameter', type=str, default="./best_parameter_HGM.pth", help='Load model from a .pth file')
    # Save Checkpoint
    parser.add_argument('--train_save', type=str,default='./data/test.pth')
    # Verification result save address
    parser.add_argument('--results_save_place', type=str, default='./data/results/'+ test_dataset_name +'_01/')
    opt = parser.parse_args()
    
    torch.backends.cudnn.enabled = False

    # ---- build models ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGM()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device=device)


    if opt.pre_parameter:
        # Our best parameter trained with two GPU
        print("############# Use best Parameter #############")
        weights_dict = torch.load(opt.pre_parameter, map_location=device)
        model_state_dict = model.state_dict()
        # if you want to use one GPU, please use this:
        #cleaned_weights_dict = {name.replace('module.', ''): param for name, param in weights_dict.items()}
        for name, param in weights_dict.items():
            if name in model_state_dict and model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
            else:
                print(f"Skipped loading parameter: {name} - Shape mismatch or not found")
        model.load_state_dict(model_state_dict)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    print("Number of batch: {} ".format(total_step))
    print("#"*20, "Start Training", "#"*20)

    

    for epoch in range(0, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, device, opt)
    
    print("-----------------------------------------")
    print(f"####### Best Average Dice #######: {Best_dice:.4f}")
    print(f"Best Average IoU: {Best_iou:.4f}")
    print(f"Best Average Acc: {Best_acc:.4f}")
    print(f"Best Average Recall: {Best_recall:.4f}")
    print(f"Best Average Precision: {Best_precision:.4f}")
    print(f"Best Average F2: {Best_f2:.4f}")
    print(f"Best Average Mae: {Best_mae:.4f}")
    print(f"Best Average: {Average:.4f}")


