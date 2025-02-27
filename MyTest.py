import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

import imageio
from utils.dataloader import test_dataset
from utils.dataloader import test_dataset, EvalDataset
from lib.pvt import HGM

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
        print(test_dataset_name)
        print(f"##### Average Dice #####: {avg_dice:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Acc: {avg_acc:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average F2: {avg_f2:.4f}")
        print(f"Average Mae: {avg_mae:.4f}")
        print(f"Average: {average:.4f}")
        return average,avg_dice, avg_iou, avg_acc, avg_recall, avg_precision, avg_f2, avg_mae
    
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


# "CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB"
test_dataset_name = "CVC-300"
# test_dataset_name = "CVC-ClinicDB"
# test_dataset_name = "Kvasir"
# test_dataset_name = "CVC-ColonDB"
# test_dataset_name = "ETIS-LaribPolypDB"
# test_dataset_name = "test"




parser = argparse.ArgumentParser()
# best_parameter_HGM.pth
parser.add_argument('--pth_path', type=str, default='./data/best_parameter_HGM.pth')
parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset/' + test_dataset_name, help='path to train dataset')
parser.add_argument('--results_save_place', type=str, default='./data/results/'+ test_dataset_name +'_HGM/')
opt = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = HGM()

weights_dict = torch.load(opt.pth_path, map_location=device)
model_state_dict = model.state_dict()

cleaned_weights_dict = {name.replace('module.', ''): param for name, param in weights_dict.items()}

for name, param in cleaned_weights_dict.items():
    if name in model_state_dict and model_state_dict[name].shape == param.shape:
        model_state_dict[name].copy_(param)
    else:
        print(f"Skipped loading parameter: {name} - Shape mismatch or not found")

model.load_state_dict(model_state_dict)

model = model.to(device)
model.eval()
validation(model,device=device,results_save_place=opt.results_save_place, test_path=opt.test_path)