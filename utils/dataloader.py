import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class PolypDataset_ASPS(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize // 4, self.trainsize // 4)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class PolypDataset_storage(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize):
        self.true_images = []
        self.true_gts = []
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        

    def __getitem__(self, index):
        # image = self.rgb_loader(self.images[index])
        # gt = self.binary_loader(self.gts[index])
        

        image = self.true_images[index]
        gt = self.true_gts[index]
        # print(np.array(image).shape)
        # print(np.array(gt).shape)
        # (1080, 1280, 3)
        # (1080, 1280)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # print(image.shape)
        # print(gt.shape)
        # torch.Size([3, 352, 352])
        # torch.Size([1, 352, 352])
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        true_images = []
        true_gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                true_images.append(img)
                true_gts.append(gt)
        self.images = images
        self.gts = gts
        self.true_images=true_images
        self.true_gts=true_gts
        self.size = len(self.true_images)
        print(self.size)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        # print(np.array(image).shape)
        # print(np.array(gt).shape)
        # (1080, 1280, 3)
        # (1080, 1280)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # print(image.shape)
        # print(gt.shape)
        # torch.Size([3, 352, 352])
        # torch.Size([1, 352, 352])
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader_ASPS(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = PolypDataset_ASPS(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = PolypDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_loader_storage(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = PolypDataset_storage(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_loader_patch(image_root, gt_root, batchsize, trainsize, offset, shuffle=True, num_workers=4, pin_memory=True):

    dataset = PolypDataset_Patch(image_root, gt_root, trainsize, offset)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class PolypDataset_Patch(data.Dataset):
    """
    Dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize=352, offset=308):
        self.patch_size = trainsize
        self.offset = offset
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.filter_files()
        
        # 计算所有patch的数量
        self.all_patches = self.calculate_all_patches()
        
    def calculate_all_patches(self):
        all_patches = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = self.rgb_loader(img_path)
            gt = self.binary_loader(gt_path)
            # img = Image.open(img_path)
            # gt = Image.open(gt_path)
            patches = self.get_patches(img, gt)
            all_patches.extend(patches)  # 收集所有patch
        return all_patches

    def get_patches(self, img, gt):
        # patches = []
        # img_np = np.array(img)
        # gt_np = np.array(gt)
        
        # # Get image dimensions
        # h, w = img_np.shape[:2]
        
        # # Iterate over the image with the specified offset and patch size
        # for i in range(0, h - self.patch_size + 1, self.offset):
        #     for j in range(0, w - self.patch_size + 1, self.offset):
        #         img_patch = img_np[i:i + self.patch_size, j:j + self.patch_size,:]
        #         gt_patch = gt_np[i:i + self.patch_size, j:j + self.patch_size]
                
        #         # Ensure there is enough foreground in the patch
        #         #if np.count_nonzero(gt_patch) >= self.patch_size * self.patch_size * 0.1:  # At least 10% foreground
        #         patches.append((Image.fromarray(img_patch), Image.fromarray(gt_patch)))
        

        patches = []

        img_np = np.array(img)
        gt_np = np.array(gt)
        
        h1, w1 = img_np.shape[:2]
        
        for i in range(0, h1, self.offset):
            for j in range(0, w1, self.offset):
                img_patch = img_np[i:i + self.patch_size, j:j + self.patch_size,:]
                gt_patch = gt_np[i:i + self.patch_size, j:j + self.patch_size]
                if img_patch.shape[0] * img_patch.shape[1] == self.patch_size * self.patch_size:
                    patches.append((Image.fromarray(img_patch), Image.fromarray(gt_patch)))
                else:
                    H, W,  _ = img_patch.shape
                    W_size = max(self.patch_size, W)
                    H_size = max(self.patch_size, H)
                    img_patch = np.pad(img_patch,((0, H_size - H),  (0, W_size - W), (0, 0)))
                    gt_patch = np.pad(gt_patch,((0, H_size - H),  (0, W_size - W)))
                    patches.append((Image.fromarray(img_patch), Image.fromarray(gt_patch)))
        return patches




    def __getitem__(self, index):
        # 获取所有patch
        img_patch, gt_patch = self.all_patches[index]
        
        # 转换为tensor
        image_tensor = self.img_transform(img_patch)
        gt_tensor = self.gt_transform(gt_patch)
        
        return image_tensor, gt_tensor

    def __len__(self):
        return len(self.all_patches)  # 返回所有patch的数量

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')




class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.images = sorted(self.images)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        size = image.size #（1250,1080）
        
        image = self.transform(image).unsqueeze(0)#image进行了翻转
        #（1080,1250）
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, size, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class test_dataset_patch:
    def __init__(self, image_root, testsize=352, offset=308):
        """
        Dataset for testing, with patch extraction.
        
        :param image_root: Path to image folder
        :param testsize: Target test image size (usually for resizing)
        :param patch_size: Size of the patches to extract
        :param offset: Step size to extract patches
        """
        self.patch_size = testsize
        self.offset = offset
        
        # Get all image paths
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        
        self.size = len(self.images)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        patches = self.get_patches(image)
        
        # Assuming we return only the first patch for simplicity; you can modify this to return all patches.
        image_patch = patches  # Get the first patch
        
        # Apply transformations
        
        image_patch = [self.transform(image) for image in image_patch]
        image_patch = torch.stack(image_patch)
        #print(image_patch.shape)
        
        # Get the original size and the name of the image
        size = image.size  # (1250, 1080)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image_patch, size, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def get_patches(self, img):
        """
        Extract patches from the image.
        
        :param img: The input image
        :return: List of patches extracted from the image
        """
        patches = []
        positions = []
        img_np = np.array(img)
        
        h1, w1 = img_np.shape[:2]
        
        for i in range(0, h1, self.offset):
            for j in range(0, w1, self.offset):
                img_patch = img_np[i:i + self.patch_size, j:j + self.patch_size,:]
                if img_patch.shape[0] * img_patch.shape[1] == self.patch_size * self.patch_size:
                    patches.append(Image.fromarray(img_patch))
                else:
                    H, W,  _ = img_patch.shape
                    W_size = max(self.patch_size, W)
                    H_size = max(self.patch_size, H)
                    img_patch = np.pad(img_patch,((0, H_size - H),  (0, W_size - W), (0, 0)))
                    patches.append(Image.fromarray(img_patch))

        return patches

    def __len__(self):
        return self.size
    
    def restore_image_from_patches(self, patches, img_size):
        """
        Restore the full image from the extracted patches.
        
        :param patches: List of patches (after model output)
        :param positions: List of (i, j) coordinates for each patch
        :param img_size: The original image size (height, width)
        :param patch_size: Size of each patch (assumed square)
        
        :return: Restored full image
        """
        #(9, 1, 352, 352)
        #patches = [img for img in patches]

        # Create an empty array to store the restored image
        restored_img = np.zeros(img_size)
        h1, w1 = img_size #(1080,1250)
        
        # Create a count map to handle overlapping areas
        count_map = np.zeros(img_size)
        count = 0 

        for i in range(0, h1, self.offset ):
            for j in range(0, w1, self.offset ):
                restored_img_patch = restored_img[i : i + self.patch_size, j : j + self.patch_size]
                if restored_img_patch.shape[0] * restored_img_patch.shape[1] == self.patch_size * self.patch_size:
                    restored_img[i : i + self.patch_size, j : j + self.patch_size] += patches[count,0,:,:]
                    count_map[i : i + self.patch_size, j : j + self.patch_size] += 1
                else:
                    H, W = restored_img_patch.shape
                    W_size = min(self.patch_size, W)
                    H_size = min(self.patch_size, H)
                    prediction2 = patches[count, 0, :H_size, :W_size]
                    # print("prediction1:",prediction1.shape)
                    # print("prediction_like:",prediction_like[i:i + patch_size, j:j + patch_size, k:k + patch_size, :].shape)
                    restored_img[i : i + self.patch_size, j : j + self.patch_size] += prediction2
                    count_map[i : i + self.patch_size, j : j + self.patch_size] += 1
                count += 1


        # # Iterate over each patch and its position
        # for patch, (i, j) in zip(patches, positions):
        #     patch = np.array(patch.squeeze(0))  # Ensure patch is in numpy array format
        #     h, w = patch.shape[:2]
            
        #     # Place the patch in the correct position in the restored image
        #     restored_img[i:i + h, j:j + w] += patch
        #     count_map[i:i + h, j:j + w] += 1  # Increment count in the count map
        
        # Avoid division by zero by handling the case where a patch wasn't added to a certain pixel
        count_map[count_map == 0] = 1
        
        # Normalize the restored image (to handle overlapping patches)
        restored_img /= count_map
        
        return restored_img


        
class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        pred_file = os.listdir(pred_root)
        label_file = os.listdir(label_root)

        name_list = []
        for iname in pred_file:
            if iname in label_file:
                name_list.append(os.path.join(iname))

        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), name_list))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), name_list))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        # print(pred.size)
        # print(gt.size)
        
        pred = transforms.ToTensor()(pred)
        gt = transforms.ToTensor()(gt)#转化为tensor时候，会变成0到1.0的tensor
        # print(pred.shape)
        # print(gt.shape)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
