## Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single GeForce RTX 3090 GPU of 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that BiMamTrans is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n BiMamTrans python=3.10`.
    
    + Installing necessary packages: `pip install -r requirements.txt `.
    
2. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [Google Drive Link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [Google Drive Link (399.5MB)](https://drive.google.com/file/d/13Zij1HbKTn9PKTW9bM19_fXtfQKWdCtD/view?usp=sharing). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).
    
    + downloading version 1 pretrained weights and move it into `./best_parameter_BiMamTrans.pth`, 
    which can be found in this [Google Drive Link (101.3MB)](https://drive.google.com/file/d/13Zij1HbKTn9PKTW9bM19_fXtfQKWdCtD/view?usp=sharing).
    
    + downloading version 2 pretrained weights and move it into `./best_parameter_BiMamTransV2.pth`, 
    which can be found in this [Google Drive Link (102MB)](https://drive.google.com/file/d/17WAvw_hoXfPZsE7KlKOkFcu2jWuW7L5p/view?usp=sharing).

3. Training Configuration:

    + Assigning your costumed path, like `--train_save`, `--train_path`, `--train_save`, and `--results_save_place` in `MyTrain.py`.
    
    + Just enjoy it!

4. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Just enjoy it!
