import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

## Torch ##
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import StratifiedKFold #StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score, accuracy_score

## Utils ##
from deeputils.feature_name_map import feature_name_dict
from deeputils.DeepLearningArchitecture import DeepLearningArchitecture
from sklearn.metrics import roc_auc_score
from deeputils.TensorboardSetup import TensorboardSetup
from deeputils.EarlyStopping import EarlyStopping
from deeputils.dataloading_pipeline import Anderson_Dataset, Anderson_Sampler

## Metrics ##
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import argparse

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        # softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(logits, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

### Use best Device (CUDA vs CPU) ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
## Print Device Properties ##
if device == torch.device('cuda'):
    print( torch.cuda.get_device_properties( device ) )


############################################# Command Line Arguments (for HP tuning purposes) ######################################
# if len(sys.argv) != 8 and len(sys.argv) != 9:
#     raise Exception('Wrong number of command line args.')
parser = argparse.ArgumentParser()
parser.add_argument("--model_variant", default='relu', type=str, help="The initial learning rate for Adam.")
parser.add_argument("--clustering_neurons", default=16, type=int, help="Batch size for training.")
parser.add_argument("--dropout1_p", default=0.5, type=float, help="Batch size for training.")
parser.add_argument("--dropout2_p", default=0.5, type=float, help="Number of epochs for training.")
parser.add_argument("--dropout3_p", default=0.5, type=float, help="Number of epochs for training.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="dataset")
parser.add_argument("--gpu_device", default=0, type=int, help="Weight decay if we apply some.")
parser.add_argument("--problem_id", default=0, type=int, help="Weight decay if we apply some.")
parser.add_argument("--time_len", default="6month", type=str, help="Weight decay if we apply some.")

# parser.add_argument('--saved_dataset', type=str, default='n', help='whether save the preprocessed pt file of the dataset')

args = parser.parse_args()
print(args)
outcome_variable = 'xixi'
model_variant = args.model_variant
clustering_neurons = args.clustering_neurons
dropout1_p = args.dropout1_p
dropout2_p = args.dropout2_p
dropout3_p = args.dropout3_p
learning_rate = args.learning_rate
gpu_device = args.gpu_device
# model_variant = str(sys.argv[1]) # penultimate activation function
# clustering_neurons = int(sys.argv[2])          # number of neurons in clustering layer
# dropout1_p = float(sys.argv[3])                # p for dropout for input layer
# dropout2_p = float(sys.argv[4])                # p for dropout after first hidden layer
# dropout3_p = float(sys.argv[5])                # p for dropout after second hidden layer 
# learning_rate = float(sys.argv[6])             # learning rate
# outcome_variable = str(sys.argv[7])            # outcome variable, should be 'death90+vent'
# if len(sys.argv) == 9:
#     gpu_device = int(sys.argv[8])              # identify which gpu to use
# else:
#     gpu_device = None


### Verify command line arguments are valid ###
if dropout1_p < 0 or dropout1_p > 1:
    raise Exception('Invalid dropout1_p')
if dropout2_p < 0 or dropout2_p > 1:
    raise Exception('Invalid dropout2_p')
if dropout3_p < 0 or dropout3_p > 1:
    raise Exception('Invalid dropout3_p')
if model_variant not in ['relu', 'softmax', 'gumbel_softmax', 'sigmoid']:
    raise Exception('Invalid model_variant')
if outcome_variable not in ['anyCatastrophic', 'vent', 'ICU', 'death30', 'death60', 'death90', 
                            'Admit30days', 'Admit60days', 'Admit90days', 'Admit7days', 'Admit14days', 'death90+vent']:
    outcome_variable = 'death90+vent'
if len(sys.argv) == 9 and gpu_device != 0 and gpu_device != 1 and gpu_device != 2 and gpu_device != 3:
    raise Exception('Invalid gpu number specified')


############################################# Setup #############################################
if gpu_device == None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device == torch.device('cuda'):
        print( torch.cuda.get_device_properties( device ) )
else:
    device = torch.device( ('cuda:'+str(gpu_device)) )
    print(device)
    print( torch.cuda.get_device_properties( device ) )
    
### Seed ###
random_state = 1234
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
## CUDNN ##
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

############################################# USER VARIABLES #############################################
## Retrain models even if output already exists ##
force_train=True

## Epoch Variables ##
patience = 200
n_epochs = 8000
batch_size = 64 # approximate number of observations (overshoot a little bit)

## Train/Test split ##
split_percentages = [0.8, 0.2]
## Number of Splits based on test set percent ##
n_splits = int(1/split_percentages[-1])

## Folder paths ##
raw_train_data = pd.read_csv('/home/wenqi/GL-SMART-aim2/pre-processed-data/pp_data_preop_{}.csv'.format(args.time_len))
print('Train Dataset: %s'%(raw_train_data.shape,) )
feature_list = raw_train_data.columns[:172]
x = raw_train_data[feature_list]
x = x.drop('PID', axis='columns')
focus_id = args.problem_id + 201
focus_label = raw_train_data.columns[focus_id]
y = raw_train_data[focus_label]
print(np.sum(y.values))
weight_num = np.sum(y.values) / len(y.values)
weight = torch.tensor([weight_num, 1 - weight_num])
print(weight)
file_path = os.path.join('/home/wenqi/GL-SMART-aim2/calibration/deep-models-{}/'.format(args.time_len), focus_label+'/')
if not os.path.exists(file_path):
    os.makedirs(file_path)
    os.makedirs(os.path.join(file_path,'results/'))
    os.makedirs(os.path.join(file_path,'tensorboards/'))
data_path = '/home/wenqi/GL-SMART-aim2/pre-processed-data/pp_data_preop_{}.csv'.format(args.time_len)
models_folder='/home/wenqi/GL-SMART-aim2/calibration/deep-models-{}/{}/'.format(args.time_len, focus_label)
tensorboard_folder = '/home/wenqi/GL-SMART-aim2/calibration/deep-models-{}/{}/tensorboard/'.format(args.time_len, focus_label)

## Create results spreadsheet if it does not already exist
results_file = '/home/wenqi/GL-SMART-aim2/calibration/deep-models-{}/{}/results/Validation-80Trained_HPTuneSpreadsheet.csv'.format(args.time_len, focus_label)
if not os.path.exists(results_file): 
    HPTuneSpreadsheet = pd.DataFrame(columns=['feature_set', 
                                              'penultimate_activation_type', 
                                              'clustering_neurons', 
                                              'dropout1_p', 'dropout2_p', 'dropout3_p', 
                                              'learning_rate', 'outcome_variable', 
                                              'AUROC', 'MCC', 'ECE'])
    HPTuneSpreadsheet.to_csv(results_file, index=False)



############################################# Model and Loss Function #############################################
model_name = 'ALLFEATURES_ACTIVATION={model_variant}_CN={clustering_neurons}_D1P={dropout1_p}_D2P={dropout2_p}_D3P={dropout3_p}_LR={learning_rate}_OUTCOME={outcome_variable}'

## Check that model with these specific hyperparameters has not already been trained
if( os.path.exists('{models_folder}{model_name}_checkpoint.pth') and not force_train):
    raise Exception('Already exists. The model with these hyperparameters has already been trained.')
else:
    print(model_name)
    print()

## Create Model ##
model = DeepLearningArchitecture(dropout1_p=dropout1_p, 
                                 dropout2_p=dropout2_p, 
                                 dropout3_p=dropout3_p,
                                 clustering_neurons=clustering_neurons,
                                 penultimate_activation_type=model_variant)
model.to(device)

## Loss and Optimizer ##
loss = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


############################################# Load Data #############################################

## Training Dataset ##

# print(y)
# input()
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

x_train = torch.tensor(x_train_val.values)
y_train = torch.tensor(y_train_val.values)
train_data = TensorDataset(x_train, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
print('Train Dataset: %s'%(x_train_val.shape,) )

x_valid = torch.tensor(x_test.values)
y_valid = torch.tensor(y_test.values)
valid_data = TensorDataset(x_valid, y_valid)
validation_dataset = valid_data
valid_sampler = RandomSampler(valid_data)
validation_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)
print('Valid Dataset: %s'%(x_test.shape,) )
############################################# TRAINING #################################################
########################################################################################################
## Tensorboard ##
tb = TensorboardSetup(foldername=model_name, runs_dir=tensorboard_folder)
tb.startService()

## Early Stopping on AUROC ##
early_stopping = EarlyStopping(patience=patience, verbose=False, path='{models_folder}{model_name}_checkpoint.pth',
                              improvement='ascending')
# weight = torch.tensor([0.33, 0.66])
criterion2 = _ECELoss(n_bins=10)
## Iterate across Epochs to Train model ##
for epoch in range(1, n_epochs+1):
    ## Epoch Start Time ##
    epoch_start = datetime.now()
    
    ## Initialize losses (sum of all samples) ##
    loss_train_epoch = torch.tensor([0.], requires_grad=True).to(device)
    loss_valid_epoch = 0
    
    ## AUROC Score ##
    auroc_train = 0
    auroc_valid = 0
    
    ### Training ###
    model.train()
    ## Iterate across image batches ##
    label_true_np = None
    label_pred_np = None
    for batch_indx, item in enumerate(train_dataloader):
        ## Assign item (dict) values ##
        batch = tuple(t.to(device) for t in item)
        features, label = batch
        features = features.float()
        label = label.float()
        # label = item['label'].float()

        ## Features/labels to GPU ##
        # features = features.to(device)
        # label = label.to(device)
        
        ## Predict ##        
        label_pred = model(features)
        label_pred = label_pred.reshape(-1)

        ## Loss: Batch Total ##
        # weight_ = weight[label.long()]
        loss_train = loss(label_pred, label)# * weight_.to(device)# sum of the BCE loss for all labels in batch
        # loss_train = loss_train.mean()
        ## Loss: Update Epoch Total ##
        loss_train_epoch += loss_train
        
        ## Loss: Backward Propagate, compute parameter gradients wrt. loss (once per batch) ##
        loss_train.backward()
        ## Optimizer: Update Parameters (once every batch) ##
        optimizer.step()
        ## Reset Gradients ##
        optimizer.zero_grad()

        if label_pred_np == None:
            label_pred_np = label_pred.cpu().detach()
            label_true_np = label.cpu().detach()
        else:
            label_pred_np = torch.cat((label_pred_np, label_pred.cpu().detach()), -1)
            label_true_np = torch.cat((label_true_np, label.cpu().detach()), -1)
        
    ## Performance Metrics ##
    label_true_np = label_true_np.numpy()
    label_pred_np = label_pred_np.numpy()
    auroc_train = roc_auc_score(label_true_np, label_pred_np)
    accuracy_train = accuracy_score(label_true_np, label_pred_np > 0.5)
    F1_train = f1_score(label_true_np, label_pred_np > 0.5)
    MCC_train = matthews_corrcoef(label_true_np, label_pred_np > 0.5)
        
        
    ### Validation ###
    model.eval()
    with torch.no_grad():
        label_true_np = None
        label_pred_np = None
        ## Iterate across image batches ##
        for batch_indx, item in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in item)
            features, label = batch
            features = features.float()
            label = label.float()

            ## Predict ##
            label_pred = model(features)
            label_pred = label_pred.reshape(-1)
            
            ## Loss: Batch Total ##
            # weight_ = weight[label.long()]
            loss_valid = loss(label_pred, label) #* weight_.to(device)# sum of the BCE loss for all labels in batch
            # loss_valid = loss_valid.mean()
            ## Loss: Update Epoch Total ##
            loss_valid_epoch += loss_valid

            if label_pred_np == None:
                label_pred_np = label_pred.cpu().detach()
                label_true_np = label.cpu().detach()
            else:
                label_pred_np = torch.cat((label_pred_np, label_pred.cpu().detach()), -1)
                label_true_np = torch.cat((label_true_np, label.cpu().detach()), -1)
  
        ## Performance Metrics ##
        logits = label_pred_np
        y_val = label_true_np.long()
        label_true_np = label_true_np.numpy()
        label_pred_np = label_pred_np.numpy()
        
        auroc_valid = roc_auc_score(label_true_np, label_pred_np)
        accuracy_valid = accuracy_score(label_true_np, label_pred_np > 0.5)
        F1_valid = f1_score(label_true_np, label_pred_np > 0.5)
        MCC_valid = matthews_corrcoef(label_true_np, label_pred_np > 0.5)
        # logits = torch.tensor(logits).unsequeeze(-1)
        new_logits = torch.zeros((len(logits), 2))
        new_logits[:,1] = logits
        new_logits[:,0] = torch.ones_like(logits) - logits
        # logits = torch.cat((logits, torch.ones_like(logits) - logits), -1)
        ece_valid = criterion2(new_logits, y_val)
        # print(ece_valid)
    ## Epoch Time (seconds) ##
    epoch_time= (datetime.now()-epoch_start).total_seconds()
            
    ### Log Epoch ###
    ## Loss: Ave Per Sample ##
    loss_train_ave = loss_train_epoch.cpu().detach().item() / len(train_sampler)
    loss_valid_ave = loss_valid_epoch.cpu().detach().item()  / len(validation_dataset)
    
    ## Log: Print ##
    print('Epoch {:<4}, Loss_Train:{:.4f}, Loss_Valid:{:.4f}, AUROC_Train:{:.5f}, AUROC_Valid:{:.5f}, ACC_Valid:{:.5f}, ECE_Valid:{:.5f}, Time (sec): {:.2f}'.format(
        epoch,
        loss_train_ave,
        loss_valid_ave,
        auroc_train,
        auroc_valid,
        accuracy_valid,
        ece_valid[0],
        epoch_time
       ) )
    
    ## Log: Tensorboard ##
    ## Training/Validation BCE Loss, accuracy, F1 score, MCC ##
    tb.writer.add_scalar('BCE Loss (train)', loss_train_ave, epoch)
    tb.writer.add_scalar('BCE Loss (valid)', loss_valid_ave, epoch)
    tb.writer.add_scalar('Accuracy (train)', accuracy_train, epoch)
    tb.writer.add_scalar('Accuracy (valid)', accuracy_valid, epoch)
    tb.writer.add_scalar('F1 Score (train)', F1_train, epoch)
    tb.writer.add_scalar('F1 Score (valid)', F1_valid, epoch)
    tb.writer.add_scalar('MCC (train)', MCC_train, epoch)
    tb.writer.add_scalar('MCC (valid)', MCC_valid, epoch)
    tb.writer.add_scalar('AUROC (train)', auroc_train, epoch)
    tb.writer.add_scalar('AUROC (valid)', auroc_valid, epoch)
    
    
    ## Early Stopping: AUROC Score ##
    early_stopping(auroc_valid, model)
    if early_stopping.early_stop:
        print("EARLY STOP")
        break


############################################# Analysis of Optimized Model (Load from EarlyStopping) #############################################
## Load EarlyStopping Saved Model ##
model_optimized = DeepLearningArchitecture(dropout1_p=dropout1_p, dropout2_p=dropout2_p, dropout3_p=dropout3_p,
                                           clustering_neurons=clustering_neurons,
                                           penultimate_activation_type=model_variant)
model_optimized.load_state_dict(torch.load('{models_folder}{model_name}_checkpoint.pth'))
model_optimized.to(device)
model_optimized.eval()

## Obtain Validation Performance ##
with torch.no_grad():
    ## Y_true, Y_pred ##
    label_true = None
    label_pred = None
    
    ## Iterate across Validation Batches ##
    for batch_indx, item in enumerate(validation_dataloader):
        print(batch_indx, '\n')
        
        batch = tuple(t.to(device) for t in item)
        features, label = batch
        features = features.float()
        label = label.float()

        ## Predict ##
        label_predd = model_optimized(features)
        label_predd = label_predd.reshape(-1)    

        if label_true == None:
            label_true = label
            label_pred = label_predd
        else:
            label_true = torch.cat((label_true, label), -1)
            label_pred = torch.cat((label_pred, label_predd), -1)

        
    ## Labels ##
    logits = label_pred
    y_test = label_true
    label_true = label_true.cpu().detach().numpy()
    label_pred = label_pred.cpu().detach().numpy()

    new_logits = torch.zeros((len(logits), 2))
    new_logits[:,1] = logits
    new_logits[:,0] = torch.ones_like(logits) - logits
    # logits = torch.cat((logits, torch.ones_like(logits) - logits), -1)
    ece_test = criterion2(new_logits.cpu().detach(), y_test.cpu().detach())


############################################# Update Results HPTune Spreadsheet #############################################
HPTuneSpreadsheet = pd.read_csv(results_file)
new_row_DF = pd.DataFrame(np.array([['All Features', model_variant, clustering_neurons, dropout1_p, dropout2_p, dropout3_p, 
                                   learning_rate, outcome_variable, roc_auc_score(label_true, label_pred), matthews_corrcoef(label_true, label_pred > 0.5), ece_test[0]]]), 
                          columns=['feature_set', 'penultimate_activation_type', 'clustering_neurons', 
                                   'dropout1_p', 'dropout2_p', 'dropout3_p', 
                                   'learning_rate', 'outcome_variable', 
                                   'AUROC', 'MCC', 'ECE'])
HPTuneSpreadsheet = pd.concat((HPTuneSpreadsheet, new_row_DF))
HPTuneSpreadsheet.to_csv(results_file, index=False)
    
    
