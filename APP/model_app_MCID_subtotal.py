import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split


class DeepLearningArchitecture(nn.Module):

    def __init__(self, dropout1_p=0.0, dropout2_p=0.5, dropout3_p=0.4, clustering_neurons=10, penultimate_activation_type='relu',
                 input_neurons=171):
        """
        input_layer (X features) --> dropout1 --> fc1(X in, 64 out) --> relu --> dropout2 --> fc2(64 in, 32 out) --> relu
            --> dropout3 --> fc3(32 in, Y out) --> penultimate_activation_function --> fc4(Y in, 1 out) --> sigmoid

        dropout1_p: p-value for dropout1
        dropout2_p: p-value for dropout2
        dropout3_p: p-value for dropout3
        clustering_neurons (Y): number of neurons in clustering layer (penultimate layer right infront of the one-neuron classification layer)
        penultimate_activation_type: 'relu', 'softmax', 'gumbel_softmax'
        input_neurons (X): number of neurons in input layer
        """
        super(DeepLearningArchitecture, self).__init__()

        self.section1 = nn.Sequential(
            nn.Dropout(p=dropout1_p),  # Dropout for input features

            nn.Linear(in_features=input_neurons, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=dropout2_p),  # Dropout layer

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=dropout3_p),  # Dropout layer

            nn.Linear(in_features=32, out_features=clustering_neurons),
        )

        self.penultimate_activation_type = penultimate_activation_type

        self.section2 = nn.Sequential(
            nn.Linear(in_features=clustering_neurons, out_features=1),
            nn.Sigmoid(),
        )

    # Default forward training function

    def forward(self, x):

        x2 = self.section1(x)

        # choose which activation function to use in penultimate layer
        if self.penultimate_activation_type == 'relu':
            x3 = F.relu(x2)
        elif self.penultimate_activation_type == 'softmax':
            x3 = F.softmax(x2, dim=-1)
        elif self.penultimate_activation_type == 'gumbel_softmax':
            x3 = F.gumbel_softmax(x2, hard=True)
        elif self.penultimate_activation_type == 'sigmoid':
            x3 = torch.sigmoid(x2)

        out = self.section2(x3)

        return out


def get_probs(input_x, model_optimized):
    input_dim = input_x.shape[-1]
    model_optimized.eval()
    logits = model_optimized(input_x.float())
    new_logits = torch.zeros((len(logits), 2))
    new_logits[:, 1] = logits.squeeze()
    new_logits[:, 0] = torch.ones_like(logits).squeeze() - logits.squeeze()
    return new_logits


def get_confidence(input_x, model_optimized):
    input_dim = input_x.shape[-1]
    model_optimized.eval()
    logits = model_optimized(input_x.float())
    new_logits = torch.zeros((len(logits), 2))
    new_logits[:, 1] = logits.squeeze()
    new_logits[:, 0] = torch.ones_like(logits).squeeze() - logits.squeeze()
    return torch.max(new_logits, -1)


def main():
    # input data
    time_len = '6month'  # 6month, 1year, 2year
    pwd = '/home/wenqi/GL-SMART-aim2'
    raw_train_data = pd.read_csv(
        pwd+'/pre-processed-data-MCID/pp_data_preop_{}.csv'.format(time_len))
    # print('Train Dataset: %s' % (raw_train_data.shape,))
    feature_list = raw_train_data.columns[:172]
    x = raw_train_data[feature_list]
    x = x.drop('PID', axis='columns')
    problem_id = 13  # 0-Function; 1-Mental Health; 2-Pain; 3-Satisfication; 4-Self-Image; 8-Total; 13-Subtotal
    focus_id = problem_id + 201
    focus_label = raw_train_data.columns[focus_id]
    y = raw_train_data[focus_label]
    # patient selection
    patient_index = 0
    x_test = torch.tensor(x.values)
    # single patient
    X_test_single = x_test[patient_index]
    # multiple patients
    X_test_multiple = x_test[patient_index:patient_index+100]

    # running args
    dropout1_p, dropout2_p, dropout3_p = 0.0, 0.0, 0.0
    model_variant = 'relu'
    clustering_neurons = 128
    learning_rate = 0.01
    outcome_variable = 'death90+vent'
    models_folder = pwd+'/deep_model_MCID/deep-models-{}/{}/'.format(
        time_len, focus_label)
    model_name = 'ALLFEATURES_ACTIVATION={}_CN={}_D1P={}_D2P={}_D3P={}_LR={}_OUTCOME={}'.format(
        model_variant, clustering_neurons, dropout1_p, dropout2_p, dropout3_p, learning_rate, outcome_variable)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check that model with these specific hyperparameters has not already been trained
    save_path = '{}{}_checkpoint.pth'.format(models_folder, model_name)
    model_optimized = DeepLearningArchitecture(dropout1_p=dropout1_p, dropout2_p=dropout2_p, dropout3_p=dropout3_p,
                                               clustering_neurons=clustering_neurons,
                                               penultimate_activation_type=model_variant)
    # a = torch.load(save_path)
    model_optimized.load_state_dict(torch.load(save_path))
    model_optimized.to(device)

    # Conduct single patient prediction. Use cpu for model inference.
    PRO_score = get_probs(X_test_single.to(
        device), model_optimized)  # model output
    PRO_score = PRO_score.detach().cpu().numpy().tolist()
    PRO_conf = get_confidence(X_test_single.to(device), model_optimized)
    # Output
    PRO_confidence = PRO_conf.values.detach().cpu().numpy().tolist()
    PRO_preds = PRO_conf.indices.detach().cpu().numpy().tolist()
    # print('==================================================')
    print('Class 0: Negative; Class 1: Positive. The prediction categories are: {}; The predicted confidences are: {}.'.format(
        PRO_preds, PRO_confidence))

    # Conduct multiple patient prediction. Use gpu for model inference.
    # PRO_score = get_probs(X_test_multiple.to(
    #     device), model_optimized)  # model output
    # PRO_score = PRO_score.detach().cpu().numpy().tolist()
    # PRO_conf = get_confidence(X_test_multiple.to(device), model_optimized)
    # PRO_confidence = PRO_conf.values.cpu().detach().numpy().tolist()
    # PRO_preds = PRO_conf.indices.cpu().detach().numpy().tolist()
    # print('==================================================')
    # print('Class 0: Negative; Class 1: Positive. The prediction probabilities for each class are: {}; The predicted confidences are: {}; The prediction categories are: {}.'.format(
    #     PRO_score, PRO_confidence, PRO_preds))

    # print(y.values)


if __name__ == '__main__':
    main()
