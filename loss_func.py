import torch.nn as nn

class CRF_Loss(nn.Module):
    ''' Loss function for BiLSTM-CRF '''
    def __init__(self):
        super(CRF_Loss, self).__init__()
    
    def forward(self, model, inputs, outputs, labels, training=True):
        text = inputs[0]
        text_mask = (text!=0) # text mask for variable length
        loss = model.crf(outputs, labels, mask=text_mask) # compute negative log likelihood using the crf layer
        if training:
            return loss
        else:
            predicts = model.crf.decode(outputs, mask=text_mask) # get predicted labels using viterbi decode
            return loss, predicts
    
