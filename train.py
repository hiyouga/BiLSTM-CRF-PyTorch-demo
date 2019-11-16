import os
import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from models import BiLSTMCRF
from loss_func import CRF_Loss
from data_utils import MyDataset, build_tokenizer, build_embedding_matrix, compute_score

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt # hyperparameters and options
        opt.tokenizer = build_tokenizer(fnames=opt.dataset_file.values()) # transfrom tokens to indices
        embedding_matrix = build_embedding_matrix(vocab=opt.tokenizer.vocab['word']) # pre-trained glove embeddings
        self.trainset = MyDataset(fname=opt.dataset_file['train'], tokenizer=opt.tokenizer) # training set
        self.valset = MyDataset(fname=opt.dataset_file['val'], tokenizer=opt.tokenizer) # validation set
        self.testset = MyDataset(fname=opt.dataset_file['test'], tokenizer=opt.tokenizer) # testing set
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device) # neural network model
        self._print_args() # print arguments
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        if self.opt.device.type == 'cuda':
            print(f"cuda memory allocated: {torch.cuda.memory_allocated(self.opt.device.index)}")
        print(f"n_trainable_params: {int(n_trainable_params)}, n_nontrainable_params: {int(n_nontrainable_params)}")
        print('training arguments:')
        for arg in vars(self.opt):
            print(f">>> {arg}: {getattr(self.opt, arg)}")
    
    def _reset_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'word_embedding' in name: # don't initialize word embeddings while using pre-trained embeddings
                    pass
                elif 'embedding' in name: # treat embedding matrices as special cases
                    weight = opt.initializer(torch.empty(*param.shape))
                    weight[0] = 0. # the vector corresponding to padding index shuold be zero
                    setattr(param, 'data', weight)
                else:
                    if len(param.shape) > 1:
                        self.opt.initializer(param) # for weight matrices
                    else:
                        torch.nn.init.zeros_(param) # for bias vectors
    
    def _train(self, optimizer, criterion, dataloader):
        train_loss, n_train = 0, 0
        n_batch = len(dataloader)
        self.model.train() # switch model to training mode
        for i_batch, sample_batched in enumerate(dataloader): # mini-batch optimization
            inputs = [sample_batched[key].to(self.opt.device) for key in ['text', 'char']] # model inputs
            labels = sample_batched['label'].to(self.opt.device) # ground-truth labels
            outputs = self.model(inputs, training=True) # compute outputs
            
            optimizer.zero_grad() # clear gradient accumulators
            loss = criterion(self.model, inputs, outputs, labels, training=True) # compute loss for this batch
            loss.backward() # compute gradients through back-propagation
            optimizer.step() # update model parameters
            
            train_loss += loss.item() * len(outputs) # training loss
            n_train += len(outputs) # number of training samples
            ratio = int((i_batch+1)*50/n_batch) # process bar
            sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
            sys.stdout.flush()
        print()
        return train_loss / n_train
    
    def _evaluate(self, optimizer, criterion, dataloader):
        test_loss, n_test = 0, 0
        lengths_all, labels_all, predicts_all = None, None, None
        self.model.eval() # switch model to evaluation mode
        with torch.no_grad(): # turn off gradients
            for sample_batched in dataloader:
                inputs = [sample_batched[key].to(self.opt.device) for key in ['text', 'char']]
                labels = sample_batched['label'].to(self.opt.device)
                lengths, outputs = self.model(inputs, training=False)
                loss, predicts = criterion(self.model, inputs, outputs, labels, training=False)
                
                test_loss += loss.item() * len(outputs)
                n_test += len(outputs)
                lengths_all = np.concatenate((lengths_all, lengths.cpu().numpy()), axis=0) if lengths_all is not None else lengths.cpu().numpy()
                labels_all = np.concatenate((labels_all, labels.cpu().numpy()), axis=0) if labels_all is not None else labels.cpu().numpy()
                predicts_all = np.concatenate((predicts_all, predicts.cpu().numpy()), axis=0) if predicts_all is not None else predicts.cpu().numpy()
        f1 = compute_score(lengths_all, predicts_all, labels_all, self.opt.tokenizer) # compute the f1 score
        return test_loss / n_test, f1
    
    def run(self):
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg) # the optimizer
        criterion = CRF_Loss() # Loss function for BiLSTM-CRF
        train_dataloader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True) # training dataloader
        val_dataloader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False) # validation dataloader
        test_dataloader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False) # testing dataloader
        
        best_val_f1 = 0 # record the best f1 score on validation set
        for epoch in range(self.opt.num_epoch):
            train_loss = self._train(optimizer, criterion, train_dataloader) # train model
            val_loss, val_f1 = self._evaluate(optimizer, criterion, val_dataloader) # evaluate model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), 'best_model.pt') # save model parameters
            print(f"{100*(epoch+1)/self.opt.num_epoch:6.2f}% > loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val f1: {val_f1:.4f}")
        print('#' * 50)
        self.model.load_state_dict(torch.load('best_model.pt')) # load model parameters with best validation performance
        test_loss, test_f1 = self._evaluate(optimizer, criterion, test_dataloader) # test model
        print(f"Test f1: {test_f1:.4f}")
    

if __name__ == '__main__':
    
    model_classes = {
        'bilstmcrf': BiLSTMCRF
    }
    
    dataset_files = {
        'conll2003': {
            'train': os.path.join('datasets', 'train.txt'),
            'val': os.path.join('datasets', 'valid.txt'),
            'test': os.path.join('datasets', 'test.txt')
        }
    }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,    # default lr=0.01
        'adam': torch.optim.Adam,          # default lr=0.001
        'adamax': torch.optim.Adamax,      # default lr=0.002
        'asgd': torch.optim.ASGD,          # default lr=0.01
        'rmsprop': torch.optim.RMSprop,    # default lr=0.01
        'sgd': torch.optim.SGD,            # default lr=0.1
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bilstmcrf', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='conll2003', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--word_emb_dim', default=300, type=int)
    parser.add_argument('--char_emb_dim', default=25, type=int)
    parser.add_argument('--word_hidden_dim', default=100, type=int)
    parser.add_argument('--char_hidden_dim', default=25, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    opt = parser.parse_args()
    
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device(opt.device) if opt.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists('dats'):
        os.mkdir('dats')
    
    ins = Instructor(opt)
    ins.run()
