# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=0, path='chkpnt.pth'):
        """Early stopping implementation

        Parameters
        ----------
        patience : int, optional
            How long to wait after last time validation loss improved,
            by default 3
        verbose : bool, optional
            If True, prints a message for each validation loss improvement,
            by default True
        delta : int, optional
            Minimum change in the monitored quantity to qualify 
            as an improvement, by default 0
        path : str, optional
            Path for the checkpoint to be saved to., by default 'chkpnt.pth'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # noqa: E501
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            if self.verbose:
                print(
                    'Validation loss decreased %.3f --> %.3f'
                    % (
                        self.val_loss_min, val_loss
                    )
                )
            self.val_loss_min = val_loss
            self.counter = 0
