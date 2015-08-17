



def on_epoch_finished(obj, train_history):
    if (len(train_history) > 20):
        inLastEight = any([h['valid_loss_best'] for h in  train_history[-7:-1]] +
                          [train_history[-1]['valid_loss'] < train_history[-2]['valid_loss'],
                           train_history[-1]['valid_loss'] < train_history[-3]['valid_loss'],
                           train_history[-1]['valid_loss'] < train_history[-4]['valid_loss'],
                           train_history[-2]['valid_loss'] < train_history[-3]['valid_loss'],
                           train_history[-2]['valid_loss'] < train_history[-4]['valid_loss'],
                           ])
        if not inLastEight:
            print("Stopping early")
            raise StopIteration
