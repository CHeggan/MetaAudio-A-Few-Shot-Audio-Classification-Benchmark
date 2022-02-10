import torch

###############################################################################
# CATAGORICAL ACC & VOTE BASED
###############################################################################
def catagorical_accuracy(y, y_pred):
    predictions = y_pred.argmax(dim=-1)
    correct = torch.eq(predictions, y).sum().item()
    return  correct/y_pred.shape[0]


def vote_catagorical_acc(targets, predictions):
    return (predictions == targets).sum().float() / targets.size(0)

def majority_vote(soft_logits, query_nums):
    y_preds = soft_logits.argmax(dim=1)

    end_index = 0
    aggregrated_preds = torch.zeros(len(query_nums))
    for idx, num in enumerate(query_nums):
        slice = y_preds[end_index:(end_index + num)]
        value, indices = torch.mode(slice)
        aggregrated_preds[idx] = value
        end_index += slice.shape[0]
    return aggregrated_preds
