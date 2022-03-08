###############################################################################
# CATAGORICAL ACCURACY
###############################################################################
def catagorical_accuracy(targets, predictions):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)