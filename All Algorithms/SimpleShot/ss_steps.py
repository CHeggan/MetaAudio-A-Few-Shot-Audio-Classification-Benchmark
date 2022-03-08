"""
File contains the meta step function used in the SimpleShot validation and testing 
    procedures.
Works like:
    -> Get feature output from model
    -> Use some transform on the ouput features
    -> Use euclidean distance to classify
"""
################################################################################
# IMPORTS
################################################################################
import sys
import torch
import numpy as np

from transforms import l2_norm, cl2n
from metric import catagorical_accuracy

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def catagorical_accuracy(targets, predictions):
    predictions = predictions.argmax(dim=1).view(targets.shape)
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

def vote_catagorical_acc(targets, predictions):
    return (predictions == targets).sum().float() / targets.size(0)
    
################################################################################
# SS META STEP FOR FIXED LENGTH
################################################################################
def ss_eval_step_fixed(model, params, loss_fn, avg_features, x, y, k_shot, n_way, q_queries, device):
    query_losses = []
    query_accs = []

    with torch.no_grad():
        for idx in range(x.shape[0]):

            meta_batch = x[idx]
            y_batch = y[idx]

            logits, feature_embeddings = model(meta_batch, features=True)

            # k lots of n support samples from a particular class
            # k lots of q query samples from those classes
            x_support = feature_embeddings[:k_shot*n_way]
            y_support = y_batch[:k_shot*n_way]

            x_queries = feature_embeddings[q_queries*n_way:]
            y_queries = y_batch[q_queries*n_way:]

            # Apply whatever transforms we want
            if params['training']['transform'] == 'L2N':
                x_support = l2_norm(x_support)
                x_queries = l2_norm(x_queries)

            elif params['training']['transform'] == 'CL2N':
                x_support = cl2n(avg_features, x_support)
                x_queries = cl2n(avg_features, x_queries)

            x_queries = x_queries.to(device)
            x_support = x_support.to(device)

            # In multishot, simpleshot uses nearest centroid pr prototype
            prototypes = compute_prototypes(x_support, n_way, k_shot)

            # Calculate squared distances between all queries and all prototypes
            # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
            distances = pairwise_distances(x_queries, prototypes, 'l2')

            # Prediction probabilities are softmax over distances
            y_pred = (-distances).softmax(dim=1)
            loss = loss_fn(y_pred, y_queries)

            acc = catagorical_accuracy( y_queries, y_pred )
            
            # Store values
            query_losses.append(loss.item())
            query_accs.append(acc.item())


    return np.mean(query_losses), np.mean(query_accs), query_accs

################################################################################
# SS META STEP FOR VARIABLE LENGTH
################################################################################
def ss_eval_step_var(model, params, loss_fn, avg_features, x_support, x_query, q_num,
    y, k_shot, n_way, q_queries, device):
    # Sets up metric tracking
    query_losses = []
    query_accs = []

    # Kepe track of the last used q_num access idx
    last_query_idx = 0
    # Iterate over num tasks
    with torch.no_grad():
        for idx in range(x_support.shape[0]):

            x_task_train = x_support[idx]

            sub_q_num = q_num[idx*(n_way*q_queries): (idx+1)*(n_way*q_queries)]
            q_num_sub_sum = sum(sub_q_num)
            x_task_val = x_query[last_query_idx: (last_query_idx + q_num_sub_sum)]
            # Update tracking index
            last_query_idx += q_num_sub_sum

            # y value access is same as a fixed length batching
            y_task_train = y[idx][:(n_way * k_shot)]
            y_task_val = y[idx][(n_way * q_queries):]

            _, support_emb = model(x_task_train, features=True)
            _, query_emb = model(x_task_val, features=True)

            # Deals with transforms for SimpleShot
            # Apply whatever transforms we want
            if params['training']['transform'] == 'L2N':
                support_emb = l2_norm(support_emb)
                query_emb = l2_norm(query_emb)

            elif params['training']['transform'] == 'CL2N':
                support_emb = cl2n(avg_features, support_emb)
                query_emb = cl2n(avg_features, query_emb)

                query_emb = query_emb.to(device)
                support_emb = support_emb.to(device)

            # Calculates prototypes of support vectors
            prototypes = compute_prototypes(support_emb, n_way, k_shot)

            # Decide what distance function to use based on algorithm, we use l2 for SS
            distance = 'l2'

            # Calculate squared distances between all queries and all prototypes
            # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
            logits = -(pairwise_distances(query_emb, prototypes, distance))

            # We scale up the y task val to directly compare for loss but use majority for acc
            sub_q_nums_tens = torch.tensor(sub_q_num).to(device)
            scaled_up_query_y = torch.repeat_interleave(
                y_task_val, sub_q_nums_tens).to(device)
            query_loss = loss_fn(logits, scaled_up_query_y)

            soft_logits = logits.softmax(dim=1)
            query_pred = majority_vote(soft_logits, sub_q_nums_tens).to(
                device, dtype=torch.long)
            post_acc = vote_catagorical_acc(y_task_val, query_pred)


            # Store values
            query_losses.append(query_loss.item())
            query_accs.append(post_acc.item())


    return np.mean(query_losses), np.mean(query_accs), query_accs

##############################################################################
# PROTOTYPE FUNCTION
##############################################################################
def compute_prototypes(support, n, k):
    """Compute class prototypes from support samples.

    Args:
        support (torch.Tensor): Tensor of shape (n * k, d) where d is the embedding
            dimension.
        n (int): number of classes in the classification task
        k (int): Number of support examples per class

    Returns:
        torch.Tensor: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(n, k, -1).mean(dim=1)
    return class_prototypes

##############################################################################
# PAIRWISE DISTANCE CALCULATOR
##############################################################################
def pairwise_distances(x, y, matching_fn):
    """Efficiently calculate pairwise distances (or other similarity scores) between
        two sets of samples.

    Args:
        x (torch.Tensor): Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y (torch.Tensor): Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn (str): [Distance metric/similarity score to compute between samples

    Returns:
        torch.Tensor: Distances between ueries and teh class prototypes
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + sys.float_info.epsilon)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + sys.float_info.epsilon)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


        