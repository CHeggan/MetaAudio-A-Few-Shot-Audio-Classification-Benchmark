"""
This file contains the following:
    -> Main proto net episodes for fixed and variable length sets
    -> Pairwise distance calculator
    -> Prototype calculator 
"""
##############################################################################
# IMPORTS
##############################################################################
import sys
import torch
import numpy as np
import torch.nn as nn
from typing import Callable
from torch.nn import Module
from torch.optim import Optimizer

from metrics import catagorical_accuracy, vote_catagorical_acc, majority_vote


##############################################################################
# FIXED PROTONET EPISODE
##############################################################################
def proto_step_fixed(model, optimiser, loss_fn, x, y, k_shot, n_way, q_queries,
                        distance, train):
    """Performs a single training episode for a Prototypical Network.
    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update
    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """

    losses = 0
    total_acc = 0
    
    all_post = []
        
    if train:
        model.train()
    else:
        model.eval()

    for idx in range(x.shape[0]):
        optimiser.zero_grad()

        meta_batch = x[idx]
        y_batch = y[idx]

        # Embed all samples
        embeddings = model(meta_batch)

        # Samples are ordered by the NShotWrapper class as follows:
        # k lots of n support samples from a particular class
        # k lots of q query samples from those classes
        support = embeddings[:k_shot*n_way]

        x_queries = embeddings[q_queries*n_way:]
        y_queries = y_batch[q_queries*n_way:]

        prototypes = compute_prototypes(support, n_way, k_shot)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = pairwise_distances(x_queries, prototypes, distance)

        # Calculate log p_{phi} (y = k | x)
        log_p_y = (-distances).log_softmax(dim=1)
        loss = loss_fn(log_p_y, y_queries)

        # Prediction probabilities are softmax over distances
        y_pred = (-distances).softmax(dim=1)

        acc = catagorical_accuracy( y_queries, y_pred )

        if train:
            losses += loss
        else:
            losses += loss.item()
        total_acc += acc

        all_post.append(acc)


    back_loss = losses / x.shape[0]
    post_acc = total_acc / x.shape[0]

    if train:
        back_loss.backward()
        optimiser.step()
        back_loss = back_loss.item()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

    return back_loss, 0, post_acc, all_post

##############################################################################
# VARIABLE PROTONET EPISODE
##############################################################################
def proto_step_var(model, optimiser, loss_fn, x_support, x_query, q_num, y,
               device, n_way, k_shot, q_queries, distance):

    optimiser.zero_grad()
    # Sets up metric tracking
    total_query_loss = 0
    pre_acc_total = 0
    post_acc_total = 0

    all_post = []

    # Kepe track of the last used q_num access idx
    last_query_idx = 0
    # Iterate over num tasks
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

        # Embed all samples
        support_emb = model(x_task_train)
        query_emb = model(x_task_val)

        # Calculates prototypes of support vectors
        prototypes = compute_prototypes(support_emb, n_way, k_shot)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        logits = pairwise_distances(query_emb, prototypes, distance)

        log_p_y = (-logits).log_softmax(dim=1)

        # We scale up the y task val to directly compare for loss but use majority for acc
        sub_q_nums_tens = torch.tensor(sub_q_num).to(device)
        scaled_up_query_y = torch.repeat_interleave(
            y_task_val, sub_q_nums_tens).to(device)

        query_loss = loss_fn(log_p_y, scaled_up_query_y)

        #soft_logits = logits.softmax(dim=1)
        sub_query_pred = (-logits).softmax(dim=1)

        query_pred = majority_vote(sub_query_pred, sub_q_nums_tens).to(
            device, dtype=torch.long)

        post_acc = vote_catagorical_acc(y_task_val, query_pred)
        
        total_query_loss += query_loss.item()
        post_acc_total += post_acc.item()

        all_post.append(post_acc.item())

    back_loss = total_query_loss / x_support.shape[0]
    avg_pre = pre_acc_total/x_support.shape[0]
    avg_post = post_acc_total/x_support.shape[0]

    return back_loss, avg_pre, avg_post, all_post


##############################################################################
# PAIRWISE DISTANCE CALCULATOR
##############################################################################
def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
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

##############################################################################
# PROTOTYPE FUNCTION
##############################################################################
def compute_prototypes(support: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Compute class prototypes from support samples.
    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(n, k, -1).mean(dim=1)
    return class_prototypes

