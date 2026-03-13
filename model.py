import numpy as np
import math


def initialize_matrices(V, d):
    """
    Initializes input and output embedding matrices with small random values.
    V: vocabulary size
    d: embedding dimension

    Returns:
        tuple: (W_in, W_out), each of shape (V, d)

    """
    W_in  = np.random.randn(V, d) * 0.01 
    W_out = np.random.randn(V, d) * 0.01   
    
    return W_in, W_out


def sigmoid(x):
    """
    Computes sigmoid function.

    Returns element-wise sigmoid of input array.
    """
    return 1 / (1 + np.exp(-x))


def forward_pass_batch(center_indices, context_indices, neg_samples, W_in, W_out):
    """
    Forward pass for a batch of Skip-gram pairs with negative sampling.

    Args:
        center_indices (np.array): indices of center words, shape (N,)
        context_indices (np.array): indices of context words, shape (N,)
        neg_samples (np.array): indices of k negative words per center, shape (N, k)
        W_in (np.array): input embedding matrix, shape (V, d)
        W_out (np.array): output embedding matrix, shape (V, d)

    Returns:
        tuple:
            - pos_errors (np.array): sigmoid(dot(v_center, u_context)) - 1, shape (N,)
            - neg_errors (np.array): sigmoid(dot(v_center, u_neg)), shape (N, k)
    """

    v_center = W_in[center_indices]      
    u_context = W_out[context_indices]    
    
    
    pos_scores = np.sum(v_center * u_context, axis=1)  
    pos_errors = sigmoid(pos_scores) - 1              

    
    u_neg = W_out[neg_samples]                         
    v_center_exp = v_center[:, np.newaxis, :]          
    neg_scores = np.sum(u_neg * v_center_exp, axis=2)  
    neg_errors = sigmoid(neg_scores)                  


    return pos_errors, neg_errors


def backward_pass_batch(center_indices, context_indices, neg_samples,
                        W_in, W_out, lr, pos_errors, neg_errors):
    """
    Performs the backward pass and updates W_in and W_out for a batch of Skip-gram pairs.

    Args:
        center_indices (np.array): indices of center words, shape (N,)
        context_indices (np.array): indices of context words, shape (N,)
        neg_samples (np.array): indices of k negative words per center, shape (N, k)
        W_in (np.array): input embedding matrix, shape (V, d)
        W_out (np.array): output embedding matrix, shape (V, d)
        lr (float): learning rate
        pos_errors (np.array): positive pair errors from forward pass, shape (N,)
        neg_errors (np.array): negative pair errors from forward pass, shape (N, k)
    """
    N = len(center_indices)
    v_center = W_in[center_indices].copy()    
    u_context = W_out[context_indices]       
    u_neg = W_out[neg_samples]               

 
    pos_errors_exp = pos_errors[:, np.newaxis]             
    grad_u_context = pos_errors_exp * v_center           
    np.add.at(W_out, context_indices, -lr * grad_u_context)


    neg_errors_exp = neg_errors[:, :, np.newaxis]        
    v_center_exp = v_center[:, np.newaxis, :]              
    grad_u_neg = neg_errors_exp * v_center_exp             
    np.add.at(W_out, neg_samples, -lr * grad_u_neg)


    grad_v_center  = pos_errors_exp * u_context      
    grad_v_center += np.sum(neg_errors_exp * u_neg, axis=1) 
    np.add.at(W_in, center_indices, -lr * grad_v_center)

