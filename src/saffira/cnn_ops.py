# -*- coding: utf-8 -*-
"""
Basic CNN operations for XOHW
"""

import numpy as np
import math
import sys
import random

##### Convolution as matrix multiplication for systolic array #####

# Naive implementation of im2col algorithm, just for test
def im2col_naive(x, window_size):
    rows = []
    for row in range(x.shape[0] - 1):
        for col in range(x.shape[1] - 1):
            window = x[row:(row + window_size), col:(col + window_size)]
            rows.append(window.flatten())
    return np.array(rows)

# TODO: add stride to the view shape calculation
# Maybe we don't need reshape step, might be faster, but then matmul should be rewritten
def im2col(x, window_size, stride = 1):
    """Reshape input matrix using im2col algorithm (using strides)"""
    window_shape = (window_size,window_size)
    view_shape = tuple(np.subtract(x.shape, window_shape) + 1) + window_shape
    
    output_width = window_size * window_size
    output_height = int((x.shape[0] - window_size)/stride + 1)
    output_height = output_height * output_height
    
    x_newview = np.lib.stride_tricks.as_strided(x, view_shape, x.strides*2)
    output = x_newview.reshape((output_height,output_width))
    
    return output

def kernel2matrix(kernels):
    """Unfold all kernels into a single matrix"""
    kernel_size = kernels.shape[0]
    kernel_num = kernels.shape[2]
    output_height = kernel_size * kernel_size
    
    # Stack all unfolded kernels row by row
    output = np.zeros((kernel_num,output_height), dtype=kernels.dtype)
    for idx in range(kernel_num):
        output[idx,:] = kernels[:,:,idx].flatten()
    return np.transpose(output)
    #return output


def matmul(a, b, mac_num):
    """
    Matrix multiplication using systolic array

    Parameters
    ----------
    a : numpy 2d array
        Input matrix
    b : numpy 2d array
        Weight matrix
    mac_num : integer
        Size of the systolic array

    Returns
    -------
    result : numpy 2d array
        A * B
    """
    a_height, a_width = a.shape
    b_height, b_width = b.shape

    if (a_width != b_height):
        raise ValueError("Wrong dimensions")
    
    result = np.zeros((a_height,b_width), dtype=a.dtype)
    steps = math.ceil(b_height/mac_num)
    for i in range(steps):
        start = mac_num*i
        end = start + mac_num if (start + mac_num) < b_height else b_height
        #result = result + hw_matmul(a[:,start:end],b[start:end,:], mac_num)
        result = result + np.dot(a[:,start:end],b[start:end,:])

    return result

def hw_matmul(a, b, mac_num):
    
    a_height, a_width = a.shape
    b_height, b_width = b.shape
    
    o_height = a_height
    o_width = b_width
    
    a_new = a
    b_new = b
    
    # Hardware accelerator expects mac_num*mac_num matrices
    # Add zeros to get required shape
    if (a_width < mac_num):
        a_new = np.c_[a_new, np.zeros((a_height,mac_num-a_width),dtype=a.dtype)]
        a_width = mac_num
    if (a_height < mac_num):
        a_new = np.r_[a_new, np.zeros((mac_num-a_height,a_width),dtype=a.dtype)]
        a_height = mac_num
        
    if (b_width < mac_num):
        b_new = np.c_[b_new, np.zeros((b_height,mac_num-b_width),dtype=b.dtype)]
        b_width = mac_num
    if (b_height < mac_num):
        b_new = np.r_[b_new, np.zeros((mac_num-b_height,b_width),dtype=b.dtype)]
        b_height = mac_num
    
    a_padded = zero_padding(a_new,1)
    b_padded = zero_padding(np.transpose(b_new),1)
    
    a_flatten = a_padded.flatten()
    b_flatten = b_padded.flatten()
    #a_flatten = a_padded.flat
    
    input_length = a_flatten.shape[0]    
    batch = 4
    
    if (input_length % batch) != 0:
        new_length = math.ceil(input_length / batch) * batch
        a_flatten = np.pad(a_flatten, (new_length-input_length,0), 'constant', constant_values=(0,))
        b_flatten = np.pad(b_flatten, (new_length-input_length,0), 'constant', constant_values=(0,))
    
    a_new = a_flatten.view(np.int32)
    b_new = b_flatten.view(np.int32)
    
    # Hardware accelerator considers big-endian order
    if (sys.byteorder == 'little'):
        a_new.byteswap(inplace=True)
        b_new.byteswap(inplace=True)
    
    for b, w in zip(reversed(a_new),reversed(b_new)):
        syst_array_write(b)
        syst_array_write(w)
    
    output = np.zeros((mac_num*mac_num,), dtype=np.int32)
    
    for i in range(mac_num*mac_num):
        output[i] = syst_array_read()
    
    return output.reshape((mac_num,mac_num))[:o_height,:o_width]

       
def syst_array_write(x):
    print(hex(x))
    
def syst_array_read():
    return random.random()


def zero_padding(x, dim):
    """
    Padd x with zeros for the systolic array along the given dimension

    Parameters
    ----------
    x : numpy 2d array
        Input matrix
    dim : integer
        Dimension: 1 for rows, 2 for columns

    Returns
    -------
    output : numpy 2d array
        Input x padded with zeros
    """
    input_size = x.shape[0]
    output_size = input_size * 2 - 1
    output = np.zeros((input_size,output_size), dtype=x.dtype)
    
    if (dim == 1):
        x_new = x
    elif (dim == 2):
        x_new = np.transpose(x)
    else:
        raise ValueError("Wrong dimension parameter: should be either 1 or 2")
    
    for (idx, row) in enumerate(range(input_size),start=1):
        start = input_size - idx
        end = start + input_size
        output[row,start:end] = x_new[row,:]
    
    if (dim == 2):
        output = np.transpose(output)
    
    return output

# TODO: stride + padding
def conv_2d(x, kernel_num, kernels, bias_vector, mac_num, stride = 1):
    """
    Convolution of one input channel with all the kernels

    Parameters
    ----------
    x : numpy 2d array
        Input matrix
    kernel_num : integer
        Number of kernels
    kernels : numpy 3d array
        Kernels stacked together
    bias_vector : numpy 1d array
        Vector of bias values
    mac_num : integer
        Size of systolic array
    stride : integer, optional
        Sliding window stride. The default is 1.

    Returns
    -------
    numpy 3d array
        Results of input x convolution with every kernel
    """
    kernel_size = kernels.shape[0]
    output_width = int((x.shape[0] - kernel_size)/stride + 1)
    output_depth = kernel_num
    
    output = np.zeros((output_width,output_width,output_depth), dtype = np.int32)
    
    # Unfold both input and kernels
    x_new = im2col(x,kernel_size)
    k_new = kernel2matrix(kernels)
    
    x_new_height = x_new.shape[0]
    
    x_steps = math.ceil(x_new_height/mac_num)
    k_steps = math.ceil(kernel_num/mac_num)
    
    for i in range(k_steps):
        k_start = mac_num * i
        k_end = k_start + mac_num if (k_start + mac_num) < kernel_num else kernel_num
        k_batch = k_new[:,k_start:k_end]
        for j in range(x_steps):
            x_start = mac_num * j
            x_end = x_start + mac_num if (x_start + mac_num) < x_new_height else x_new_height
            x_batch = x_new[x_start:x_end,:]
            mm_result = matmul(x_batch, k_batch, mac_num)
            for idx, window in enumerate(range(x_start, x_end)):
                x_idx = math.floor(window / output_width)
                y_idx = window - x_idx * output_width
                output[x_idx,y_idx,k_start:k_end] = mm_result[idx,:]
    
    return output + bias_vector

def conv(x, kernel_num, kernels, bias_vector, mac_num, stride = 1):
    """
    Convolution of input with the kernels

    Parameters
    ----------
    x : numpy 3d array
        Input matrix
    kernel_num : integer
        Number of kernels
    kernels : numpy 3d array
        Kernels stacked together
    bias_vector : numpy 1d array
        Vector of bias values
    mac_num : integer
        Size of systolic array
    stride : integer, optional
        Sliding window stride. The default is 1.

    Returns
    -------
    numpy 3d array
        Results of input x convolution with the kernels
    """
    kernel_size = kernels.shape[0]
    output_width = int((x.shape[0] - kernel_size)/stride + 1)
    output_depth = kernel_num
    
    output = np.zeros((output_width,output_width,output_depth), dtype = np.int32)
    
    input_depth = x.shape[2]
    for c in range(input_depth):
        output = output + conv_2d(x[:,:,c],kernel_num,kernels[:,:,c,:],bias_vector,mac_num,stride)
        
    return output
