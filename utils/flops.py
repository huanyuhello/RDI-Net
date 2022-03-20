import torch

multiply_adds = 1


def count_conv2d(input, output, kernel_size):
    batch_size = input.size(0)
    out_h = output.size(2)
    out_w = output.size(3)
    c_out = output.size(1)
    c_in = input.size(1)
    output_elements = batch_size * out_w * out_h * c_out

    total_ops = output_elements * kernel_size * kernel_size * c_in
    return total_ops


def count_bn(input):
    x = input[0]

    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements

    return total_ops


def count_relu(input):
    x = input[0]

    nelements = x.numel()

    return nelements


def count_softmax(input):
    x = input[0]

    batch_size, nfeatures = x.size()
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return total_ops


def count_linear(input, c_in, c_out):
    batch_size = input.size(0)
    total_mul = c_in
    total_add = c_in - 1
    total_ops = batch_size * (total_mul + total_add) * c_out

    return total_ops


def flops_conv2d(c_in, c_out, out_h, out_w, kernel_size):
    kernel_ops = multiply_adds * kernel_size * kernel_size
    ops_per_element = kernel_ops

    output_elements = out_w * out_h * c_out
    total_ops = output_elements * ops_per_element * c_in

    return total_ops


def flops_linear(c_in, c_out):
    total_mul = c_in
    total_add = c_in - 1
    total_ops = (total_mul + total_add) * c_out

    return total_ops
