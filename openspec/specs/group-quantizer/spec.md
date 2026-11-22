# group-quantizer Specification

## Purpose
Post-training weight quantization with per-group scales at 8 and 4 bits, symmetric and asymmetric, written from scratch, with measured reconstruction error per layer, as the quantizer for the efficiency experiments and the numerical check of the quantization-error bounds.

## Requirements

### Requirement: Group-wise quantize and dequantize
The quantizer SHALL quantize a 2-D weight matrix in groups of consecutive input features, with one scale per group (and one integer zero point per group in asymmetric mode), at 8 or 4 bits, and SHALL dequantize it back to a float matrix of the same shape. Every reconstructed element SHALL be within half of its group's step size of the original.

#### Scenario: Symmetric round trip
- **WHEN** a random matrix is quantized symmetrically at 4 bits with group size 64 and dequantized
- **THEN** the shape is unchanged, the integers lie in [-8, 7], and every element differs from the original by at most half its group's scale

#### Scenario: Asymmetric round trip
- **WHEN** a matrix whose values are all positive is quantized asymmetrically at 8 bits and dequantized
- **THEN** the integers lie in [0, 255] and every element differs from the original by at most half its group's step size

#### Scenario: More bits, less error
- **WHEN** the same matrix is quantized at 8 and at 4 bits in the same mode
- **THEN** the 8-bit reconstruction error is smaller than the 4-bit one

### Requirement: Degenerate groups and invalid group sizes
A group of all zeros or of one constant value SHALL quantize without NaN or infinity and reconstruct exactly or to within half its step size, and a group size that does not divide the number of input features SHALL raise an error naming both numbers.

#### Scenario: Zero group
- **WHEN** a matrix with an all-zero row is quantized in either mode
- **THEN** that row is reconstructed as zeros and no value is NaN

#### Scenario: Group size does not divide
- **WHEN** a matrix with 100 input features is quantized with group size 64
- **THEN** the call raises an error that names 100 and 64

### Requirement: Per-layer reconstruction error for a model
Quantizing a model SHALL return a copy in which the weight of every linear layer is replaced by its dequantized value, leaving the original model, embeddings, biases and normalisation layers unchanged, together with one row per linear layer holding its name, shape, root-mean-square error, maximum absolute error and relative Frobenius error.

#### Scenario: One row per linear layer
- **WHEN** a model with n linear layers is quantized
- **THEN** the report has n rows and the original model's weights are unchanged

#### Scenario: Eight bits barely changes the model
- **WHEN** a model is quantized at 8 bits and at 4 bits and both are run on the same input
- **THEN** the maximum difference in logits from the original is smaller at 8 bits than at 4 bits

### Requirement: Error follows the uniform-error model
For Gaussian weights the measured mean squared error SHALL agree within 10% with the mean of Δ²/12 over groups, where Δ is each group's step size, and the dot-product error of a row SHALL not exceed the sum over its groups of Δ/2 times the L1 norm of the matching part of the input.

#### Scenario: Mean squared error
- **WHEN** a 512 by 512 Gaussian matrix is quantized at 4 bits with group size 64
- **THEN** the measured mean squared error is within 10% of the mean of Δ²/12 over all groups

#### Scenario: Dot-product bound
- **WHEN** a quantized row is multiplied with an input vector
- **THEN** the absolute difference from the original product is at most the sum over groups of Δ/2 times the group's L1 norm of the input

### Requirement: Storage accounting
The quantizer SHALL report the bytes a packed representation of a quantized matrix needs: the integers at their bit width, plus 16 bits per group scale and, in asymmetric mode, one zero point at the bit width per group.

#### Scenario: Four-bit size
- **WHEN** a 128 by 128 matrix is quantized symmetrically at 4 bits with group size 64
- **THEN** the reported size is 8192 bytes of integers plus 512 bytes of scales
