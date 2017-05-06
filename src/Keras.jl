# This file (Keras.jl) is licensed under the MIT License:

# Copyright (c) 2017: Zhang ShiWei.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

module Keras

using PyCall

@pyimport keras.models as models
@pyimport keras.layers as layers
@pyimport keras.layers.merge as merges
@pyimport keras.optimizers as optimizers
@pyimport keras.callbacks as callbacks
@pyimport keras.backend as K

for model in (:Model, :load_model)
    @eval begin
        const $model = models.$model
    end
end

for layer in (
    :Activation, :ActivityRegularization, :AtrousConv2D, :AtrousConvolution2D, :AveragePooling1D, :AveragePooling2D, :AveragePooling3D,
    :BatchNormalization, :Conv1D, :Conv2D, :Conv3D, :Convolution1D, :Convolution2D, :Convolution3D, :Deconv2D, :Deconvolution2D, :Dense,
    :Dropout, :ELU, :Embedding, :Flatten, :GRU, :GaussianDropout, :GaussianNoise, :Highway, :Input, :InputLayer, :InputSpec, :LSTM, :Lambda,
    :LeakyReLU, :LocallyConnected1D, :LocallyConnected2D, :Masking, :MaxPooling1D, :MaxPooling2D, :MaxPooling3D, :MaxoutDense, :Merge, :PReLU,
    :Permute, :RepeatVector, :Reshape, :SeparableConv2D, :SeparableConvolution2D, :SimpleRNN, :ThresholdedReLU, :TimeDistributed,
    :UpSampling1D, :UpSampling2D, :UpSampling3D, :Wrapper, :ZeroPadding1D, :ZeroPadding2D, :ZeroPadding3D
)
    @eval begin
        const $layer = layers.$layer
    end
end

for merge in (:Add, :Average, :Concatenate, :Dot, :Maximum, :Multiply)
    @eval begin
        const $merge = merges.$merge
    end
end

for optimizer in (:Adadelta, :Adagrad, :Adam, :Adamax, :Nadam, :RMSprop, :SGD)
    @eval begin
        const $optimizer = optimizers.$optimizer
    end
end

for callback in (:ProgbarLogger, :ModelCheckpoint, :EarlyStopping, :RemoteMonitor, :LearningRateScheduler, :ReduceLROnPlateau, :CSVLogger, :LambdaCallback)
    @eval begin
        const $callback = callbacks.$callback
    end
end

const dim_ordering = K.image_dim_ordering()

end
