include("OhMyJulia.jl")
include("Fire.jl")
include("Keras.jl")

using OhMyJulia
using Fire
using Keras
using StatsBase
using HDF5

const model = let
    input1 = Keras.Input(shape=(256, 63, 10))
    input2 = Keras.Input(shape=(256,  1, 10))
    input3 = Keras.Input(shape=(2,))

    conv = input1 |>
        Keras.Convolution2D(16, (1, 3), activation="relu", padding="same") |>
        Keras.MaxPooling2D((1, 3), strides=(1, 3)) |>
        Keras.Convolution2D(32, (1, 5), activation="relu", padding="same") |>
        Keras.MaxPooling2D((1, 3), strides=(1, 3)) |>
        Keras.Convolution2D(32, (1, 7), activation="relu", padding="valid")

    fc = input1 |>
        Keras.Convolution2D(64, (1, 63), activation="relu", padding="valid") |>
        Keras.Convolution2D(32,  (1, 1), activation="relu", padding="valid")

    res = [conv, fc, input2] |>
        Keras.Concatenate(3) |>
        Keras.Convolution2D(32, (1, 1), activation="relu", padding="same") |>
        Keras.Flatten() |>
        Keras.Dense(30, activation="sigmoid")

    output = [res, input3] |>
        Keras.Concatenate(1) |>
        Keras.Dense(32, activation="sigmoid") |>
        Keras.Dense(1, activation="sigmoid")

    Keras.Model(inputs=[input1, input2, input3], outputs=[output])
end

const callbacks = [Keras.ModelCheckpoint("model3_weight.{epoch:02d}-{val_acc:.4f}.h5", monitor="val_acc")]

const phase_one = readdir(".") ~ filter(x->startswith(x, "model3_weight.19"))

function prepare_data()
    images  = readdir(".") ~ filter(x->endswith(x, ".image")) ~ map(i"1:end-6")
    txts    = readdir(".") ~ filter(x->endswith(x, ".txt"))   ~ map(i"1:end-4")
    samples = intersect(images, txts)

    results = map(samples) do sam
        image = open(sam * ".image")
        txt   = open(sam * ".txt")
        txt   = map(split, readlines(txt))
        y     = map(x->parse(f32, x[4]), txt)
        freq  = map(x->parse(f32, x[5]), txt)
        depth = map(x->parse(i32, x[6]), txt)
        X     = Array{f32}(length(y), 256, 64, 10)
        for (i, d) in enumerate(depth)
            reads = [read(image, f32, 64, 10) for i in 1:d]
            if d > 256
                for (j, r) in enumerate(sample(reads, 256, replace=false, ordered=true))
                    X[i, j, :, :] = r
                end
            else
                for (j, r) in enumerate(reads)
                    X[i, j, :, :] = r
                end
                X[i, d+1:end] = 0.
            end
        end
        X[:, :, 1:63, :], X[:, :, [64], :], [freq min.(depth, 2048)./2048], y
    end

    X1 = [map(i"1", results)...;]
    X2 = [map(i"2", results)...;]
    X3 = [map(i"3", results)...;]
    y  = [map(i"4", results)...;]

    h5open("model3_data.h5", "w") do f
        write(f, "X1", X1)
        write(f, "X2", X2)
        write(f, "X3", X3)
        write(f,  "y",  y)
    end

    X1, X2, X3, y
end

@main function train()
    if isfile("model3_data.h5")
        X1, X2, X3, y = h5open("model3_data.h5") do f
            read(f, "X1"), read(f, "X2"), read(f, "X3"), read(f, "y")
        end
    else
        prt(STDERR, now(), "preparing data")
        X1, X2, X3, y = prepare_data()
    end

    prt(STDERR, now(), "start training")
    if !isempty(phase_one)
        model[:compile](Keras.SGD(lr=1e-3, decay=1e-4), "binary_crossentropy", metrics=["accuracy"])
        model[:load_weights](phase_one[])
        model[:fit]([X1, X2, X3], y, batch_size=256, epochs=30, validation_split=.01, callbacks=callbacks, initial_epoch=20)
    else
        model[:compile](Keras.SGD(lr=2e-3, momentum=.95), "binary_crossentropy", metrics=["accuracy"])
        model[:fit]([X1, X2, X3], y, batch_size=64, epochs=20, validation_split=.01, callbacks=callbacks)
    end
end
