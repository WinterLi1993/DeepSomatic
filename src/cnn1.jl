include("OhMyJulia.jl")
include("Fire.jl")
include("Keras.jl")

using OhMyJulia
using Fire
using Keras
using StatsBase
using HDF5

const model = let
    input = Keras.Input(shape=(256, 64, 10))

    output = input |>
        Keras.Convolution2D(32, 5, 5, activation="relu", border_mode="same")  |>
        Keras.MaxPooling2D((2, 2), strides=(2, 2)) |>

        Keras.Convolution2D(64, 5, 5, activation="relu", border_mode="same") |>
        Keras.MaxPooling2D((2, 2), strides=(2, 2)) |>

        Keras.Convolution2D(128, 3, 3, activation="relu", border_mode="same") |>
        Keras.MaxPooling2D((2, 2), strides=(2, 2)) |>

        Keras.Convolution2D(256, 5, 1, activation="relu", border_mode="same") |>
        Keras.Convolution2D(256, 1, 5, activation="relu", border_mode="same") |>
        Keras.MaxPooling2D((4, 1), strides=(4, 1)) |>

        Keras.Convolution2D(256, 3, 3, activation="relu", border_mode="same") |>
        Keras.MaxPooling2D((2, 2), strides=(2, 2)) |>

        Keras.Flatten() |>
        Keras.Dense(256, activation="sigmoid") |>
        Keras.Dense(1, activation="sigmoid")

    Keras.Model(input=input, output=output)
end

const callbacks = [Keras.ModelCheckpoint("cnn1_weight.{epoch:02d}-{val_acc:.4f}.h5", monitor="val_acc", save_weights_only=true)]

const phase_one = readdir(".") ~ filter(x->startswith(x, "cnn1_weight.14"))

function prepare_data()
    images  = readdir(".") ~ filter(x->endswith(x, ".image")) ~ map(i"1:end-6")
    txts    = readdir(".") ~ filter(x->endswith(x, ".txt"))   ~ map(i"1:end-4")
    samples = intersect(images, txts)

    results = map(samples) do sam
        image = open(sam * ".image")
        txt   = open(sam * ".txt")
        txt   = map(split, readlines(txt))
        y     = map(x->parse(f32, x[4]), txt)
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
        X, y
    end

    X = [map(car, results)...;]
    y = [map(cadr, results)...;]

    h5open("cnn1_data.h5", "w") do f
        write(f, "X", X)
        write(f, "y", y)
    end

    X, y
end

@main function train()
    if isfile("cnn1_data.h5")
        X, y = h5open("cnn1_data.h5") do f
            read(f, "X"), read(f, "y")
        end
    else
        prt(STDERR, now(), "preparing data")
        X, y = prepare_data()
    end

    prt(STDERR, now(), "start training")
    if !isempty(phase_one)
        model[:compile](Keras.SGD(lr=1e-3, decay=1e-4), "binary_crossentropy", metrics=["accuracy"])
        model[:load_weights](phase_one[])
        model[:fit](X, y, batch_size=256, nb_epoch=30, validation_split=.01, callbacks=callbacks, initial_epoch=20)
    else
        model[:compile](Keras.Adagrad(lr=.01), "binary_crossentropy", metrics=["accuracy"])
        model[:fit](X, y, batch_size=64, nb_epoch=20, validation_split=.01, callbacks=callbacks)
    end
end
