include("OhMyJulia.jl")
include("BioDataStructures.jl")
include("Fire.jl")
include("Keras.jl")
include("Falcon.jl")

using OhMyJulia
using BioDataStructures
using Fire
using Falcon
using Keras
using StatsBase

const base_code = b"NATCG"

function encode_base(b)::Byte
    b == Byte('A') ? 1 :
    b == Byte('T') ? 2 :
    b == Byte('C') ? 3 :
    b == Byte('G') ? 4 : 0
end

function decode_base(b)::Byte
    base_code[b+1]
end

function encode_read(read, center)
    image = fill(f32(0), 64, 10)
    refpos, relpos = read.pos, 1

    fill_pixel(f, args...) = if center - 31 <= refpos <= center + 31
        f(refpos - center + 32, args...)
    end

    pixel_match(offset) = begin
        alt = encode_base(read.seq[relpos])
        if alt != 0
            image[offset, alt] = min(read.qual[relpos], 60) / 60
            image[offset, alt + 6] = 1.
        end
    end

    pixel_mismatch(offset, ref) = begin
        alt = encode_base(read.seq[relpos])
        if alt != 0
            image[offset, alt] = min(read.qual[relpos], 60) / 60
            image[offset, encode_base(ref) + 6] = 1.
        end
    end

    pixel_insert(offset, len) = begin
        image[offset, 5] = min(len, 20) / 20
    end

    pixel_delete(offset, ref) = begin
        image[offset, 6] = 1.
        image[offset, encode_base(ref) + 6] = 1.
    end

    for mut in read.muts
        while relpos < mut.pos
            fill_pixel(pixel_match)
            relpos += 1
            refpos += 1
        end

        if isa(mut, SNP)
            fill_pixel(pixel_mismatch, mut.ref)
            relpos += 1
            refpos += 1
        elseif isa(mut, Insertion)
            fill_pixel(pixel_insert, length(mut.bases))
            relpos += length(mut.bases)
        elseif isa(mut, Deletion)
            for ref in mut.bases
                fill_pixel(pixel_delete, ref)
                refpos += 1
            end
        end
    end

    while relpos < length(read.seq)
        fill_pixel(pixel_match)
        relpos += 1
        refpos += 1
    end

    image[64, 1] = min(read.mapq, 60) / 60
    image[64, 2] = read.flag & 0x0010 == 0
    image[64, 3] = read.flag & 0x0010 != 0
    image[64, 4] = (x = abs(read.tlen)) < 256 ? x / 640 : log(2, min(x, 1048576)) / 20
    image[64, 5] = read.flag & 0x0900 == 0

    image
end

function load_bam(bam)
    bam = BamLoader(bam)
    reads = collect(bam)
    index = Dict{String, IntRangeDict{i32, i32}}()
    chr = i32(-2)
    local dict::IntRangeDict{i32, i32}
    for (idx, read) in enumerate(reads) @when read.refID >= 0
        if read.refID != chr
            chr = read.refID
            index[bam.refs[chr+1] |> car] = dict = IntRangeDict{i32, i32}()
        end

        start = read.pos |> i32
        stop = read.pos + calc_distance(read) - 1 |> i32

        push!(dict[start:stop], i32(idx))
    end
    reads, index
end

function load_pileup(pileup)
    gDNA = Dict{Tuple{String, i32}, Tuple{i32, f32}}()

    for line in eachline(split, pileup)
        chr = line[1]
        pos = parse(i32, line[2])
        depth = parse(i32, line[4])
        depth == 0 && continue
        freq  = let
            i, counter = 1, fill(0, 6) # ATCGID
            while i < length(line[5])
                c = uppercase(line[5][i])
                m = findfirst("ATCG+-", c)

                if m > 4
                    counter[m] += 1
                    len, i = parse(line[5], i+1, greedy=false)
                    i += len
                elseif m > 0
                    counter[m] += 1
                    i += 1
                elseif c == '^'
                    i += 2
                else
                    i += 1
                end
            end
            maximum(counter) / depth
        end
        gDNA[(chr, pos)] = depth, freq
    end

    gDNA
end

function load_model(model)
    f = contains(model, "model1") ? prepare_data1 :
        contains(model, "model2") ? prepare_data2 :
        error("unknown model")
    Keras.load_model(model), f
end

function prepare_data1(reads, indices, pos)
    depth = length(indices)
    if depth > 256
        indices = sample(indices, 256, replace=false) |> sort
        depth = 256
    end

    data = Array{f32}(1, 256, 64, 10)

    for i in 1:depth
        data[1, i, :, :] = encode_read(reads[indices[i]], pos)
    end

    data[1, depth+1:end, :, :] = 0.

    data
end

function prepare_data2(reads, indices, pos)
    depth = length(indices)

    images = map(x->encode_read(x, pos), reads[sort(indices)])

    freq = let counter = fill(0, 6)
        for i in images
            mut = findfirst(i[32, 1:4])
            if mut == 0
                counter[6] += i[31, 6] == 0 && i[32, 6] != 0
            elseif i[32, mut+6] == 0.
                counter[mut] += 1
            end
            if i[33, 5] != 0.
                counter[5] += 1
            end
        end
        maximum(counter) / depth
    end

    if depth > 256
        images = sample(images, 256, replace=false, ordered=true)
    end

    image = Array{f32}(1, 256, 64, 10)

    for (i, img) in enumerate(images)
        image[1, i, :, :] = img
    end

    image[1, depth+1:end, :, :] = 0.

    [image[[1], :, 1:63, :], image[[1], :, [64], :], [freq min(depth, 2048)/2048]]
end

function best_cut(y, ŷ)
    ind = sortperm(ŷ)
    y = y[ind]
    cutoff = findmax(cumsum(.5 .- y)) |> cadr
    correct = sum(1 - y[1:cutoff]) + sum(y[cutoff+1:end])
    ŷ[ind[cutoff]], correct / length(y)
end

function auc(y, ŷ, cut)
    T = y .== 1
    F = !T

    last = 0., 0.

    points = map([0:.0001:.019; .02:.01:.98; .981:.0001:1]) do i
        P  = ŷ .> i
        TP = sum(T & P) / sum(T)
        FP = sum(F & P) / sum(F)

        if i >= cut
            cut = 2
            @printf(STDERR, "%.4f, %.4f *\n", FP, TP)
        elseif last != (FP, TP)
            last = FP, TP
            @printf(STDERR, "%.4f, %.4f\n", FP, TP)
        end

        FP, TP
    end

    ∫ = map(1:length(points)) do i
        x1, y1 = i == 1 ? (1.,1.) : points[i-1]
        x2, y2 = points[i]
        (x1 - x2) * (y1 + y2) / 2
    end

    sum(∫)
end

@main function predict(model, bam, vcf)
    STDERR << now() << " - loading bam" << '\n'
    reads, index = load_bam(bam)

    STDERR << now() << " - loading model" << '\n'
    model, prepare = load_model(model)

    STDERR << now() << " - start predicting" << '\n'
    for line in eachline(split, vcf)
        pos = try parse(i32, line[2]) catch prt(line...); continue end
        depth = parse(i32, split(line[end], ':')[3])
        freq  = parse(f32, split(line[end], ':')[6][1:end-1])
        if depth < 64 || freq < 2 || freq > 48
            prt(line..., '.')
            continue
        end
        rs = index[line[1]][pos]
        prt(line..., model[:predict_on_batch](prepare(reads, rs, pos))[1])
    end
end

@main function evaluate(model, bam, vcf, pileup)
    STDERR << now() << " - loading bam" << '\n'
    reads, index = load_bam(bam)

    STDERR << now() << " - loading model" << '\n'
    model, prepare = load_model(model)

    STDERR << now() << " - loading gDNA" << '\n'
    gDNA = load_pileup(pileup)

    truth, pred, fpred = f32[], f32[], f32[]

    STDERR << now() << " - start evaluating" << '\n'
    for line in eachline(split, vcf)
        pos = try parse(i32, line[2]) catch prt(line...); continue end
        chr = line[1]
        depth = parse(i32, split(line[end], ':')[3])
        freq  = parse(f32, split(line[end], ':')[6][1:end-1]) / 100

        p = depth < 64 ? '.' :
            freq < .02 ? 0. :
            freq > .48 ? 1. :
            model[:predict_on_batch](prepare(reads, index[chr][pos], pos))[1]

        gd, gf = if (chr, pos) in keys(gDNA)
            d, f = gDNA[(chr, pos)]
            d >= 20 ? (d,f) : (d,'.')
        else
            '.', '.'
        end

        prt(line..., gd, gf, p)

        if gf != '.' && !(.01 <= gf <= .24) && p != '.'
            push!(truth, gf > .24 ? 1 : 0)
            push!(pred,  p)
            push!(fpred, freq)
        end
    end

    STDERR << now() << " - start analysis" << '\n'
    cutoff, acc = best_cut(truth, pred)
    fcut, facc  = best_cut(truth, fpred)
    prt(STDERR, cutoff, acc)
    prt(STDERR, fcut, facc)
    prt(STDERR, auc(truth, pred, cutoff))
end
