include("OhMyJulia.jl")
include("BioDataStructures.jl")
include("Fire.jl")
include("Keras.jl")
include("Falcon.jl")

using OhMyJulia
using BioDataStructures
using Fire
using Falcon
using HDF5
using Keras
using Libz

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
    reads = collect(BamLoader(bam))
    index = Dict{String, IntRangeDict{i32, i32}}()
    chr = -2
    local dict::IntRangeDict{i32, i32}
    for (idx, read) in enumerate(bam) @when read.refID >= 0
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
        freq  = depth < 20 ? -1. : let i, counter = 1, fill(0, 6) # ATCGID
            while i < length(seq)
                c = uppercase(seq[i])
                m = findfirst("ATCG+-", c)

                if m > 4
                    counter[m] += 1
                    len, i = parse(seq, i+1, greedy=false)
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
    f = startswith(model, "model1") ? prepare_data1 :
        startswith(model, "model2") ? prepare_data2 :
        startswith(model, "model3") ? prepare_data3 :
        error("unknown model")
    Keras.load_model(model), f
end

function prepare_data1(reads, indices, pos)
    depth = length(indices)
    if depth > 256
        indices = sample(indices, 256, replace=false) |> sort
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

    for i in images
        image[1, i, :, :] = i
    end

    image[1, depth+1:end, :, :] = 0.

    [image[1, :, 1:63, :], image[1, :, [64], :], [freq min(depth, 2048)/2048]]
end

@main funciton predict(model, bam, vcf)
    reads, index   = load_bam(bam)
    model, prepare = load_model(model)

    for line in eachline(split, vcf)
        pos = try parse(i32, line[2]) catch prt(line...); continue end
        chr = line[1]
        if chr in keys(index) && length(rs = index[chr][pos]) >= 64
            prt(line..., model[:predict_on_batch](prepare(reads, rs, pos))[1])
        else
            prt(line..., '.')
        end
    end
end

@main function evaluate(model, bam, vcf, pileup)
    reads, index   = load_bam(bam)
    model, prepare = load_model(model)
    gDNA = load_pileup(pileup)

    for line in eachline(split, vcf)
        pos = try parse(i32, line[2]) catch prt(line...); continue end
        chr = line[1]
        if chr in keys(index) && length(rs = index[chr][pos]) >= 64
            prt(line..., model[:predict_on_batch](prepare(reads, rs, pos))[1])
        else
            prt(line..., '.')
        end
    end
end
