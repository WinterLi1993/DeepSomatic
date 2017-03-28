include("OhMyJulia.jl")
include("BioDataStructures.jl")
include("Fire.jl")
include("Falcon.jl")

using OhMyJulia
using BioDataStructures
using Fire
using Falcon
using HDF5
using Libz

function most_significant_mut_count(seq)
    i, counter = 1, fill(0, 6) # ATCGID

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

    maximum(counter)
end

function read_gdna(f)
    channel = Channel{Tuple{String, i32, i32}}(32)
    @schedule begin
        for line in eachline(split, f)
            depth = parse(Int, line[4])

            if depth > 80
                freq = most_significant_mut_count(line[5]) / depth

                if freq < .01
                    put!(channel, (line[1], parse(i32, line[2]), 0))
                elseif freq > .24 || (depth > 200 && freq > .2)
                    put!(channel, (line[1], parse(i32, line[2]), 1))
                end
            end
        end
        close(channel)
    end
    channel
end

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

function read_cfdna(f)
    bam = Bam(f)
    index = make_index(bam)
    bam, index
end

"""
1-31: position before
32: position of interest
33-63: position after
64: properties of whole read

features of position: qual_A, qual_T, qual_C, qual_G, len_I, is_D, ref_is_A, ref_is_T, ref_is_C, ref_is_G
features of whole read: mapping_qual, is_forward, is_reverse, template_length, is_primary, 0, 0, 0, 0, 0
"""
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
            image[offset, alt + 6] = 1.
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

@main function collect_data(bam, pileup, image, txt)
    taskname = basename(bam)

    prt(STDERR, now(), taskname, "start reading bam"); flush(STDERR)
    bam, index = read_cfdna(bam)

    out_image, out_txt = open(image, "w"), open(txt, "w")

    prt(STDERR, now(), taskname, "start hijinking"); flush(STDERR)
    for (chr, pos, genotype) in read_gdna(pileup)
        reads = index[chr][pos]
        depth = length(reads)
        depth < 64 && continue

        reads = bam.reads[reads]
        images = map(x->encode_read(x, pos), reads)

        mut, freq = let counter = fill(0, 6)
            for i in images
                mut = findfirst(i[32, 1:4])
                if mut == 0
                    counter[6] += i[32, 6] != 0
                elseif i[32, mut+6] == 0.
                    counter[mut] += 1
                end
                if i[33, 5] != 0.
                   counter[5] += 1
                end
            end
            support, mut = findmax(counter)
            support == 0 && continue
            b"ATCG+-"[mut], support / depth
        end

        # randomly drop some "easy" samples to save time and memory
        freq < 0.01 && genotype == 0 && rand() < .99 && continue
        freq > 0.24 && genotype == 1 && rand() < .80 && continue

        foreach(image->write(out_image, image), images)
        prt(out_txt, chr, pos, mut, genotype, freq, depth)
    end

    prt(STDERR, now(), taskname, "done"); flush(STDERR)
end
