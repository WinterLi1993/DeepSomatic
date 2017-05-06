include("OhMyJulia.jl")
include("BioDataStructures.jl")
include("Fire.jl")
include("Falcon.jl")

using OhMyJulia
using BioDataStructures
using Fire
using Falcon
using HDF5

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

function get_chr_anchors(f)
    chr, anchors = "__invalid__", Dict{String, Int}()
    while !eof(f)
        p = position(f)
        line = readline(f)
        startswith(line, chr) && continue
        chr = line[1:findfirst(line, '\t')]
        anchors[chr[1:end-1]] = p
    end
    anchors
end

function read_gdna(f, chr)
    channel = Channel{Tuple{i32, i32}}(32)
    @schedule while true
        line = readline(f) |> split

        line[1] == chr || return close(channel)

        depth = parse(Int, line[4])

        if depth > 80
            freq = most_significant_mut_count(line[5]) / depth

            if freq < .01 && rand(Bool) # drop some to balance the label
                put!(channel, (parse(i32, line[2]), 0))
            elseif freq > .24 || (depth > 200 && freq > .2)
                put!(channel, (parse(i32, line[2]), 1))
            end
        end

        eof(f) && return close(channel)
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

function next_chromosome(bam::BamLoader)
    chr, reads, index = -1, Read[], IntRangeDict{i32, i32}()

    for (idx, read) in enumerate(bam) @when read.refID >= 0
        if chr == -1
            chr = read.refID
        elseif chr != read.refID
            break # loss a read here, but it doesn't matter
        end

        start = read.pos |> i32
        stop = read.pos + calc_distance(read) - 1 |> i32

        push!(reads, read)
        push!(index[start:stop], i32(length(reads)))
    end

    reads, index
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

@main function collect_data(bam, pileup, image, txt)
    pileup, out_image, out_txt = open(pileup), open(image, "w"), open(txt, "w")
    bam = BamLoader(bam)
    anchors = get_chr_anchors(pileup)

    while !eof(bam.handle)
        all_reads, index = next_chromosome(bam)
        chr = car(bam.refs[all_reads[1].refID + 1])

        chr in keys(anchors) ? seek(pileup, anchors[chr]) : continue

        for (pos, genotype) in read_gdna(pileup, chr)
            reads = index[pos]
            depth = length(reads)
            depth < 64 && continue

            reads = all_reads[reads]
            images = map(x->encode_read(x, pos), reads)

            mut, freq = let counter = fill(0, 6)
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
                support, mut = findmax(counter)
                support == 0 && continue
                "ATCG+-"[mut], support / depth
            end

            # randomly drop some "easy" samples to save time and memory
            freq < 0.02 && continue
            freq > 0.48 && continue
            freq < 0.06 && genotype == 0 && rand() < .99 && continue
            freq > 0.24 && genotype == 1 && rand() < .80 && continue

            foreach(image->write(out_image, image), images)
            prt(out_txt, chr, pos, mut, genotype, freq, depth)
        end
    end

    prt(STDERR, now(), "done")
end
