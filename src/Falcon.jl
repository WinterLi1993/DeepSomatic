# This file (Falcon.jl) is licensed under the MIT License:

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

__precompile__()

module Falcon

using OhMyJulia
using StatsBase
using Libz
using BioDataStructures
import Base: start, next, done, iteratorsize, eltype, view,
             getindex, setindex!, show, ==, hash, write

export SNP, Insertion, Deletion

abstract Mut

immutable SNP <: Mut
    pos::i32
    ref::Byte
    alt::Byte
end

immutable Insertion <: Mut
    pos::i32
    bases::Bytes
end

immutable Deletion <: Mut
    pos::i32
    bases::Bytes
end

==(x::SNP, y::SNP)             = x.pos==y.pos && x.ref==y.ref && x.alt==y.alt
==(x::Insertion, y::Insertion) = x.pos==y.pos && x.bases==y.bases
==(x::Deletion, y::Deletion)   = x.pos==y.pos && x.bases==y.bases
hash(x::SNP, y::u64)           = hash(x.pos, hash(x.ref, hash(x.alt, y)))
hash(x::Insertion, y::u64)     = hash(x.pos, hash(x.bases, y))
hash(x::Deletion, y::u64)      = hash(x.pos, hash(x.bases, y))
show(io::IO, snp::SNP)         = io << "SNP(" << snp.pos << ":" << snp.ref << "->" << snp.alt << ')'
show(io::IO, indel::Insertion) = io << "Insertion(" << indel.pos << ":" << indel.bases << ')'
show(io::IO, indel::Deletion)  = io << "Deletion(" << indel.pos << ":" << indel.bases << ')'

macro advance_cigar()
    esc(quote
        i += 1
        i > length(cigar) && break
        op, len = cigar[i] & 0x0f, cigar[i] >> 4
    end)
end

substring2byte(s::SubString{String}) = s.string.data[s.offset+1]

function reconstruct_mut_by_md(cigar, md, seq)
    md = matchall(r"\d+|[ATCG]|\^[ATCG]+", md)
    i, j, r, p = 1, 1, 0, 1
    op, len = cigar[i] & 0x0f, cigar[i] >> 4
    r = parse(Int, md[j])
    muts = Mut[]

    while true
        # ‘MIDNSHP=X’→‘012345678’
        if op == 0
            if j % 2 == 0 # SNP
                push!(muts, SNP(p, substring2byte(md[j]), seq[p]))
                len -= 1
                j += 1
                r = parse(Int, md[j])
                p += 1
            else # match
                l = min(len, r)
                len -= l
                r -= l
                p += l
                if r == 0
                    j += 1
                end
            end
            len == 0 && @advance_cigar
        elseif op == 1
            push!(muts, Insertion(p, seq[p:p+len-1]))
            p += len
            @advance_cigar
        elseif op == 2
            s = md[j][2:end] |> String
            push!(muts, Deletion(p-1, s.data))
            @advance_cigar
            j += 1
            r = parse(Int, md[j])
        elseif op == 4 # NOTE: `md` doesn't contains info about softcliped bases
            p += len
            @advance_cigar
        elseif op == 5
            @advance_cigar
        else
            error("TODO: cigar op: $op")
        end
    end

    muts
end

export Read, @tag_str, calc_distance, calc_ref_pos, calc_read_pos, map_to_read, map_to_ref, phred_to_prob, prob_to_phred

const seqcode = b"=ACMGRSVTWYHKDBN"

macro tag_str(x)
    reinterpret(u16, x.data)[1]
end

#===
NOTE: about pos of indels:
      relpos of insertion: first base of the insertion
      refpos of insertion: the base before the insertion
      relpos of deletion:  the base before the deletion
      refpos of deletion:  fitst base of the deletion
all positions are 1-based
===#

type Read
    refID::i32
    pos::i32
    mapq::Byte
    flag::u16
    next_refID::i32
    next_pos::i32
    tlen::i32
    qname::String
    cigar::Vector{u32}
    seq::Bytes
    qual::Bytes
    tags::Dict{u16, Any}
    muts::Vector{Mut}
    mate::Read

    function Read(f::IO)
        block_size = f >> i32
        refID      = f >> i32
        pos        = f >> i32
        l_qname    = f >> Byte
        mapq       = f >> Byte
        n_cigar_op = f >>> u16 >> u16
        flag       = f >> u16
        l_seq      = f >> i32
        next_refID = f >> i32
        next_pos   = f >> i32
        tlen       = f >> i32
        qname      = f >> l_qname |> del_end! |> String

        cigar = read(f, u32, n_cigar_op)

        seq = Bytes(l_seq)
        for i in 1:l_seq÷2
            c = f >> Byte
            seq[2i-1] = seqcode[c>>4+1]
            seq[2i] = seqcode[c&0x0f+1]
        end
        if isodd(l_seq)
            seq[l_seq] = seqcode[f>>Byte>>4+1]
        end

        qual = f >> l_seq
        tags = f >> (block_size - 32 - l_qname - 4*n_cigar_op - (l_seq+1)÷2 - l_seq) |> parse_tags

        muts = if n_cigar_op != 0
            if haskey(tags, tag"MD")
                try
                    reconstruct_mut_by_md(cigar, tags[tag"MD"], seq)
                catch
                    Mut[]
                end
            else
                println(STDERR, "no `MD` found, try with --reference")
                Mut[]
            end
        else
            Mut[]
        end

        new(refID, pos+1, mapq, flag, next_refID, next_pos+1, tlen, qname, cigar, seq, qual, tags, muts)
    end
end

function hastag(r::Read, tag::u16)
    tag in r.tags
end

function getindex(r::Read, tag::u16)
    get(r.tags, tag, nothing)
end

function setindex!(r::Read, value, tag)
    r.tags[reinterpret(u16, string(tag).data)[1]] = value
end

function parse_tags(x::Bytes)
    f    = IOBuffer(x)
    tags = Dict{u16, Any}()

    while !eof(f)
        tag = f >> u16
        c   = f >> Byte
        value = c == Byte('Z') ? readuntil(f, '\0') |> del_end! :
                c == Byte('i') ? f >> Int32 :
                c == Byte('I') ? f >> UInt32 :
                c == Byte('c') ? f >> Int8 :
                c == Byte('C') ? f >> UInt8 :
                c == Byte('s') ? f >> Int16 :
                c == Byte('S') ? f >> UInt16 :
                c == Byte('f') ? f >> Float32 :
                c == Byte('A') ? f >> Byte |> Char :
                c == Byte('H') ? error("TODO") :
                c == Byte('B') ? error("TODO") :
                error("unknown tag type $(Char(c))")
        tags[tag] = value
    end

    tags
end

function show(io::IO, r::Read)
    io << r.qname << '\n'

    @printf(io, "ChrID: %-2d  Pos(1-based): %-9d  MapQ(0-60): %-d\n", r.refID,      r.pos,      r.mapq)
    @printf(io, "RNext: %-2d  PNext       : %-9d  TempLength: %-d\n", r.next_refID, r.next_pos, r.tlen)

    io << "Cigar: "
    isempty(r.cigar) ? io << '*' : for i in r.cigar
        io << (i >> 4) << cigarcode[i&0x0f+1]
    end
    @printf(io, "  Flag: %d (", r.flag)
    showflag(io, r.flag)
    io << ")\n"

    io << r.seq << '\n'
    io << (r.qual[1] == 0xff ? '*' : map(x->x+0x21, r.qual)) << '\n'

    for (k,v) in r.tags
        write(io, k)
        io << ':' << tagtype(v) << ':' << v << "  "
    end

    io << '\n'
    for i in r.muts
        io << i << ' '
    end

    io << '\n'
end

function showflag(io::IO, flag::u16)
    is_first = true
    interpunct() = is_first ? (is_first=false; "") : " · "
    flag & 0x0001 != 0 && io << interpunct() << "pair_seq"
    flag & 0x0002 != 0 && io << interpunct() << "aligned"
    flag & 0x0004 != 0 && io << interpunct() << "unmapped"
    flag & 0x0008 != 0 && io << interpunct() << "mate_unmapped"
    flag & 0x0010 != 0 && io << interpunct() << "reverse"
    flag & 0x0020 != 0 && io << interpunct() << "mate_reverse"
    flag & 0x0040 != 0 && io << interpunct() << "r1"
    flag & 0x0080 != 0 && io << interpunct() << "r2"
    flag & 0x0100 != 0 && io << interpunct() << "secondary"
    flag & 0x0200 != 0 && io << interpunct() << "not_pass_filter"
    flag & 0x0400 != 0 && io << interpunct() << "duplicate"
    flag & 0x0800 != 0 && io << interpunct() << "supplementary"
    io
end

"distance between the first and last base in ref"
function calc_distance(r::Read)
    reduce(0, r.cigar) do len, cigar
        ifelse(cigar&0b1101 == 0, len + cigar>>4, len)
    end
end

# return (ref_pos, cigar_op)
function calc_ref_pos(r::Read, relpos::Integer)
    refpos = r.pos - 1
    for cigar in r.cigar
        if cigar & 0x0f == 0
            x = min(cigar >> 4, relpos)
            if relpos == x
                return refpos + x, 0x00
            else
                relpos = relpos - x
                refpos = refpos + x
            end
        elseif cigar & 0x0f == 1
            x = min(cigar >> 4, relpos)
            if relpos == x
                return refpos, 0x01
            else
                relpos = relpos - x
            end
        elseif cigar & 0x0f == 2
            refpos = refpos + cigar >> 4
        elseif cigar & 0x0f == 4
            x = min(cigar >> 4, relpos)
            if relpos == x
                return refpos, 0x04
            else
                relpos = relpos - x
            end
        elseif cigar & 0x0f != 5
            error("TODO: cigar op not supported")
        end
    end
    i32(0), 0xff
end

function calc_read_pos(r::Read, refpos::Integer)
    relpos, refpos = 0, refpos - r.pos + 1
    for cigar in r.cigar
        if cigar & 0x0f == 0
            x = min(cigar >> 4, refpos)
            if refpos == x
                return relpos + x, 0x00
            else
                relpos = relpos + x
                refpos = refpos - x
            end
        elseif cigar & 0x0f == 1
            relpos = relpos + cigar >> 4
        elseif cigar & 0x0f == 2
            x = min(cigar >> 4, refpos)
            if refpos == x
                return relpos, 0x02
            else
                refpos = refpos - x
            end
        elseif cigar & 0x0f == 4
            relpos = relpos + cigar >> 4
        elseif cigar & 0x0f != 5
            error("TODO: cigar op not supported")
        end
    end
    i32(0), 0xff
end

map_to_read(x::SNP, r::Read)       = SNP(car(calc_read_pos(r, x.pos)), x.ref, x.alt)
map_to_read(x::Insertion, r::Read) = Insertion(car(calc_read_pos(r, x.pos))+1, x.bases)
map_to_read(x::Deletion, r::Read)  = Deletion(car(calc_read_pos(r, x.pos)), x.bases)
map_to_ref(x::SNP, r::Read)        = SNP(car(calc_ref_pos(r, x.pos)), x.ref, x.alt)
map_to_ref(x::Insertion, r::Read)  = Insertion(car(calc_ref_pos(r, x.pos)), x.bases)
map_to_ref(x::Deletion, r::Read)   = Deletion(car(calc_ref_pos(r, x.pos))+1, x.bases)

phred_to_prob(x) = 10 ^ (-i64(x) / 10)
prob_to_phred(x) = rount(Byte, -10 * log(10, x))

function del_end!(s::Bytes)
    ccall(:jl_array_del_end, Void, (Any, UInt), s, 1)
    return s
end

function del_end!(s::String)
    ccall(:jl_array_del_end, Void, (Any, UInt), s.data, 1)
    return s
end

const cigarcode = b"MIDNSHP=X"

tagtype(::Char)    = Byte('A')
tagtype(::Integer) = Byte('i')
tagtype(::Float32) = Byte('f')
tagtype(::String)  = Byte('Z')

export Bam, BamLoader

abstract AbstractBam

immutable BamLoader <: AbstractBam
    file::AbstractString
    header_chunk::Bytes
    refs::Vector{Tuple{String, i32}}
    handle::IO

    function BamLoader(file::AbstractString)
        f = open(file) |> ZlibInflateInputStream
        @assert f >> 4 == [0x42, 0x41, 0x4d, 0x01]

        l_text = f >> i32
        text   = f >> l_text
        n_ref  = f >> i32

        refs   = Vector{Tuple{String, i32}}(n_ref)
        for i in 1:n_ref
            l_name  = f >> i32
            name    = f >> (l_name - 1)
            l_ref   = f >>> 1 >> i32
            refs[i] = name, l_ref
        end

        new(file, text, refs, f)
    end
end

start(bl::BamLoader)            = nothing
next(bl::BamLoader, ::Void)     = Read(bl.handle), nothing
done(bl::BamLoader, ::Void)     = eof(bl.handle)
iteratorsize(::Type{BamLoader}) = Base.SizeUnknown()
eltype(::Type{BamLoader})       = Read

show(io::IO, bl::BamLoader) = show(io, "BamLoader($(bl.file))")

immutable Bam <: AbstractBam
    file::AbstractString
    header_chunk::Bytes
    refs::Vector{Tuple{String, i32}}
    reads::Vector{Read}
end

Bam(file::AbstractString) = let bl = BamLoader(file)
    Bam(bl.file, bl.header_chunk, bl.refs, collect(bl))
end

start(bam::Bam)           = start(bam.reads)
next(bam::Bam, x)         = next(bam.reads, x)
done(bam::Bam, x)         = done(bam.reads, x)
iteratorsize(::Type{Bam}) = iteratorsize(Vector{Read})
eltype(::Type{Bam})       = eltype(Vector{Read})

show(io::IO, bam::Bam) = show(io, "Bam($(bam.file))")

export fast_pair!

"only find mates for primary reads (flag & 0x900 == 0)"
function fast_pair!(bam::Bam)
    namedict = Dict{String, Read}()
    for r in bam.reads
        r.flag & 0x900 == 0 || continue
        if r.qname in keys(namedict)
            r.mate = namedict[r.qname]
            namedict[r.qname].mate = r
            delete!(namedict, r.qname)
        else
            namedict[r.qname] = r
        end
    end
    bam
end

export make_index

function make_index(bam::AbstractBam)
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
    index
end

end # module
