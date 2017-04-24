# This file (OhMyJulia.jl) is licensed under the MIT License:

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

module OhMyJulia

export ∞, Bytes, Byte, u8, u16, u32, u64, i32, i64, f32, f64

const ∞ = Inf

typealias Bytes Vector{UInt8}
typealias Byte UInt8

typealias u8  UInt8
typealias u16 UInt16
typealias u32 UInt32
typealias u64 UInt64
typealias i32 Int32
typealias i64 Int64
typealias f32 Float32
typealias f64 Float64

typealias _u8  UInt8
typealias _u16 UInt16
typealias _u32 UInt32
typealias _u64 UInt64
typealias _i32 Int32
typealias _i64 Int64
typealias _f32 Float32
typealias _f64 Float64

export car, cdr, cadr,
       nrow, ncol, void

@inline car(x::ANY)   = x[1]
@inline cdr(x::ANY)   = x[2:end]
@inline cdr(x::Tuple) = Base.tail(x)
@inline cadr(x::ANY)  = x[2]

@inline nrow(x::ANY) = size(x,1)
@inline ncol(x::ANY) = size(x,2)

@inline void(x::ANY...) = nothing

import Base: <<, >>, >>>

<<(x::IO, y) = (print(x, y); x)
<<(x::IO, y::Byte) = (write(x, y); x)
<<(x::IO, y::Bytes) = (write(x, y); x)
<<(x::IO, f::Function) = (f(x); x)

>>(x::IO, y) = read(x, y)
>>(x::IO, f::Function) = f(x)

>>>(x::IO, y) = (read(x, y); x)
>>>(x::IO, f::Function) = (f(x); x)

export prt

prt(xs...) = prt(STDOUT, xs...)
prt(io::IO, xs...) = begin
    lock(io)
    try
        print(io, car(xs))
        for x in cdr(xs)
            print(io, '\t')
            print(io, x)
        end
        print(io, '\n')
    finally
        unlock(io)
    end
end

import Base.eachline

eachline(f::Function, x) = (f(x) for x in eachline(x))

import Base: map, map!, min, max, conv
export Δ, groupby, level_to_edge

map(f::Base.Callable, g::Base.Callable, x) = map(f, map(g, x))
map!(f::Base.Callable, g::Base.Callable, x) = map!(f, map(g, x))

Δ{T}(x::Vector{T}) = Tuple{T,T}[(x[i+1],x[i]) for i in 1:length(x)-1]
Δ(f::Function, x::Vector) = [f(x[i+1],x[i]) for i in 1:length(x)-1]
Δ{T}(f::Function, ::Type{T}, x::Vector) = T[f(x[i+1],x[i]) for i in 1:length(x)-1]

function min(x...; by=identity, lt=<)
    length(x) == 0 ? throw(ArgumentError("min() needs at least one argument")) :
    length(x) == 1 ? car(x) :
    reduce(x) do a,b
        lt(by(a), by(b)) ? a : b
    end
end

function max(x...; by=identity, lt=<)
    length(x) == 0 ? throw(ArgumentError("max() needs at least one argument")) :
    length(x) == 1 ? car(x) :
    reduce(x) do a,b
        lt(by(a), by(b)) ? b : a
    end
end

function groupby(f, op, v0, itr; dict=Dict())
    @assert isimmutable(v0) "v0 is mutable, use lambda instead"

    for i in itr
        key = f(i)
        dict[key] = op(get(dict, key, v0), i)
    end

    dict
end

function groupby(f, op, v0::Base.Callable, itr; dict=Dict())
    for i in itr
        key = f(i)
        if key in keys(dict)
            dict[key] = op(dict[key], i)
        else
            dict[key] = op(v0(), i)
        end
    end

    dict
end

function groupby(f, op, itr; dict=Dict())
    for i in itr
        key = f(i)
        if key in keys(dict)
            dict[key] = op(dict[key], i)
        else
            dict[key] = i
        end
    end

    dict
end

groupby(f, itr) = groupby(f, push!, ()->[], itr)

function level_to_edge(arr)
    result = Tuple{Int, eltype(arr)}[]
    isempty(arr) && return result
    state = arr[1]
    push!(result, (1, state))

    for (i, v) in enumerate(arr)
        if state != v
            state = v
            push!(result, (i, v))
        end
    end

    result
end

function conv(f::Function, A::Vector; kernel::Int=2, stride::Int=1, ret_t::Type=Any)
    A′ = Vector{ret_t}((length(A)-kernel)÷stride+1)
    for i in eachindex(A′)
        offset = (i - 1) * stride
        A′[i]  = f(A[offset+1:offset+kernel]...)
    end
    A′
end

export @~, @i_str, @with, @when, @unless

function curring_call_trans(acc, rest)
    isempty(rest) && return acc

    h = car(rest)

    if isa(h, Symbol)
        h = Expr(:call, h, acc)
    elseif h.head == :call
        if isa(h.args[1], Expr) && h.args[1].head == :quote
            n = h.args[1].args[1]
            h.args[1] = acc
            unshift!(h.args, n)
        else
            push!(h.args, acc)
        end
    elseif h.head == :macrocall
        h = Expr(:call, h, acc)
    else
        throw(ArgumentError("unexpected $h"))
    end

    curring_call_trans(h, cdr(rest))
end

macro ~(name, call...)
    esc(curring_call_trans(name, call))
end

macro i_str(ind)
    ex = parse("x[$ind]")
    ex.args[2] = esc(ex.args[2])
    Expr(:->, :x, ex)
end

"""
a = @with Dict{Int, Int}() do x
    x[2] = 3
end
"""
macro with(exp)
    a = shift!(exp.args)
    b = shift!(exp.args)
    unshift!(exp.args, a)

    esc(quote
        let $(b.args[1].args[1]) = $exp
            $(b.args[2])
            $(b.args[1].args[1])
        end
    end)
end

macro when(exp)
    :( !$(esc(exp)) && continue )
end

macro unless(exp)
    :( $(esc(exp)) && continue )
end


end
