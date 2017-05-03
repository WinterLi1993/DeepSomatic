include("OhMyJulia.jl")
include("Fire.jl")

using OhMyJulia
using Fire

@main function stat()
    images  = readdir(".") ~ filter(x->endswith(x, ".image")) ~ map(i"1:end-6")
    txts    = readdir(".") ~ filter(x->endswith(x, ".txt"))   ~ map(i"1:end-4")
    samples = intersect(images, txts)

    println([x for x in images if !(x in samples)])
    println([x for x in txts   if !(x in samples)])

    genos, freqs = i32[], f32[]

    for sample in samples
        try
            txt = map(split, readlines(sample * ".txt"))
            append!(genos, map(x->parse(i32, x[4]), txt))
            append!(freqs, map(x->parse(f32, x[5]), txt))
        catch e
            println(sample)
            println(e)
        end
    end

    ind = sortperm(freqs)
    freqs = freqs[ind]
    genos = genos[ind]
    cutoff = findmax(cumsum(.5 .- genos)) |> cadr
    correct = sum(1 - genos[1:cutoff]) + sum(genos[cutoff+1:end])
    println("""
        best cutoff frequency: $(freqs[cutoff])
        best accuracy of cutoff: $(correct / length(ind))
        contingency table:
        \t\thigh freq \tlow freq \tsum
        germline\t$(sum(genos[cutoff+1:end]))  \t\t$(sum(genos[1:cutoff])) \t\t$(sum(genos))
        somatic \t$(length(genos[cutoff+1:end])-sum(genos[cutoff+1:end]))  \t\t$(cutoff-sum(genos[1:cutoff])) \t\t$(length(genos)-sum(genos))
        sum     \t$(length(genos[cutoff+1:end]))  \t\t$cutoff \t\t$(length(genos))
        """)
end

@main function clean()
    images  = readdir(".") ~ filter(x->endswith(x, ".image")) ~ map(i"1:end-6")
    txts    = readdir(".") ~ filter(x->endswith(x, ".txt"))   ~ map(i"1:end-4")
    samples = intersect(images, txts)

    run(`rm -rf clean`)
    run(`mkdir clean`)

    results = map(samples) do sample
        image = open(sample * ".image")
        txt   = open(sample * ".txt")
        imageout = open("clean/" * sample * ".image", "w")
        txtout   = open("clean/" * sample * ".txt", "w")

        for line in eachline(txt)
            x = split(line)
            length(x) != 6 && break

            depth = parse(Int, x[6])

            try
                imageout << read(image, sizeof(f32) * 64 * 10 * depth)
                txtout << line
            catch e
                if isa(e, EOFError)
                    break
                else
                    rethrow()
                end
            end
        end
    end
end

@main function check(image, txt)
    image = open(image)

    for (chr, pos, mut, geno, freq, depth) in eachline(split, txt)
        reads = [read(image, f32, 64, 10) for i in 1:parse(Int, depth)]

        rmut, rfreq = let counter = fill(0, 6)
            for i in reads
                m = findfirst(i[32, 1:4])
                if m == 0
                    counter[6] += i[31, 6] == 0 && i[32, 6] != 0
                elseif i[32, m+6] == 0.
                    counter[m] += 1
                end
                if i[33, 5] != 0.
                   counter[5] += 1
                end
            end
            support, m = findmax(counter)
            support == 0 && continue
            "ATCG+-"[m], support / parse(Int, depth)
        end

        prt(mut, freq, rmut, rfreq)
    end
end
