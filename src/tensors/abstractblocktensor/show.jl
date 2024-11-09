# Show
# ----
function Base.show(io::IO, t::AbstractBlockTensorMap)
    summary(io, t)
    get(io, :compact, false) && return nothing
    println(io, ":")
    for (c, b) in TensorKit.blocks(t)
        println(io, "* Block for sector $c:")
        show(io, b)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", t::AbstractBlockTensorMap)
    # header:
    summary(io, t)
    nnz = nonzero_length(t)
    println(
        io, " with ", nnz, " stored entr", isone(nnz) ? "y" : "ies", iszero(nnz) ? "" : ":"
    )

    # body:
    compact = get(io, :compact, false)::Bool
    (iszero(nnz) || compact) && return nothing
    if issparse(t)
        show_braille(io, t)
    else
        show_elements(io, t)
    end

    return nothing
end

function show_elements(io::IO, x::AbstractBlockTensorMap)
    nzind = nonzero_keys(x)
    length(nzind) == 0 && return nothing
    limit = get(io, :limit, false)::Bool
    compact = get(io, :compact, true)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    pads = map(1:ndims(x)) do i
        return ndigits(maximum(getindex.(nzind, i)))
    end
    io = IOContext(io, :compact => compact)
    nz_pairs = sort(vec(collect(nonzero_pairs(x))); by=first)
    for (k, (ind, val)) in enumerate(nz_pairs)
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            println(io, "  ", '[', Base.join(lpad.(Tuple(ind), pads), ","), "]  =  ", val)
        elseif k == half_screen_rows
            println(io, "   ", Base.join(" " .^ pads, " "), "   \u22ee")
        end
    end
end

# adapted from SparseArrays.jl
const brailleBlocks = UInt16['⠁', '⠂', '⠄', '⡀', '⠈', '⠐', '⠠', '⢀']
function show_braille(io::IO, x::AbstractBlockTensorMap)
    m = prod(getindices(size(x), codomainind(x)))
    n = prod(getindices(size(x), domainind(x)))
    reshape_helper = reshape(CartesianIndices((m, n)), size(x))

    # The maximal number of characters we allow to display the matrix
    local maxHeight::Int, maxWidth::Int
    maxHeight = displaysize(io)[1] - 4 # -4 from [Prompt, header, newline after elements, new prompt]
    maxWidth = displaysize(io)[2] ÷ 2

    if get(io, :limit, true) && (m > 4maxHeight || n > 2maxWidth)
        s = min(2maxWidth / n, 4maxHeight / m)
        scaleHeight = floor(Int, s * m)
        scaleWidth = floor(Int, s * n)
    else
        scaleHeight = m
        scaleWidth = n
    end

    # Make sure that the matrix size is big enough to be able to display all
    # the corner border characters
    if scaleHeight < 8
        scaleHeight = 8
    end
    if scaleWidth < 4
        scaleWidth = 4
    end

    brailleGrid = fill(UInt16(10240), (scaleWidth - 1) ÷ 2 + 4, (scaleHeight - 1) ÷ 4 + 1)
    brailleGrid[1, :] .= '⎢'
    brailleGrid[end - 1, :] .= '⎥'
    brailleGrid[1, 1] = '⎡'
    brailleGrid[1, end] = '⎣'
    brailleGrid[end - 1, 1] = '⎤'
    brailleGrid[end - 1, end] = '⎦'
    brailleGrid[end, :] .= '\n'

    rowscale = max(1, scaleHeight - 1) / max(1, m - 1)
    colscale = max(1, scaleWidth - 1) / max(1, n - 1)

    for I′ in nonzero_keys(x)
        I = reshape_helper[I′]
        si = round(Int, (I[1] - 1) * rowscale + 1)
        sj = round(Int, (I[2] - 1) * colscale + 1)

        k = (sj - 1) ÷ 2 + 2
        l = (si - 1) ÷ 4 + 1
        p = ((sj - 1) % 2) * 4 + ((si - 1) % 4 + 1)

        brailleGrid[k, l] |= brailleBlocks[p]
    end

    foreach(c -> print(io, Char(c)), @view brailleGrid[1:(end - 1)])
    return nothing
end
