##############
# StateLabel #
##############
struct StateLabel{N}
    label::NTuple{N}
    hash::UInt64
    StateLabel(label::NTuple{N}) where N = new{N}(label, hash(label)) 
    StateLabel(label::NTuple{N}, h::UInt64) where N = new{N}(label, h) 
end

StateLabel(s::StateLabel) = s
# StateLabel{N}(label::NTuple{N}) = StateLabel{N}(label)
StateLabel(i...) = StateLabel(i)

nfactors(::Type{StateLabel{N}}) where N = N
Base.getindex(s::StateLabel, i) = s.label[i]
Base.getindex(s::StateLabel, arr::AbstractArray) = s.label[arr]

Base.copy(s::StateLabel{N}) where N = StateLabel{N}(s.label, s.hash)
Base.hash(s::StateLabel) = s.hash
Base.hash(s::StateLabel, h::UInt64) = hash(hash(s), h)

Base.:(==)(a::StateLabel{N},b::StateLabel{N}) where N = hash(a) == hash(b)

Base.iterate(s::StateLabel) = iterate(s.label)
Base.iterate(s::StateLabel, i) = iterate(s.label, i)
# Base.done(s::StateLabel, i) = done(s.label, i)

Base.first(s::StateLabel) = first(s.label)
Base.last(s::StateLabel) = last(s.label)
# Base.endof(s::StateLabel) = endof(s.label)

Base.length(::StateLabel{N}) where N = N

Base.map(f::Union{Function,DataType}, s::StateLabel) = StateLabel(map(f, s.label))

tensor(a::StateLabel, b::StateLabel) = StateLabel(tuple(a.label..., b.label...))

labelstr(s::StateLabel) = join(map(repr, s.label), ',')
Base.repr(s::StateLabel) = repr(typeof(s)) * "(" * labelstr(s) * ")"
Base.show(io::IO, s::StateLabel) = print(io, repr(s))

##############
# OpLabel #
##############
struct OpLabel{N}
    k::StateLabel{N}
    b::StateLabel{N}
    hash::UInt64
    OpLabel(k::StateLabel{N}, b::StateLabel{N}) where N = new{N}(k, b, hash(k, hash(b)))
    OpLabel(k::StateLabel{N}, b::StateLabel{N}, h::UInt64) where N = new{N}(k, b, h)
end

OpLabel(op::OpLabel) = op
OpLabel(::StateLabel, ::StateLabel) = error("OpLabel can only be constructed if both StateLabels have the same number of factors")
# OpLabel(k::StateLabel{N}, b::StateLabel{N}) where N = OpLabel{N}(k, b)
OpLabel(k, b) = OpLabel(StateLabel(k), StateLabel(b))

klabel(o::OpLabel) = o.k
blabel(o::OpLabel) = o.b

Base.copy(o::OpLabel{N}) where N = OpLabel{N}(o.k, o.b, o.hash)
Base.hash(o::OpLabel) = o.hash
Base.hash(o::OpLabel, h::UInt64) = hash(hash(o), h)

Base.:(==)(a::OpLabel{N}, b::OpLabel{N})  where N = hash(a) == hash(b)

Base.length(::OpLabel{N}) where N = N

Base.adjoint(o::OpLabel) = OpLabel(o.b, o.k)
tensor(o1::OpLabel, o2::OpLabel) = OpLabel(tensor(o1.k, o2.k), tensor(o1.b, o2.b))

ptranspose(k::StateLabel{N}, b::StateLabel{N}, i) where N = OpLabel{N}(setindex(k, b[i], i), setindex(b, k[i], i))
ptranspose(o::OpLabel, i) = ptranspose(o.k, o.b, i)
ptranspose_dual(o::OpLabel, i) = ptranspose(o.b, o.k, i)

traceout(k::StateLabel, b::StateLabel, i) = OpLabel(except(k, i), except(b, i))
traceout(o::OpLabel, i) = traceout(o.k, o.b, i)
traceout_dual(o::OpLabel, i) = traceout(o.b, o.k, i)

Base.repr(o::OpLabel) = repr(typeof(o)) * "(" * ktstr(o.k) * "," * brstr(o.b) * ")"
Base.show(io::IO, o::OpLabel) = print(io, repr(o))

####################
# Helper Functions #
####################
is_sum_x(o::OpLabel, x) = sum(klabel(o))==sum(blabel(o))==x
is_sum_x(s::StateLabel, x) = sum(s) == x

ctpair(k,v) = (k', v')
nzcoeff(k,v) = v!=0
second(t) = t[2]
except(label::StateLabel, i) = StateLabel(label[1:i-1]..., label[i+1:end]...)
setindex(label::StateLabel, x, y) = StateLabel(label[1:y-1]..., x, label[y+1:end]...)

permute(label::StateLabel{N}, perm::Vector) where N = StateLabel{N}(label[perm])
permute(o::OpLabel, perm::Vector) = OpLabel(permute(o.k, perm), permute(o.b, perm))

function switch!(arr, i, j)
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
    return arr
end
switch(label::StateLabel{N}, i, j) where N =  StateLabel{N}(label[switch!([1:N], i, j)])

switch(o::OpLabel, i, j) = OpLabel(switch(o.k, i, j), switch(o.b, i, j))

export StateLabel,
    OpLabel,
    klabel,
    blabel

