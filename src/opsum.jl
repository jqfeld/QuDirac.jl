###################
# OpSum/DualOpSum #
###################
abstract type AbsOpSum{P,N,T} <: DiracOp{P,N} end

const OpDict{N,T} = Dict{OpLabel{N},T}

mutable struct OpSum{P,N,T} <: AbsOpSum{P,N,T}
    ptype::P
    dict::OpDict{N,T}
    # OpSum(ptype, dict) = new(ptype, dict)
    OpSum(ptype, dict::OpDict{0}) = error("Cannot construct a 0-factor operator; did you mean to construct a scalar?")
end

OpSum(ptype::P, dict::OpDict{N,T}) where {P,N,T} = OpSum{P,N,T}(ptype, dict)
OpSum(kt::Ket{P,N,A}, br::Bra{P,N,B}) where {P,N,A,B} = OpSum(ptype(kt), cons_outer!(OpDict{N,promote_type(A,B)}(), kt, br))

function cons_outer!(result, kt, br)
    for (k,kc) in dict(kt)
        for (b,bc) in dict(br)
            newc = kc * bc'
            if newc != 0
                result[OpLabel(k, b)] = newc
            end
        end
    end
    return result
end

struct DualOpSum{P,N,T} <: AbsOpSum{P,N,T}
    op::OpSum{P,N,T}
end

# DualOpSum(op::OpSum{P,N,T}) where {P,N,T} = DualOpSum{P,N,T}(op)
DualOpSum(items...) = DualOpSum(OpSum(items...))

Base.convert(::Type{OpSum}, opc::DualOpSum) = eager_ctran(opc.op)
Base.promote_rule(::Type{O}, ::Type{D}) where {O<:OpSum, D<:DualOpSum} = OpSum

######################
# Accessor functions #
######################
dict(op::OpSum) = op.dict
dict(opc::DualOpSum) = dict(opc.op)

ptype(op::OpSum) = op.ptype
ptype(opc::DualOpSum) = ptype(opc.op)

#######################
# Dict-Like Functions #
#######################
Base.eltype(::AbsOpSum{P,N,T}) where {P,N,T} = T

Base.copy(op::OpSum) = OpSum(ptype(op), copy(dict(op)))
Base.copy(opc::DualOpSum) = DualOpSum(copy(opc.op))

Base.similar(op::OpSum, d=similar(dict(op)); P=ptype(op)) = OpSum(P, d)
Base.similar(opc::DualOpSum, d=similar(dict(opc)); P=ptype(opc)) = DualOpSum(P, d)

Base.:(==)(a::OpSum{P,N}, b::OpSum{P,N}) where {P,N} = ptype(a) == ptype(b) && dict(filternz(a)) == dict(filternz(b))
Base.:(==)(a::DualOpSum{P,N}, b::DualOpSum{P,N}) where {P,N} = a.op == b.op
Base.:(==)(a::DiracOp, b::DiracOp) = ==(promote(a,b)...)

Base.hash(op::AbsOpSum) = hash(dict(filternz(op)), hash(ptype(op)))
Base.hash(op::AbsOpSum, h::UInt64) = hash(hash(op), h)

Base.length(op::AbsOpSum) = length(dict(op))

Base.getindex(op::OpSum, label::OpLabel) = op.dict[label]
Base.getindex(op::OpSum, k::StateLabel, b::StateLabel) = op.dict[OpLabel(k,b)]
Base.getindex(opc::DualOpSum, label::OpLabel) = opc.op[label']'
Base.getindex(opc::DualOpSum, k::StateLabel, b::StateLabel) = opc.op[OpLabel(b,k)]'
Base.getindex(op::AbsOpSum, k, b) = op[StateLabel(k), StateLabel(b)]

Base.setindex!(op::OpSum, c, label::OpLabel) = (op.dict[label] = c)
Base.setindex!(op::OpSum, c, k::StateLabel, b::StateLabel) = (op.dict[OpLabel(k,b)] = c)
Base.setindex!(opc::DualOpSum, c, label::OpLabel) = (opc.op[label'] = c')
Base.setindex!(opc::DualOpSum, c, k::StateLabel, b::StateLabel) = (opc.op[OpLabel(b,k)] = c')
Base.setindex!(op::AbsOpSum, c, k, b) = setindex!(op, c, StateLabel(k), StateLabel(b))

Base.haskey(op::OpSum, label::OpLabel) = haskey(dict(op), label)
Base.haskey(opc::DualOpSum, label::OpLabel) = haskey(opc.op, label')
Base.haskey(op::AbsOpSum, k, b) = haskey(op, OpLabel(k, b))

Base.get(op::OpSum, label::OpLabel, default=predict_zero(eltype(op))) = get(dict(op), label, default)
Base.get(opc::DualOpSum, label::OpLabel, default=predict_zero(eltype(opc))) = get(dict(opc), label, default')'
Base.get(op::AbsOpSum, k, b, default=predict_zero(eltype(op))) = get(op, OpLabel(k, b), default)

Base.delete!(op::OpSum, label::OpLabel) = (delete!(dict(op), label); return op)
Base.delete!(opc::DualOpSum, label::OpLabel) = delete!(opc.op, label')
Base.delete!(op::AbsOpSum, k, b) = delete!(op, OpLabel(k, b))

Base.collect(op::OpSum) = collect(dict(op))
Base.collect(opc::DualOpSum{P,N,T}) where {P,N,T} = collect_pairs!(Array(@compat(Tuple{OpLabel{N}, T}), length(opc)), opc)

function collect_pairs!(result, opc::DualOpSum)
    i = 1
    for (k,v) in dict(opc)
        result[i] = ctpair(k,v)
        i += 1
    end
    return result
end

Base.iterate(op::AbsOpSum) = iterate(dict(op))
Base.iterate(op::OpSum, i) = iterate(dict(op), i)

function Base.iterate(opc::DualOpSum, i)
    (k,v), n = iterate(dict(opc), i)
    return ((k',v'), n)
end

# Base.done(op::AbsOpSum, i) = done(dict(op), i)
Base.first(op::AbsOpSum) = iterate(op, iterate(op))

#############
# ctranpose #
#############
eager_ctran(op::OpSum) = similar(op, Dict(collect(op')))

Base.adjoint(op::OpSum) = DualOpSum(op)
Base.adjoint(opc::DualOpSum) = opc.op

#########
# inner #
#########
function inner(br::Bra{P,N,A}, op::OpSum{P,N,B}) where {P,N,A,B}
    prodtype = ptype(op)
    result = StateDict{N, inner_coefftype(br, op)}()
    return Bra(prodtype, inner_load!(result, br, op, prodtype))
end

function inner(op::OpSum{P,N,A}, kt::Ket{P,N,B}) where {P,N,A,B}
    prodtype = ptype(op)
    result = StateDict{N, inner_coefftype(op, kt)}()
    return Ket(prodtype, inner_load!(result, op, kt, prodtype))
end

function inner(a::OpSum{P,N,A}, b::OpSum{P,N,B}) where {P,N,A,B}
    prodtype = ptype(a)
    result = OpDict{N, inner_coefftype(a, b)}()
    return OpSum(prodtype, inner_load!(result, a, b, prodtype))
end

function inner(a::OpSum{P,N,A}, b::DualOpSum{P,N,B}) where {P,N,A,B}
    prodtype = ptype(a)
    result = OpDict{N, inner_coefftype(a, b)}()
    return OpSum(prodtype, inner_load!(result, a, b, prodtype))
end

function inner(a::DualOpSum{P,N,A}, b::OpSum{P,N,B}) where {P,N,A,B}
    prodtype = ptype(a)
    result = OpDict{N, inner_coefftype(a, b)}()
    return OpSum(prodtype, inner_load!(result, a, b, prodtype))
end

inner(br::Bra, opc::DualOpSum) = inner(opc.op, br')'
inner(opc::DualOpSum, kt::Ket) = inner(kt', opc.op)'
inner(a::DualOpSum, b::DualOpSum) = inner(b.op, a.op)'

function inner_load!(result, a::OpSum, b::OpSum, prodtype)
    for (o1,v) in dict(a), (o2,c) in dict(b)
        add_to_dict!(result, 
                     OpLabel(klabel(o1), blabel(o2)),
                     inner_mul(v, c, prodtype, blabel(o1), klabel(o2)))
    end
    return result
end

function inner_load!(result, a::OpSum, b::DualOpSum, prodtype)
    for (o1,v) in dict(a), (o2,c) in dict(b)
        add_to_dict!(result, 
                     OpLabel(klabel(o1), klabel(o2)),
                     inner_mul(v, c', prodtype, blabel(o1), blabel(o2)))
    end
    return result
end

function inner_load!(result, a::DualOpSum, b::OpSum, prodtype)
    for (o1,v) in dict(a), (o2,c) in dict(b)
        add_to_dict!(result,
                     OpLabel(blabel(o1), blabel(o2)),
                     inner_mul(v', c, prodtype, klabel(o1), klabel(o2)))
    end
    return result
end

function inner_load!(result::Dict{K,T}, br::Bra, op::OpSum, prodtype) where {K,T}
    for (o,v) in dict(op)
        add_to_dict!(result, blabel(o), brcoeff(dict(br), prodtype, klabel(o), v, T))
    end
    return result
end

function inner_load!(result::Dict{K,T}, op::OpSum, kt::Ket, prodtype) where {K,T}
    for (o,v) in dict(op)
        add_to_dict!(result, klabel(o), ktcoeff(dict(kt), prodtype, blabel(o), v, T))
    end
    return result
end

function brcoeff(brdict, prodtype, klabel, v, ::Type{T}) where {T}
    coeff = predict_zero(T)
    for (blabel,c) in brdict
        coeff += inner_mul(c', v, prodtype, klabel, blabel) 
    end
    return coeff'
end

function ktcoeff(ktdict, prodtype, blabel, v, ::Type{T}) where {T}
    coeff = predict_zero(T)
    for (klabel,c) in ktdict
        coeff += inner_mul(c, v, prodtype, klabel, blabel)
    end
    return coeff
end

Base.:*(br::Bra, op::DiracOp) = inner(br,op)
Base.:*(op::DiracOp, kt::Ket) = inner(op,kt)
Base.:*(a::DiracOp, b::DiracOp) = inner(a,b)

###################################
# Functional Operator Application #
###################################
struct DualFunc
    f::Function
end

Base.:*(op::Function, kt::Ket) = op(kt)
Base.:*(br::Bra, op::Function) = op(br)
Base.:*(op::DualFunc, kt::Ket) = (kt' * op.f)'
Base.:*(br::Bra, op::DualFunc) = (op.f * br')'

Base.adjoint(f::Function) = DualFunc(f)
Base.adjoint(fc::DualFunc) = fc.f

##############
# act/act_on #
##############
act_on(op::AbsOpSum, br::Bra, i) = act_on(op', br', i)'

# clear up ambiguity warnings
act_on(op::OpSum{P,1}, kt::Ket{P,1}, i) where P = i==1 ? inner(op, kt) : throw(BoundsError()) 
act_on(opc::DualOpSum{P,1}, kt::Ket{P,1}, i) where P = i==1 ? inner(opc, kt) : throw(BoundsError())

function act_on(op::OpSum{P,1,A}, kt::Ket{P,N,B}, i) where {P,N,A,B}
    prodtype = ptype(op)
    result = StateDict{N, inner_coefftype(op, kt)}()
    return Ket(prodtype, act_on_dict!(result, op, kt, i, prodtype))
end

function act_on(op::DualOpSum{P,1,A}, kt::Ket{P,N,B}, i) where {P,N,A,B}
    prodtype = ptype(op)
    result = StateDict{N, inner_coefftype(op, kt)}()
    return Ket(prodtype, act_on_dict!(result, op, kt, i, prodtype))
end

function act_on_dict!(result, op::OpSum, kt::Ket, i, prodtype)
    for (o,c) in dict(op), (k,v) in dict(kt)
        add_to_dict!(result, 
                     setindex(k, klabel(o)[1], i),
                     inner_mul(c, v, prodtype, blabel(o)[1], k[i]))
    end
    return result
end

function act_on_dict!(result, op::DualOpSum, kt::Ket, i, prodtype)
    for (o,c) in dict(op), (k,v) in dict(kt)
        add_to_dict!(result,
                     setindex(k, blabel(o)[1], i),
                     inner_mul(c', v, prodtype, klabel(o)[1], k[i]))
    end
    return result
end

##########
# tensor #
##########
tensor(a::OpSum{P}, b::OpSum{P}) where {P} = OpSum(ptype(a), tensordict(dict(a), dict(b)))
tensor(a::DualOpSum, b::DualOpSum) = tensor(a.opc, b.opc)'
tensor(a::DiracOp, b::DiracOp) = tensor(promote(a,b)...)

Base.:*(kt::Ket, br::Bra) = tensor(kt,br)

###########
# Scaling #
###########
scale!(op::OpSum, c::Number) = (dscale!(dict(op), c); return op)
scale!(c::Number, op::OpSum) = scale!(op, c)
scale!(opc::DualOpSum, c::Number) = DualOpSum(scale!(opc.op, c'))
scale!(c::Number, opc::DualOpSum) = scale!(opc, c)

scale(op::OpSum, c::Number) = similar(op, dscale(dict(op), c))
scale(c::Number, op::OpSum) = scale(op, c)
scale(opc::DualOpSum, c::Number) = DualOpSum(scale(opc.op, c'))
scale(c::Number, opc::DualOpSum) = scale(opc, c)

Base.:*(c::Number, op::DiracOp) = scale(c, op)
Base.:*(op::DiracOp, c::Number) = scale(op, c)
Base.:/(op::DiracOp, c::Number) = scale(op, 1/c)

###########
# + and - #
###########
Base.:-(op::OpSum) = scale(-1, op)
Base.:-(opc::DualOpSum) = DualOpSum(-opc.op)

Base.:+(a::OpSum{P,N}, b::OpSum{P,N}) where {P,N} = similar(b, add_merge(dict(a), dict(b)))
Base.:-(a::OpSum{P,N}, b::OpSum{P,N}) where {P,N} = similar(b, sub_merge(dict(a), dict(b)))

Base.:+(a::DualOpSum{P,N}, b::DualOpSum{P,N}) where {P,N} = DualOpSum(a.op + b.op)
Base.:-(a::DualOpSum{P,N}, b::DualOpSum{P,N}) where {P,N} = DualOpSum(a.op - b.op)

Base.:+(a::DiracOp, b::DiracOp) = +(promote(a,b)...)
Base.:-(a::DiracOp, b::DiracOp) = a + (-b)

#################
# Normalization #
#################
norm(op::OpSum) = sqrt(sum(abs2, values(dict(op))))
norm(opc::DualOpSum) = norm(opc.op)

normalize(op::DiracOp) = scale(1/norm(op), op)
normalize!(op::DiracOp) = scale!(1/norm(op), op)

#########
# Trace #
#########
function LinearAlgebra.tr(op::OpSum)
    result = predict_zero(eltype(op))
    for (o,v) in dict(op)
        if klabel(o)==blabel(o)
            result += v
        end
    end
    return result
end

LinearAlgebra.tr(opc::DualOpSum) = tr(opc.op)'

#################
# Partial trace #
#################
ptrace(op::DiracOp{P,1}, over) where {P} = over == 1 ? tr(op) : throw(BoundsError())
ptrace(op::DiracOp{P,N}, over) where {P,N} = OpSum(ptype(op), ptrace_dict!(OpDict{N-1,eltype(op)}(), op, over))

function ptrace_dict!(result, op::OpSum, over)
    for (o,v) in dict(op)
        if klabel(o)[over] == blabel(o)[over]
            add_to_dict!(result, traceout(o, over), v)
        end
    end
    return result
end

function ptrace_dict!(result, opc::DualOpSum, over)
    for (o,v) in dict(opc)
        if blabel(o)[over] == klabel(o)[over]
            add_to_dict!(result, traceout_dual(o, over), v')
        end
    end
    return result
end

#####################
# Partial Transpose #
#####################
ptranspose(op::DiracOp{P,N}, over) where {P,N} = OpSum(ptype(op), ptrans_dict!(OpDict{N,eltype(op)}(), op, over))

function ptrans_dict!(result, op::OpSum, over)
    for (o,v) in dict(op)
        add_to_dict!(result, ptranspose(o, over), v)
    end
    return result
end

function ptrans_dict!(result, opc::DualOpSum, over)
    for (o,v) in dict(opc)
        add_to_dict!(result, ptranspose_dual(o, over), v')
    end
    return result
end

########################
# Misc. Math Functions #
########################
nfactors(::AbsOpSum{P,N}) where {P,N} = N
xsubspace(op::AbsOpSum, x) = similar(op, filter((k,v)->is_sum_x(k,x), dict(op)))
filternz!(op::AbsOpSum) = (filter!(nzcoeff, dict(op)); return op)
filternz(op::AbsOpSum) = similar(op, filter(nzcoeff, dict(op)))
switch(op::AbsOpSum, i, j) = maplabels(label->switch(label,i,j), op)
permute(op::AbsOpSum, perm::Vector) = maplabels(label->permute(label,perm), op)

purity(op::DiracOp) = tr(op^2)
commute(a::DiracOp, b::DiracOp) = (a*b) - (b*a)
anticommute(a::DiracOp, b::DiracOp) = (a*b) + (b*a)

inner_eval(f, op::DiracOp) = mapcoeffs(x->inner_eval(f,x),op)

function matrep(op::DiracOp, labels)
    T = promote_type(inner_rettype(ptype(op)), eltype(op))
    return T[bra(i) * op * ket(j) for i in labels, j in labels]
end

function matrep(op::DiracOp, labels...)
    iter = product(labels...)
    T = promote_type(inner_rettype(ptype(op)), eltype(op))
    return T[bra(i...) * op * ket(j...) for i in iter, j in iter]
end

function matrep(op::Union{DualFunc, Function}, labels)
    return [bra(i) * op * ket(j) for i in labels, j in labels]
end

function matrep(op::Union{DualFunc, Function}, labels...)
    iter = Iterators.product(labels...)
    return [bra(i...) * op * ket(j...) for i in iter, j in iter]
end

######################
# Printing Functions #
######################
labelrepr(op::OpSum, o::OpLabel, pad) = "$pad$(op[o]) $(ktstr(klabel(o)))$(brstr(blabel(o)))"
labelrepr(opc::DualOpSum, o::OpLabel, pad) = "$pad$(opc[o']) $(ktstr(blabel(o)))$(brstr(klabel(o)))"

Base.summary(op::DiracOp) = "$(typeof(op)) with $(length(op)) operator(s)"
Base.show(io::IO, op::AbsOpSum) = dirac_show(io, op)
# Base.showcompact(io::IO, op::AbsOpSum) = dirac_showcompact(io, op)
Base.repr(op::AbsOpSum) = dirac_repr(op)

export OpSum,
    ptrace,
    ptranspose,
    matrep,
    xsubspace,
    nfactors,
    maplabels!,
    mapcoeffs!,
    mapcoeffs,
    maplabels,
    filternz,
    filternz!,
    purity,
    act_on,
    inner_eval
