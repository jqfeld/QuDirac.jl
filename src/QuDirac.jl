module QuDirac
    
    using Compat
    import Base.Iterators.product
    import LinearAlgebra

    if !(v"0.3-" <= VERSION < v"0.4-")
        @warn("QuDirac v0.1 only officially supports the v0.3 release of Julia. Your version of Julia is $VERSION.")
    end
    
    ####################
    # String Constants #
    ####################
    const lang = "\u27E8"
    const rang = "\u27E9"
    const otimes = "\u2297"
    const vdots ="\u205E"

    ##################
    # Abstract Types #
    ##################
    abstract type AbstractInner end
    
    struct UndefinedInner <: AbstractInner end 
    struct KroneckerDelta <: AbstractInner end
    
    abstract type AbstractDirac{P<:AbstractInner,N} end
    abstract type DiracOp{P,N} <: AbstractDirac{P,N} end
    abstract type DiracState{P,N} <: AbstractDirac{P,N} end

    abstract type DiracScalar <: Number end

    #############
    # Functions #
    #############
    # These functions will be in 
    # QuBase when it releases
    tensor() = error("Cannot call tensor function without arguments")
    tensor(s...) = reduce(tensor, s)

    ######################
    # Include Statements #
    ######################
    include("labels.jl")
    include("innerexpr.jl")
    include("state.jl")
    include("opsum.jl")
    include("outerproduct.jl")
    include("printfuncs.jl")
    include("dictfuncs.jl")
    include("mapfuncs.jl")
    include("str_macros.jl")
    
    #################
    # default_inner #
    #################
    # DEFAULT_INNER = KroneckerDelta
   
    DEFAULT_INNER = KroneckerDelta()
    default_inner() = DEFAULT_INNER
    function default_inner!(ptype::T) where T<:AbstractInner 
        global DEFAULT_INNER = ptype
    end

    default_inner!(KroneckerDelta())

    OpSum(dict::Dict) = OpSum(default_inner(), dict)
    Ket(dict::Dict) = Ket(default_inner(), dict)
    ket(label::StateLabel) = Ket(default_inner(), label)
    ket(items...) = Ket(default_inner(), StateLabel(items))


    export AbstractInner,
        UndefinedInner,
        KroneckerDelta,
        default_inner,
        default_inner!,
        AbstractDirac,
        DiracState,
        DiracOp,
        # All functions that conflict 
        # with QuBase should be exported 
        # below:
        tensor,
        commute,
        anticommute,
        normalize,
        normalize!

end # module QuDirac
