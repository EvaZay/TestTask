const SUPPORTED_TYPES = Union{<:Integer, Float32, Float64, Complex{<:Integer}, ComplexF32, ComplexF64}
"""
    ConvolutionalInterleaver(
        N::NType, 
        B::BType, 
        ic::ICType, 
        input_size::NTuple{K, Int64}, 
        input_type::DataType) where {K, NType, BType, ICType}

    Блок сверточного перемежителя

    Аргументы:
     - N -- количество сдвиговых регистров, используемых блоком внутри системы
     - B -- количество дополнительных символов, которые помещаются в каждый последующий сдвиговый регистр, если первый регистр содержит ноль символов
     - ic -- значения, заполняющие каждый регистр сдвига, регистрируются в начале моделирования
     - input_size -- размер входного сигнала (должен быть либо скаляром (), либо вектором (n,))
     - input_type -- тип входного сигнала (должен совпадать с типом элементов ic)

    Поля:
     - register_idx -- индекс сдвигового регистра
     - output -- кэш для выходного значения
     - buffers -- вектор внутренних буферов для всех сдвиговых регистров
     - curr_buffers -- вспомогательный вектор внутренних буферов для всех сдвиговых регистров
"""
mutable struct ConvolutionalInterleaver{OutType, BuffType}
    register_idx::Int64
    output::OutType
    buffers::BuffType
    curr_buffers::BuffType

    function ConvolutionalInterleaver(
        N::NType, 
        B::BType, 
        ic::ICType, 
        input_size::NTuple{K, Int64}=(), 
        input_type::DataType=Float64) where {K, NType, BType, ICType}

        length(input_size) > 1 && throw("The input size must be scalar or vector")

        (isempty(size(N)) && NType <: Real && isinteger(N) && N > 0) || 
            throw("The number of shift register must be a positive integer scalar")

        (isempty(size(B)) && BType <: Real && isinteger(B) && B > 0) || 
            throw("The register length step must be a positive integer scalar")

        input_type <: SUPPORTED_TYPES || 
            throw("The input data type must be double or single precision floating point, \
                boolean, or integer")

        (input_type <: Real && eltype(ICType) <: Complex) && 
            throw("The 'Initial conditions' parameter must be real when the input is real")

        real_input_type = real(input_type)
        (real_input_type <: AbstractFloat || real_input_type == real(eltype(ICType))) || 
            throw("The data type of the `Initial condition` parameter elements must be equal to the input data type")

        (isempty(size(ic)) || size(ic) == (N,)) || 
            throw("The `Initial condition` parameter must either be a finite-valued scalar, or a vector with length \
                equal to the value of the `Rows of shift registers` parameter")

        init = zeros(input_type, N)
        init .= ic

        output = isempty(input_size) ? zero(input_type) : zeros(input_type, input_size)

        buffers = Vector{Vector{input_type}}()
        push!(buffers, zeros(input_type, 0))
        for i in 2:N
            push!(buffers, zeros(input_type, B*(i-1)))
            buffers[i] .= init[i]
        end
        curr_buffers = deepcopy(buffers)

        new{typeof(output), Vector{Vector{input_type}}}(1, output, buffers, curr_buffers)
    end
end

# Вычисление выходного значения
"""
    compute_output(block::ConvolutionalInterleaver, input::Number)

    Функция для подсчета выходного значения блока

    Аргументы:
     - block -- блок сверточного перемежителя
     - input -- скалярный входной сигнал
"""
function compute_output(block::ConvolutionalInterleaver, input::Number)
    if block.register_idx == 1
        block.output = input
    else
        block.output = block.buffers[block.register_idx][end]
    end
    return block.output
end

"""
    compute_output(block::ConvolutionalInterleaver, input::Vector{<:Number})

    Функция для подсчета выходного значения блока

    Аргументы:
     - block -- блок сверточного перемежителя
     - input -- векторный входной сигнал
"""
function compute_output(block::ConvolutionalInterleaver, input::Vector{<:Number})
    curr_idx = block.register_idx
    for i in eachindex(block.curr_buffers)
        block.curr_buffers[i] .= block.buffers[i]
    end
    for i in eachindex(input)
        if curr_idx == 1
            block.output[i] = input[i]
        else
            block.output[i] = block.curr_buffers[curr_idx][end]
        end
        curr_idx = _recalculate!(block.curr_buffers, curr_idx, input[i])
    end
    return block.output
end

# Обновленые внутренних параметров
"""
    recalculate_internal_params!(block::ConvolutionalInterleaver, input::Union{Vector{<:Number}, <:Number})

    Пересчет внутренних параметров

    Аргументы:
     - block -- блок сверточного перемежителя
     - input -- скалярный или векторный входной сигнал
"""
function recalculate_internal_params!(block::ConvolutionalInterleaver, input::Union{Vector{<:Number}, <:Number})
    for i in eachindex(input)
        block.register_idx = _recalculate!(block.buffers, block.register_idx, input[i])
    end
    return nothing
end