"""
    _recalculate!(buffer::Vector{Vector{T}}, idx::Int64, input::T) where T <: Number

    Вспомогательная функция для пересчета внутренних параметров блока

    Аргументы:
     - buffers -- вектор внутренних буферов для всех сдвиговых регистров
     - idx -- текущий индекс сдвигового регистра
     - input -- скалярный входной сигнал блока или элемент векторогого входа блока
"""
function _recalculate!(buffers::Vector{Vector{T}}, idx::Int64, input::T) where T <: Number
    if idx != 1
        _update_buffer!(buffers[idx], input)
    end
    if idx < length(buffers)
        idx += 1
    else
        idx = 1
    end
    return idx
end

"""
    _update_buffer!(buffer::Vector{T}, input::T) where T <: Number

    Вспомогательная функция для обновления буфера

    Аргументы:
     - buffer -- внутренний буфер для конкретного сдвигового регистра
     - input -- скалярный входной сигнал блока или элемент векторогого входа блока
"""
function _update_buffer!(buffer::Vector{T}, input::T) where T <: Number
    for i in length(buffer):-1:2
        buffer[i] = buffer[i-1]
    end
    buffer[1] = input
    return nothing
end

"""
    plot_with_constant_interpolation!(
        t::Vector{Float64}, y::Vector{<:Vector{<:Number}}, save::Bool=false, filename::String="test.png")

    Функция для построения графиков с константной интерполяцией

    Аргументы:
     - t -- вектор времени (ось абсцисс)
     - y -- вектор векторов скалярных значений (ось ординат)
     - save -- флаг для сохранения графика в файл
     - filename -- имя файла
"""
function plot_with_constant_interpolation!(
    t::Vector{Float64}, y::Vector{<:Vector{<:Number}}, save::Bool=false, filename::String="test.png")

    new_t = zeros(eltype(t), length(t)*2-1)
    new_y = [zeros(Float64, 201) for _ in 1:2]

    idx = 2
    new_t[1] = t[1]
    for i in 2:2:length(new_t)
        new_t[i] = t[idx]
        new_t[i+1] = t[idx]
        idx += 1
    end

    for i in eachindex(y)
        idx = 1
        for j in 1:2:length(new_y[i])-1
            new_y[i][j] = y[i][idx]
            new_y[i][j+1] = y[i][idx]
            idx += 1
        end
        new_y[i][end] = y[i][end]
    end
    
    pl = plot(new_t, new_y, label=["original signal" "resulting signal"], lw=2)
    if save
        savefig(pl, filename)
    end
end

"""
    plot_with_constant_interpolation!(
        t::Vector{Float64}, y::Vector{<:Vector{<:Vector{<:Number}}}, save::Bool=false, filename::String="test.png")

    Функция для построения графиков с константной интерполяцией

    Аргументы:
     - t -- вектор времени (ось абсцисс)
     - y -- вектор векторов векторных значений (ось ординат)
     - save -- флаг для сохранения графика в файл
     - filename -- имя файла
"""
function plot_with_constant_interpolation!(
    t::Vector{Float64}, y::Vector{<:Vector{<:Vector{<:Number}}}, save::Bool=false, filename::String="test.png")
    vector_len = length(y[1][1])
    new_t = zeros(eltype(t), length(t)*2-1)
    new_y = [zeros(Float64, length(t)*2-1) for _ in 1:2*vector_len]

    idx = 2
    new_t[1] = t[1]
    for i in 2:2:length(new_t)
        new_t[i] = t[idx]
        new_t[i+1] = t[idx]
        idx += 1
    end
    for i in eachindex(y)
        idx = 1
        for j in 1:2:length(new_y[i])-1
            new_y[i][j] = y[i][idx][1]
            new_y[i][j+1] = y[i][idx][1]
            new_y[i+vector_len][j] = y[i][idx][2]
            new_y[i+vector_len][j+1] = y[i][idx][2]
            idx += 1
        end
        new_y[i][end] = y[i][end][1]
        new_y[i+vector_len][end] = y[i][end][2]
    end
    labels = String[]
    for i in 1:vector_len
        push!(labels, "original signal $i")
        push!(labels, "resulting signal $i")
    end
    pl = plot(new_t, new_y, label=reshape(labels, 1, length(labels)), lw=2, ylim=[-5.0, 12.0])
    if save
        savefig(pl, filename)
    end
end