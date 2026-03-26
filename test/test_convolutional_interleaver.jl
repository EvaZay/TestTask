import TestTask: ConvolutionalInterleaver, compute_output, recalculate_internal_params!

@testset "ConvolutionalInterleaver" begin
    @testset "Test validation" begin
        expected_text = "The input size must be scalar or vector"
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, 0, (1, 5)) end
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, 0, (2, 2, 2)) end

        expected_text = "The number of shift register must be a positive integer scalar"
        @test_throws expected_text begin ConvolutionalInterleaver(-1, 2, 0) end
        @test_throws expected_text begin ConvolutionalInterleaver(0.2, 2, 0) end
        @test_throws expected_text begin ConvolutionalInterleaver(2+im, 2, 0) end
        @test_throws expected_text begin ConvolutionalInterleaver([1.0, 2.0], 2, 0) end

        expected_text = "The register length step must be a positive integer scalar"
        @test_throws expected_text begin ConvolutionalInterleaver(3, -2, 0) end
        @test_throws expected_text begin ConvolutionalInterleaver(3, 0.2, 0) end
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2+im, 0) end
        @test_throws expected_text begin ConvolutionalInterleaver(3, [1.0, 2.0], 0) end

        expected_text = "The input data type must be double or single precision floating point, \
            boolean, or integer"
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, 0, (), Float16) end
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, 0, (), Tuple{Int64, Float64}) end
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, 0, (), Vector{Complex{UInt8}}) end

        expected_text = "The 'Initial conditions' parameter must be real when the input is real"
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, 2+im, (), Float64) end

        expected_text = "The data type of the `Initial condition` parameter elements must be equal to the input data type"
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, 2.2, (), Int64) end

        expected_text = "The `Initial condition` parameter must either be a finite-valued scalar, or a vector with length \
            equal to the value of the `Rows of shift registers` parameter"
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, [1 2; 3 4], (), Float64) end
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, [1 2 3], (), Float64) end
        @test_throws expected_text begin ConvolutionalInterleaver(3, 2, [1, 2, 3, 4, 5], (), Float64) end
    end

    @testset "Test block with scalar real input" begin
        block = ConvolutionalInterleaver(3, 2, Int8(0), (), Int8)
        expected_outputs = Int8[
            0, 0, 0, 3, 0, 0, 6, 1, 0, 9, 4, 0, 12, 7, 2, 15, 10, 5, 18, 13, 8, 21, 16, 11, 24, 
            19, 14, 27, 22, 17, 30, 25, 20, 33, 28, 23, 36, 31, 26, 39, 34, 29, 42, 37, 32, 45, 
            40, 35, 48, 43, 38, 51, 46, 41, 54, 49, 44, 57, 52, 47, 60, 55, 50, 63, 58, 53, 66, 
            61, 56, 69, 64, 59, 72, 67, 62, 75, 70, 65, 78, 73, 68, 81, 76, 71, 84, 79, 74, 87, 
            82, 77, 90, 85, 80, 93, 88, 83, 96, 91, 86, 99, 94]
        inputs = Int8[i for i in 0:100]
        for i in eachindex(inputs)
            @test expected_outputs[i] === compute_output(block, inputs[i])
            recalculate_internal_params!(block, inputs[i])
        end

        block = ConvolutionalInterleaver(3, 2, [-1, -2, -3], (), Float32)
        expected_outputs = Float32[
            0, -2, -3, 3, -2, -3, 6, 1, -3, 9, 4, -3, 12, 7, 2, 15, 10, 5, 18, 13, 8, 21, 16, 11, 24, 
            19, 14, 27, 22, 17, 30, 25, 20, 33, 28, 23, 36, 31, 26, 39, 34, 29, 42, 37, 32, 45, 
            40, 35, 48, 43, 38, 51, 46, 41, 54, 49, 44, 57, 52, 47, 60, 55, 50, 63, 58, 53, 66, 
            61, 56, 69, 64, 59, 72, 67, 62, 75, 70, 65, 78, 73, 68, 81, 76, 71, 84, 79, 74, 87, 
            82, 77, 90, 85, 80, 93, 88, 83, 96, 91, 86, 99, 94]
        inputs = Float32[i for i in 0:100]
        for i in eachindex(inputs)
            @test expected_outputs[i] === compute_output(block, inputs[i])
            recalculate_internal_params!(block, inputs[i])
        end
    end

    @testset "Test block with vector real input" begin
        block = ConvolutionalInterleaver(3, 2, 2.2, (2,), Float64)
        expected_outputs = Vector{Float64}[
            [0.0, 2.2], [2.2, 2.0], [2.2, 2.2], [3.0, 1.0], [2.2, 5.0], [2.0, 2.2], [6.0, 4.0], [1.0, 8.0], [5.0, 3.0], [9.0, 7.0], 
            [4.0, 11.0], [8.0, 6.0], [12.0, 10.0], [7.0, 14.0], [11.0, 9.0], [15.0, 13.0], [10.0, 17.0], [14.0, 12.0], [18.0, 16.0], 
            [13.0, 20.0], [17.0, 15.0], [21.0, 19.0], [16.0, 23.0], [20.0, 18.0], [24.0, 22.0], [19.0, 26.0], [23.0, 21.0], 
            [27.0, 25.0], [22.0, 29.0], [26.0, 24.0], [30.0, 28.0], [25.0, 32.0], [29.0, 27.0], [33.0, 31.0], [28.0, 35.0], 
            [32.0, 30.0], [36.0, 34.0], [31.0, 38.0], [35.0, 33.0], [39.0, 37.0], [34.0, 41.0], [38.0, 36.0], [42.0, 40.0], 
            [37.0, 44.0], [41.0, 39.0], [45.0, 43.0], [40.0, 47.0], [44.0, 42.0], [48.0, 46.0], [43.0, 50.0], [47.0, 45.0], 
            [51.0, 49.0], [46.0, 53.0], [50.0, 48.0], [54.0, 52.0], [49.0, 56.0], [53.0, 51.0], [57.0, 55.0], [52.0, 59.0], 
            [56.0, 54.0], [60.0, 58.0], [55.0, 62.0], [59.0, 57.0], [63.0, 61.0], [58.0, 65.0], [62.0, 60.0], [66.0, 64.0], 
            [61.0, 68.0], [65.0, 63.0], [69.0, 67.0], [64.0, 71.0], [68.0, 66.0], [72.0, 70.0], [67.0, 74.0], [71.0, 69.0], 
            [75.0, 73.0], [70.0, 77.0], [74.0, 72.0], [78.0, 76.0], [73.0, 80.0], [77.0, 75.0], [81.0, 79.0], [76.0, 83.0], 
            [80.0, 78.0], [84.0, 82.0], [79.0, 86.0], [83.0, 81.0], [87.0, 85.0], [82.0, 89.0], [86.0, 84.0], [90.0, 88.0], 
            [85.0, 92.0], [89.0, 87.0], [93.0, 91.0], [88.0, 95.0], [92.0, 90.0], [96.0, 94.0], [91.0, 98.0], [95.0, 93.0], 
            [99.0, 97.0], [94.0, 101.0]]
        inputs = [Float64[i, i+1] for i in 0:100]
        for i in eachindex(inputs)
            @test expected_outputs[i] == compute_output(block, inputs[i])
            recalculate_internal_params!(block, inputs[i])
        end
        
        block = ConvolutionalInterleaver(3, 2, UInt8[1, 2, 3], (2,), UInt8)
        expected_outputs = Vector{UInt8}[
            [0.0, 2], [3, 2.0], [2, 3], [3.0, 1.0], [3, 5.0], [2.0, 3], [6.0, 4.0], [1.0, 8.0], [5.0, 3.0], [9.0, 7.0], 
            [4.0, 11.0], [8.0, 6.0], [12.0, 10.0], [7.0, 14.0], [11.0, 9.0], [15.0, 13.0], [10.0, 17.0], [14.0, 12.0], [18.0, 16.0], 
            [13.0, 20.0], [17.0, 15.0], [21.0, 19.0], [16.0, 23.0], [20.0, 18.0], [24.0, 22.0], [19.0, 26.0], [23.0, 21.0], 
            [27.0, 25.0], [22.0, 29.0], [26.0, 24.0], [30.0, 28.0], [25.0, 32.0], [29.0, 27.0], [33.0, 31.0], [28.0, 35.0], 
            [32.0, 30.0], [36.0, 34.0], [31.0, 38.0], [35.0, 33.0], [39.0, 37.0], [34.0, 41.0], [38.0, 36.0], [42.0, 40.0], 
            [37.0, 44.0], [41.0, 39.0], [45.0, 43.0], [40.0, 47.0], [44.0, 42.0], [48.0, 46.0], [43.0, 50.0], [47.0, 45.0], 
            [51.0, 49.0], [46.0, 53.0], [50.0, 48.0], [54.0, 52.0], [49.0, 56.0], [53.0, 51.0], [57.0, 55.0], [52.0, 59.0], 
            [56.0, 54.0], [60.0, 58.0], [55.0, 62.0], [59.0, 57.0], [63.0, 61.0], [58.0, 65.0], [62.0, 60.0], [66.0, 64.0], 
            [61.0, 68.0], [65.0, 63.0], [69.0, 67.0], [64.0, 71.0], [68.0, 66.0], [72.0, 70.0], [67.0, 74.0], [71.0, 69.0], 
            [75.0, 73.0], [70.0, 77.0], [74.0, 72.0], [78.0, 76.0], [73.0, 80.0], [77.0, 75.0], [81.0, 79.0], [76.0, 83.0], 
            [80.0, 78.0], [84.0, 82.0], [79.0, 86.0], [83.0, 81.0], [87.0, 85.0], [82.0, 89.0], [86.0, 84.0], [90.0, 88.0], 
            [85.0, 92.0], [89.0, 87.0], [93.0, 91.0], [88.0, 95.0], [92.0, 90.0], [96.0, 94.0], [91.0, 98.0], [95.0, 93.0], 
            [99.0, 97.0], [94.0, 101.0]]
        inputs = [UInt8[i, i+1] for i in 0:100]
        for i in eachindex(inputs)
            @test expected_outputs[i] == compute_output(block, inputs[i])
            recalculate_internal_params!(block, inputs[i])
        end
    end

    @testset "Test block with scalar complex input" begin
        block = ConvolutionalInterleaver(3, 2, Int32(0), (), Complex{Int32})
        expected_outputs = Int32[
            0, 0, 0, 3, 0, 0, 6, 1, 0, 9, 4, 0, 12, 7, 2, 15, 10, 5, 18, 13, 8, 21, 16, 11, 24, 
            19, 14, 27, 22, 17, 30, 25, 20, 33, 28, 23, 36, 31, 26, 39, 34, 29, 42, 37, 32, 45, 
            40, 35, 48, 43, 38, 51, 46, 41, 54, 49, 44, 57, 52, 47, 60, 55, 50, 63, 58, 53, 66, 
            61, 56, 69, 64, 59, 72, 67, 62, 75, 70, 65, 78, 73, 68, 81, 76, 71, 84, 79, 74, 87, 
            82, 77, 90, 85, 80, 93, 88, 83, 96, 91, 86, 99, 94]
        inputs = Complex{Int32}[i+i*im for i in 0:100]
        for i in eachindex(inputs)
            @test expected_outputs[i] + expected_outputs[i]*im === compute_output(block, inputs[i])
            recalculate_internal_params!(block, inputs[i])
        end

        block = ConvolutionalInterleaver(3, 2, Complex{UInt16}[1+im, 2+2im, 3+3im], (), Complex{UInt16})
        expected_outputs = UInt16[
            0, 2, 3, 3, 2, 3, 6, 1, 3, 9, 4, 3, 12, 7, 2, 15, 10, 5, 18, 13, 8, 21, 16, 11, 24, 
            19, 14, 27, 22, 17, 30, 25, 20, 33, 28, 23, 36, 31, 26, 39, 34, 29, 42, 37, 32, 45, 
            40, 35, 48, 43, 38, 51, 46, 41, 54, 49, 44, 57, 52, 47, 60, 55, 50, 63, 58, 53, 66, 
            61, 56, 69, 64, 59, 72, 67, 62, 75, 70, 65, 78, 73, 68, 81, 76, 71, 84, 79, 74, 87, 
            82, 77, 90, 85, 80, 93, 88, 83, 96, 91, 86, 99, 94]
        inputs = Complex{UInt16}[i+i*im for i in 0:100]
        for i in eachindex(inputs)
            @test expected_outputs[i] + expected_outputs[i]*im === compute_output(block, inputs[i])
            recalculate_internal_params!(block, inputs[i])
        end
    end

    @testset "Test block with vector complex input" begin
        block = ConvolutionalInterleaver(3, 2, 2.2+2.2im, (2,), ComplexF64)
        expected_outputs = Vector{Float64}[
            [0.0, 2.2], [2.2, 2.0], [2.2, 2.2], [3.0, 1.0], [2.2, 5.0], [2.0, 2.2], [6.0, 4.0], [1.0, 8.0], [5.0, 3.0], [9.0, 7.0], 
            [4.0, 11.0], [8.0, 6.0], [12.0, 10.0], [7.0, 14.0], [11.0, 9.0], [15.0, 13.0], [10.0, 17.0], [14.0, 12.0], [18.0, 16.0], 
            [13.0, 20.0], [17.0, 15.0], [21.0, 19.0], [16.0, 23.0], [20.0, 18.0], [24.0, 22.0], [19.0, 26.0], [23.0, 21.0], 
            [27.0, 25.0], [22.0, 29.0], [26.0, 24.0], [30.0, 28.0], [25.0, 32.0], [29.0, 27.0], [33.0, 31.0], [28.0, 35.0], 
            [32.0, 30.0], [36.0, 34.0], [31.0, 38.0], [35.0, 33.0], [39.0, 37.0], [34.0, 41.0], [38.0, 36.0], [42.0, 40.0], 
            [37.0, 44.0], [41.0, 39.0], [45.0, 43.0], [40.0, 47.0], [44.0, 42.0], [48.0, 46.0], [43.0, 50.0], [47.0, 45.0], 
            [51.0, 49.0], [46.0, 53.0], [50.0, 48.0], [54.0, 52.0], [49.0, 56.0], [53.0, 51.0], [57.0, 55.0], [52.0, 59.0], 
            [56.0, 54.0], [60.0, 58.0], [55.0, 62.0], [59.0, 57.0], [63.0, 61.0], [58.0, 65.0], [62.0, 60.0], [66.0, 64.0], 
            [61.0, 68.0], [65.0, 63.0], [69.0, 67.0], [64.0, 71.0], [68.0, 66.0], [72.0, 70.0], [67.0, 74.0], [71.0, 69.0], 
            [75.0, 73.0], [70.0, 77.0], [74.0, 72.0], [78.0, 76.0], [73.0, 80.0], [77.0, 75.0], [81.0, 79.0], [76.0, 83.0], 
            [80.0, 78.0], [84.0, 82.0], [79.0, 86.0], [83.0, 81.0], [87.0, 85.0], [82.0, 89.0], [86.0, 84.0], [90.0, 88.0], 
            [85.0, 92.0], [89.0, 87.0], [93.0, 91.0], [88.0, 95.0], [92.0, 90.0], [96.0, 94.0], [91.0, 98.0], [95.0, 93.0], 
            [99.0, 97.0], [94.0, 101.0]]
        inputs = [ComplexF64[i+i*im, i+1+(i+1)*im] for i in 0:100]
        for i in eachindex(inputs)
            @test expected_outputs[i] .+ expected_outputs[i] .* im == compute_output(block, inputs[i])
            recalculate_internal_params!(block, inputs[i])
        end
        
        block = ConvolutionalInterleaver(3, 2, Int64[0, 0, 0], (2,), Complex{Int64})
        expected_outputs = Vector{Int64}[
            [0.0, 0], [0, 2.0], [0, 0], [3.0, 1.0], [0, 5.0], [2.0, 0], [6.0, 4.0], [1.0, 8.0], [5.0, 3.0], [9.0, 7.0], 
            [4.0, 11.0], [8.0, 6.0], [12.0, 10.0], [7.0, 14.0], [11.0, 9.0], [15.0, 13.0], [10.0, 17.0], [14.0, 12.0], [18.0, 16.0], 
            [13.0, 20.0], [17.0, 15.0], [21.0, 19.0], [16.0, 23.0], [20.0, 18.0], [24.0, 22.0], [19.0, 26.0], [23.0, 21.0], 
            [27.0, 25.0], [22.0, 29.0], [26.0, 24.0], [30.0, 28.0], [25.0, 32.0], [29.0, 27.0], [33.0, 31.0], [28.0, 35.0], 
            [32.0, 30.0], [36.0, 34.0], [31.0, 38.0], [35.0, 33.0], [39.0, 37.0], [34.0, 41.0], [38.0, 36.0], [42.0, 40.0], 
            [37.0, 44.0], [41.0, 39.0], [45.0, 43.0], [40.0, 47.0], [44.0, 42.0], [48.0, 46.0], [43.0, 50.0], [47.0, 45.0], 
            [51.0, 49.0], [46.0, 53.0], [50.0, 48.0], [54.0, 52.0], [49.0, 56.0], [53.0, 51.0], [57.0, 55.0], [52.0, 59.0], 
            [56.0, 54.0], [60.0, 58.0], [55.0, 62.0], [59.0, 57.0], [63.0, 61.0], [58.0, 65.0], [62.0, 60.0], [66.0, 64.0], 
            [61.0, 68.0], [65.0, 63.0], [69.0, 67.0], [64.0, 71.0], [68.0, 66.0], [72.0, 70.0], [67.0, 74.0], [71.0, 69.0], 
            [75.0, 73.0], [70.0, 77.0], [74.0, 72.0], [78.0, 76.0], [73.0, 80.0], [77.0, 75.0], [81.0, 79.0], [76.0, 83.0], 
            [80.0, 78.0], [84.0, 82.0], [79.0, 86.0], [83.0, 81.0], [87.0, 85.0], [82.0, 89.0], [86.0, 84.0], [90.0, 88.0], 
            [85.0, 92.0], [89.0, 87.0], [93.0, 91.0], [88.0, 95.0], [92.0, 90.0], [96.0, 94.0], [91.0, 98.0], [95.0, 93.0], 
            [99.0, 97.0], [94.0, 101.0]]
        inputs = [Complex{Int64}[i+i*im, i+1+(i+1)*im] for i in 0:100]
        for i in eachindex(inputs)
            @test expected_outputs[i] .+ expected_outputs[i] .* im == compute_output(block, inputs[i])
            recalculate_internal_params!(block, inputs[i])
        end
    end
end