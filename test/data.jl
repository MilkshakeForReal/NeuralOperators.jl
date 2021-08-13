@testset "get burgers data" begin
    xs, ys = get_burgers_data(n=1000)

    @test size(xs) == (2, 1024, 1000)
    @test size(ys) == (1024, 1000)
end