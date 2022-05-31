
using Revise
using Test
using MadNLP
using MadNLPTests

n = 10
B = MadNLP.CompactLBFGS(n)

@test B.current_mem == 0

s = ones(n)
y = ones(n)

MadNLP.update_values!(B, s, y)

p = 1
@test B.current_mem == p
@test B.SdotS == B.Sk' * B.Sk
@test size(B) == (n, p)
@test size(B.Sk) == (n, p)
@test size(B.Yk) == (n, p)
@test size(B.Lk) == (p, p)
@test size(B.Dk) == (p, )
@test size(B.Mk) == (p, p)
@test size(B.Tk) == (2*p, 2*p)
@test size(B.U)  == (n, 2*p)

