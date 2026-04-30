using Test
using TNRKit
using TensorKit
using TensorKitSectors
using QuadGK
using ParallelTestRunner

testsuite = find_tests(@__DIR__)
args = parse_args(ARGS)
ParallelTestRunner.runtests(TNRKit, args)
