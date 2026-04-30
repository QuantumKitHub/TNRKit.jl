using TNRKit
using ParallelTestRunner

testsuite = find_tests(@__DIR__)
args = parse_args(ARGS)
ParallelTestRunner.runtests(TNRKit, args)
