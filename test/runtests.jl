using TestItemRunner
using DiFoG

# Run all tests EXCEPT those tagged with :job
@run_package_tests filter = ti -> !(:job in ti.tags)