using TestItemRunner

DotEnv.load!(joinpath(pkgdir(Thesis), "src", ".env"))

# Run all tests EXCEPT those tagged with :job
@run_package_tests filter = ti -> !(:job in ti.tags)