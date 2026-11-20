# Retrieval test (hit@5)

Model used to rewrite: qwen2.5-coder:1.5b

## Summary

| search | questions answered correctly in the top 5 |
|---|---|
| plain | 10 / 15 |
| rewritten | 8 / 15 |
| plain + rewritten | 10 / 15 |

## Per question (number = position of the first correct result, - = not found)

| # | question | expected | plain | rewritten | both |
|---|---|---|---|---|---|
| 1 | copy the contents of one MultiFab into another MultiFab | MultiFab::Copy | 1 | 1 | 1 |
| 2 | compute dst = a*x + b*y for three MultiFabs | LinComb | - | - | - |
| 3 | fill the ghost cells of a MultiFab with a Neumann zero-gradient boundary condition | FillDomainBoundary, AMReX_BCUtil.H | 3 | - | 4 |
| 4 | set every cell of a MultiFab to a constant value | setVal | - | 4 | 4 |
| 5 | add one MultiFab to another | MultiFab::Add | 1 | 2 | 1 |
| 6 | find the largest absolute value in a MultiFab | norm0 | - | - | - |
| 7 | sum all the values in a MultiFab | MultiFab::sum | 1 | 1 | 1 |
| 8 | exchange ghost cell data between neighbouring boxes | FillBoundary | - | - | - |
| 9 | copy data between two MultiFabs that have different box layouts | ParallelCopy | - | - | - |
| 10 | read a number from the inputs file | ParmParse | 1 | 2 | 1 |
| 11 | start up and shut down AMReX in main | Initialize, Finalize | 5 | - | 3 |
| 12 | write a plotfile of a MultiFab | Plotfile | 1 | 4 | 1 |
| 13 | solve a Poisson equation with the multigrid solver | CellPoisson, MLMG, MLPoisson | 4 | 2 | 2 |
| 14 | write a 7-point Laplacian on an Array4 | laplacian | 3 | 1 | 1 |
| 15 | convert a plain C++ triple loop into a loop that runs on CPU and GPU | ParallelFor | 2 | - | - |

## Rewritten queries

1. amrex::MultiFab::copy
2. amrex compute dst = a*x + b*y
3. amrex fill ghost cells neumann zero-gradient boundary condition
4. amrex::MultiFab::setVal
5. amrex::MultiFab::addOne
6. amrex::MultiFab::max_amplitude
7. amrex sum multifab
8. amrex::MultiFab::exchangeGhostCellData
9. amrex::MultiFab::copyDataBetweenBoxes
10. amrex read input
11. amrex start-up-shutdown
12. amrex plotfile multifab
13. amrex solve poisson multigrid
14. amrex::Array4<amrex::Real> laplacian(const amrex::Array4<amrex::Real>& u, const amrex::Box& box)
15. Convert a plain C++ triple loop to a loop that runs on CPU and GPU using AMReX C++ library.
