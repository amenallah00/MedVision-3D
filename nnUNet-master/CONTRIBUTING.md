# Contributing to MedVision 3D

Thank you for your interest in contributing to MedVision 3D.

MedVision 3D is maintained by Amen Allah Ben Abdrabah and is built upon the nnU-Net framework (developed and maintained by researchers at DKFZ and Helmholtz Imaging Applied Computer Vision Lab). There is no dedicated funding or staff for maintaining the repository, and development happens alongside research and teaching responsibilities. Our bandwidth for reviewing external contributions is therefore limited, and review times may be long.

> [!NOTE]
> **Upstream nnU-Net Contributions**
> MedVision 3D is downstream of the original nnU-Net framework. If your contribution, core architectural enhancement, or bug fix applies to the general nnU-Net framework itself, please direct your contributions to the upstream repository at [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).

## General principles

MedVision 3D inherits nnU-Net's design philosophy: it is intentionally designed to be focused, stable, and generally applicable across datasets and use cases.
Contributions should respect this philosophy and should not introduce unnecessary complexity or specialization.

New functionality must either be generally valid across datasets and setups or convincingly benefit a large enough
portion of the user base. We aim to avoid bloating the framework or increasing its complexity further.

## How to contribute

For larger features and refactors, please open a GitHub issue to discuss the idea before starting work.

To submit a contribution to MedVision 3D, fork the repository, make your changes on a branch, and open a pull request.
For contributions targeting core upstream nnU-Net, please submit them directly to the [upstream repository](https://github.com/MIC-DKFZ/nnUNet) and tag @FabianIsensee.

## Bug reports and bug fixes

Bug reports must include a minimal reproducible example. Without a repro, it is usually impossible for us to
investigate issues.

Pull requests fixing bugs should also include a clear reproduction of the issue and an explanation of how the fix
resolves it.

## Performance improvements

If a pull request claims performance improvements, it must include benchmarks demonstrating the effect. The benchmark
setup must be described clearly enough for us to reproduce the results independently. We may run additional
tests ourselves before merging.

## Contributions that are unlikely to be merged

To keep the framework maintainable and the workload manageable on our end, we deprioritize:

- dataset-specific code
- features that only apply to niche setups
- narrow custom architectures or training pipelines
- large refactorings without prior discussion
- small PRs fixing minor typos or formatting issues

## Cleanups we cannot do yet

Sometimes the right simplification would break custom trainers or existing results files, and has to wait for a
major version. Those are collected in [DEFERRED_CLEANUPS.md](DEFERRED_CLEANUPS.md) rather than left in commit
messages. If your change creates that kind of debt, add an entry in the same PR saying what to do, why it is
deferred and what it breaks.

## Final note

We appreciate the effort people invest in improving MedVision 3D!
