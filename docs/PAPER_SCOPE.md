# Paper-to-release scope

Manuscript title: **Measure2Act: Modular probability transfer for multimodal
aircraft trajectory prediction**.

The central release scope is the probability-assignment interface on an
unchanged five-trajectory support:

1. uncertain source-to-target correspondence;
2. transported source probability;
3. Energy–KL/TPMO refinement using target-specific risk and pairwise distance;
4. fixed-support and auxiliary aircraft-forecast evaluation;
5. reproducibility metadata for the reported chronological airport splits and
   paired seeds.

The candidate source tree contains the paper handoff’s model/operator modules
and supporting protocol implementations. It does not assert that every
historical experiment, external baseline or third-party implementation is part
of the final public release.
