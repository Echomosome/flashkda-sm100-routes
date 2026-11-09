# Third-party notices

## FlashKDA

Upstream: https://github.com/MoonshotAI/FlashKDA
Pinned commit: `1ce47ea`

The FlashKDA source tree is not redistributed in this repository.  The fetch
script downloads it, and the generated patches check the six recorded source
hashes before modifying a copy.  The upstream MIT license is reproduced in
`third_party_flashkda_LICENSE.txt`.

## flash-linear-attention (`fla_kda_ref/`)

Upstream: https://github.com/fla-org/flash-linear-attention
Pinned commit: `a3edffc`

The `fla_kda_ref/` directory is a snapshot of the KDA reference implementation,
used only as a numerical reference.  It is MIT licensed; the license text is in
`fla_kda_ref/LICENSE`:

> MIT License
>
> Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

## CUTLASS

Upstream: https://github.com/NVIDIA/cutlass
Pinned commit: `5c149f5`

CUTLASS is fetched as a FlashKDA submodule and is not redistributed here.  See
the upstream license for terms.

## Noto Sans CJK

Not used by the code in this repository.  The original course presentation used
Noto Sans CJK SC under the SIL Open Font License; no font files are
redistributed.
