import argparse
import numpy as np
from plyfile import PlyData


def check_array(name, array):
    array = np.asarray(array)

    nan_count = np.isnan(array).sum()
    inf_count = np.isinf(array).sum()
    finite_count = np.isfinite(array).sum()

    print(f"\n{name}")
    print(f"  shape       : {array.shape}")
    print(f"  NaN count   : {nan_count}")
    print(f"  Inf count   : {inf_count}")
    print(f"  finite count: {finite_count}")

    if nan_count > 0:
        indices = np.argwhere(np.isnan(array))
        print(f"  NaN indices : {indices[:20]}")

    if inf_count > 0:
        indices = np.argwhere(np.isinf(array))
        print(f"  Inf indices : {indices[:20]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ply",
        required=True,
        help="Path to point_cloud.ply"
    )
    args = parser.parse_args()

    print(f"Loading: {args.ply}")

    ply = PlyData.read(args.ply)
    vertex = ply["vertex"].data

    print(f"Number of Gaussians: {len(vertex)}")

    properties = vertex.dtype.names

    print("\nProperties:")
    for prop in properties:
        print(f"  {prop}")

    # --------------------------------------------------
    # 各プロパティを検査
    # --------------------------------------------------
    for prop in properties:
        data = np.asarray(vertex[prop])
        check_array(prop, data)

    # --------------------------------------------------
    # Gaussian単位でNaNを持つものを調べる
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Gaussian-level NaN / Inf check")
    print("=" * 60)

    n_gaussians = len(vertex)

    nan_gaussians = np.zeros(n_gaussians, dtype=bool)
    inf_gaussians = np.zeros(n_gaussians, dtype=bool)

    for prop in properties:
        data = np.asarray(vertex[prop])

        if data.ndim == 1:
            nan_gaussians |= np.isnan(data)
            inf_gaussians |= np.isinf(data)
        else:
            nan_gaussians |= np.isnan(data).any(axis=1)
            inf_gaussians |= np.isinf(data).any(axis=1)

    nan_indices = np.where(nan_gaussians)[0]
    inf_indices = np.where(inf_gaussians)[0]

    print(f"\nGaussian containing NaN : {len(nan_indices)}")
    print(f"Gaussian containing Inf : {len(inf_indices)}")

    if len(nan_indices) > 0:
        print("\nNaN Gaussian indices:")
        print(nan_indices[:100])

    if len(inf_indices) > 0:
        print("\nInf Gaussian indices:")
        print(inf_indices[:100])

    # --------------------------------------------------
    # scale specifically
    # --------------------------------------------------
    scale_properties = [
        p for p in properties
        if p.startswith("scale_")
    ]

    if scale_properties:
        scale = np.column_stack([
            np.asarray(vertex[p]) for p in scale_properties
        ])

        scale_nan = np.isnan(scale).any(axis=1)
        scale_inf = np.isinf(scale).any(axis=1)

        print("\n" + "=" * 60)
        print("Scale check")
        print("=" * 60)

        print(f"Scale properties      : {scale_properties}")
        print(f"Gaussians with NaN    : {scale_nan.sum()}")
        print(f"Gaussians with Inf    : {scale_inf.sum()}")

        if scale_nan.any():
            print("NaN scale indices:")
            print(np.where(scale_nan)[0][:100])

        if scale_inf.any():
            print("Inf scale indices:")
            print(np.where(scale_inf)[0][:100])


if __name__ == "__main__":
    main()