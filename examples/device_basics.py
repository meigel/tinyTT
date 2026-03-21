import tinytt as tt
import tinytt._backend as tn


def main():
    device = tn.default_device()
    print('default device:', device if device is not None else 'CPU')
    print('default float dtype:', tn.default_float_dtype(device))
    print('supports fp64:', tn.supports_fp64(device))

    full = tn.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tn.default_float_dtype(device), device=device)
    tensor = tt.TT(full, eps=1e-12, device=device, dtype=full.dtype)

    print('dense tensor device:', full.device)
    print('tt core devices:', [str(core.device) for core in tensor.cores])
    print('tt reconstruction:\n', tensor.full().numpy())

    print('set TINYTT_DEVICE=CPU|CUDA|METAL|CL to switch the default backend before import.')
    print('on fp64-limited accelerators, tinyTT may fall back to float32 or CPU-backed operations.')


if __name__ == '__main__':
    main()
