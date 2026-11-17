# Building the engine

The engine is the `engine/` tree: llama.cpp with the qwen-next changes. It builds like llama.cpp with the CUDA backend.

## Tested toolchain

- Windows 11, Visual Studio 2022 (MSVC 19.44), CUDA toolkit 13.3, CMake, Ninja
- GPU: RTX 5080 Laptop (compute capability 12.0)

Other compilers, CUDA versions, GPUs and operating systems were not tested.

## Configure and build

From a "x64 Native Tools Command Prompt for VS 2022" in the repository root:

    cmake -S engine -B engine/build -G Ninja ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 ^
      -DGGML_NATIVE=ON ^
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF

    cmake --build engine/build --target llama-server llama-cli llama-bench

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability (120 for RTX 50-series). `LLAMA_OPENSSL=OFF` builds without the HTTPS client; models must already be on disk. The build takes about four minutes on the test machine.

The binaries land in `engine/build/bin`. `start-qwen-next.ps1` looks there by default; pass `-Bin` or set `QNEXT_BIN` to use another folder.

## Running

    .\start-qwen-next.ps1 -Model <first GGUF shard> -Draft <MTP draft GGUF>

Add `-DryRun` to print the command line and the `QNEXT_*` environment switches without starting the server.

## Notes

- All qwen-next optimizations are off unless their switch is set; with no switches the binary behaves like the seed.
- The memory-lending features rely on CUDA virtual memory management. On Windows, CUDA offers less VRAM than the card reports (15,067 MiB of 16,275 MiB on the test machine), and allocating beyond what CUDA reports as free silently slows the GPU down instead of failing. Keep the default fit margin.
- Close other GPU applications before starting; the profile is sized to use almost all of the VRAM CUDA offers.
