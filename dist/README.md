# fbm-gpu.exe (Windows, x86-64)

A prebuilt `fbm-gpu` for Windows, so the owner of an AMD card (such as an RX
9060 XT) can run the GPU kernels without installing a compiler. It needs
only the normal AMD Adrenalin driver, which provides OpenCL.

## Provenance

- **Source:** built from commit `8b5b422` of this repository with
  `make gpu-win`. It prints that version in `fbm-gpu.exe report`.
- **Compiler:** mingw-w64, `x86_64-w64-mingw32-gcc (GCC) 13-win32`, Ubuntu
  24.04. Statically linked.
- **Imports:** only `KERNEL32.dll` and `msvcrt.dll`. `OpenCL.dll` is loaded at
  run time from the driver.
- **SHA-256:**
  `9b79677feb4ed121548e01c8db6fd6ca0d5670b35cb464fb8d936f6300f741c7`
- **Tested here** only under Wine with PoCL, a CPU OpenCL: the full
  `fbm-gpu test` suite passes and `mine` finds block 125552. It has not run on
  a real GPU yet. That is what the report below is for.

To rebuild it yourself instead:

```sh
sudo apt install gcc-mingw-w64-x86-64 opencl-headers && make gpu-win
```

The rebuilt file may differ in timestamps.

## What to run on the card

Open a command prompt in the repository folder and run:

```bat
curl -L -o poclbm130302.cl https://raw.githubusercontent.com/ckolivas/cgminer/v3.7.2/poclbm130302.cl
dist\fbm-gpu.exe list
dist\fbm-gpu.exe test --baseline poclbm130302.cl
dist\fbm-gpu.exe report --baseline poclbm130302.cl
```

- The `curl` line is optional. It fetches cgminer's 2013 kernel so the report
  can time it against these kernels. It is not part of this repository.
- `report` takes about 12 minutes. It writes `fbm-gpu-report.txt` and
  `fbm-gpu-kernels.bin`.
- During the sustained part, note the board power from Adrenalin's metrics
  overlay: Windows gives programs no portable way to read it.
- Close games and videos first.
- If any test fails, stop and send the report. A failure on a stable, stock
  card would be a bug.
- Optional: if you have hashcat, `hashcat -b -m 1400` (SHA-256) gives an
  independent calibration of the card's integer rate.
