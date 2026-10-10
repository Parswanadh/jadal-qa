# Pico hardware test report

Date: 2026-10-10
Board: Raspberry Pi Pico (RP2040, non-W), USB ID 2e8a:0005, `/dev/ttyACM0`
MicroPython: 1.29.0 (RPI_PICO build)
Host: Python venv `jadal` with `PYTHONNOUSERSITE=1`. No sudo, no pip. The jadal-qa code was not modified.

## What was verified and what was not

Verified by the agent over USB serial:
- Board identity and MicroPython version.
- Firmware deploy, with a byte-for-byte readback (sha256 matched).
- Heartbeat (HB) at about 1 s, with increasing ticks.
- Acceptance of the LED commands G, A, R and OFF. The HB echo reports the same state after each command.
- FLASH command accepted with no error.
- PING/PONG round trip and clock-offset estimate.

Not verified by the agent:
- Physical LED output. The agent had no camera, so the G/A/R/OFF sequence and the FLASH pattern were not observed. The user reported that they saw the LED show on the board. No per-colour observation is recorded in this report.
- The onboard LED is a single colour. G, A and R are distinguishable only with an external RGB LED.

## Step 0 - access

The serial device was not accessible for about 2.5 minutes (10 polls at 15 s). Access was granted after the user changed the permissions.

```
crw-rw-rw- 1 root dialout 166, 0 Oct 10 11:47 /dev/ttyACM0
ID 2e8a:0005 MicroPython Board in FS mode
```

## Step 1 - identify, LED, fs

```
$ mpremote connect /dev/ttyACM0 exec "import sys,machine; print(sys.implementation); print(sys.platform); print(machine.freq())"
(name='micropython', version=(1, 29, 0, ''), _machine='Raspberry Pi Pico with RP2040', _mpy=4870, _build='RPI_PICO', _thread='unsafe')
rp2
125000000
```

LED: `machine.Pin(25, machine.Pin.OUT)` blinked 3 times (0.3 s on/off). The code ran without exception and printed `blink25 ok`. `Pin('LED')` was not tried.

```
$ mpremote connect /dev/ttyACM0 fs ls
ls :            (empty root, before copy)
```

## Step 2 - deploy firmware

Source: `pico/main.py` (7963 bytes), copied to the test bench (test scripts are not committed to this repo).
The first `fs cp` attempt failed with `TransportError: could not enter raw repl`. The cause was not diagnosed; a retry succeeded.

```
$ mpremote connect /dev/ttyACM0 fs cp ./main.py :main.py
$ mpremote connect /dev/ttyACM0 fs ls
        7963 main.py
$ mpremote connect /dev/ttyACM0 fs cp :main.py ./main_readback.py   # readback
sha256 32f8f003...8fd18b  (local) == 32f8f003...8fd18b  (board readback)
$ mpremote connect /dev/ttyACM0 reset
port back after 1 s; /dev/ttyACM0 present, 2e8a:0005 still enumerated
```

Note: the `main.py` docstring says it targets Pico 2 (RP2350). This board is an RP2040 Pico. GPIO 25 is the onboard LED on both boards, and the firmware ran and answered.

## Step 3 - PicoLink exercise

Scripts and logs (`pico_hw_check.py` / `run1.log`, supplementary `pico_hw_check2.py` / `run2.log`) are not committed to this repo. Outputs are copied below.

```
[OK] connect(): '/dev/ttyACM0'
[OK] status #1: {'connected': False, 'port': '/dev/ttyACM0', 'last_hb_age_s': None, ...}   # no HB yet, first HB is ~1 s after open
[OK] status #2: {'connected': True, 'last_hb_age_s': 0.013, 'last_hb_ticks_ms': 8125, 'last_hb_led': 'OFF'}
[OK] set_led(G) / set_led(A) / set_led(R) / set_led(OFF)   (0.5 s each)
[OK] flash('A5A5A5A5')
[OK] estimate_offset(): n=8 rtt_ms min 14.94 median 15.07 max 20.22
     offset_ms (min-RTT) -8491426.07  median -8491426.53  sync_error_bound_ms 7.47
[OK] disconnect()
unittest test_pico*: Ran 40 tests in 3.562s  OK
```

Supplementary (`run2.log`):

```
status A: last_hb_age_s 0.273, last_hb_ticks_ms 26125, last_hb_led OFF
status B: last_hb_age_s 0.487, last_hb_ticks_ms 27125, last_hb_led OFF
ticks increasing: True | age < 2 s: True
set G -> HB led reports G;  set A -> A;  set R -> R;  set OFF -> OFF
after disconnect connected: False
```

## Interpretation and limits

- Heartbeat: OK. The period is about 1 s, the age is well under 2 s, and the ticks increase. The LED state echoed in HB matches each command, so the firmware accepted all four colours.
- LED physical output: not verified by the agent (see above). The expected sequence is G, A, R, OFF. The FLASH pattern A5A5A5A5 at 100 ms per bit takes about 3.2 s.
- FLASH: the command was accepted with no error. The pattern itself was not observed by the agent.
- `estimate_offset`: the offset (about -8.49e6 ms) is meaningful only as a relative quantity. It is the host monotonic clock compared with the board `ticks_ms`, which starts at the board's reset. Its spread across samples is about 3 ms. The RTT (about 15 ms) is dominated by the reader's 20 ms poll interval, so the `sync_error_bound_ms` of 7.5 ms is conservative. Do not read this as sub-frame sync accuracy. The library docstring says the same.
- The first `status()` right after `connect()` reports disconnected because no HB had arrived yet. This is expected.
- The serial access was denied for about 2.5 minutes before the permission fix (Step 0).
